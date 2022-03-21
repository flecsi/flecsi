/*
 * Demonstrate how to compute Y = a*X + Y over a distributed array
 * using FleCSI.
 */

#include <flecsi/data.hh>
#include <flecsi/execution.hh>
#include <flecsi/flog.hh>
#include <flecsi/run/control.hh>
#include <flecsi/topo/index.hh>

// In a larger program, this namespace would typically appear in a header file,
// where the inline keywords are necessary.
namespace flaxpy {

// Let the user specify the vector length on the command line.
inline flecsi::program_option<std::size_t> vector_length(
  "Flaxpy-specific Options",
  "length,l",
  "Specify the length of the vectors to add.",
  {{flecsi::option_default, 1000000}});

// Return a vector of the number of indices to assign to each color.
// Indices are divided as evenly as possible among colors.
std::vector<std::size_t>
divide_indices_among_colors(std::size_t ncolors) {
  // Initially assign the same number of indices to each color.
  std::size_t min_indexes_per_color = vector_length.value() / ncolors;
  std::vector<std::size_t> ind_per_col(ncolors, min_indexes_per_color);

  // Evenly distribute the leftover indices among the first colors.
  std::size_t leftovers = vector_length.value() % ncolors;
  for(std::size_t i = 0; i < leftovers; ++i)
    ind_per_col[i]++;
  return ind_per_col;
}

// Define a distributed vector type called dist_vector as a specialization
// of the "user" topology.
struct dist_vector
  : flecsi::topo::specialization<flecsi::topo::user, dist_vector> {
  // Return the number of indices to assign to each color.
  static coloring color() {
    // Specify one color per process, and distribute indices accordingly.
    return divide_indices_among_colors(flecsi::processes());
  }
};

// Define three FleCSI control-point identifiers.
enum class cp { initialize, mul_add, finalize };

// Overload "*" to convert a control-point identifier to a string.
//
// In a larger program in which this function appeared in a header
// file, it could be declared inline.
inline const char *
operator*(cp control_point) {
  switch(control_point) {
    case cp::initialize:
      return "initialize";
    case cp::mul_add:
      return "mul_add";
    case cp::finalize:
      return "finalize";
  }
  flog_fatal("invalid control point");
}

// Define a control policy that specifies that "initialize" should run
// first, then "mul_add", and finally "finalize".
struct control_policy {
  using control_points_enum = cp;
  struct node_policy {};
  template<auto CP>
  using control_point = flecsi::run::control_point<CP>;
  using control_points = std::tuple<control_point<cp::initialize>,
    control_point<cp::mul_add>,
    control_point<cp::finalize>>;
};

// Define a fully qualified control type that implements our control policy.
using control = flecsi::run::control<control_policy>;

} // namespace flaxpy

// For this example, it is supposed that the following declarations
// would not be needed in any further source files so they are given
// internal linkage.
namespace {

// Add two fields, x_field and y_field, to dist_vector.
//
// For clarity we specify flecsi::data::layout::dense as a template
// parameter, but this is in fact the default and would normally be
// omitted.
template<typename T>
using one_field = flecsi::field<T, flecsi::data::layout::dense>;
const one_field<double>::definition<flaxpy::dist_vector> x_field, y_field;

// Declare a coloring of the distributed vector.
flaxpy::dist_vector::slot dist_vector_slot;
flaxpy::dist_vector::cslot dist_vector_cslot;

// Define a task that initializes the elements of the distributed vector.
void
initialize_vectors_task(one_field<double>::accessor<flecsi::wo> x_acc,
  one_field<double>::accessor<flecsi::wo> y_acc) {
  // Compute our starting offset into the global vector.
  std::vector<std::size_t> num_indices_per_color =
    flaxpy::divide_indices_among_colors(flecsi::colors());
  std::size_t current_color = flecsi::color();
  std::size_t base = 0;
  for(std::size_t i = 0; i < current_color; ++i)
    base += num_indices_per_color[i];

  // Arbitrarily initialize x[i] = i and y[i] = 0.  We use a forall
  // for the latter because it can run in parallel without access to
  // the index variable.
  std::size_t num_local_elts =
    num_indices_per_color[current_color]; // Same as x_acc.span().size()
  for(size_t i = 0; i < num_local_elts; ++i)
    x_acc[i] = double(base + i);
  forall(elt, y_acc.span(), "init_y") { elt = 0; };
}

// Implement an action for the initialize control point.
int
initialize_action() {
  dist_vector_cslot.allocate();
  dist_vector_slot.allocate(dist_vector_cslot.get());
  flecsi::execute<initialize_vectors_task>(
    x_field(dist_vector_slot), y_field(dist_vector_slot));
  return 0;
}

// Define a task that assigns Y <- a*X + Y.
void
mul_add_task(double a,
  one_field<double>::accessor<flecsi::ro> x_acc,
  one_field<double>::accessor<flecsi::rw> y_acc) {
  std::size_t num_local_elts = x_acc.span().size();
  for(std::size_t i = 0; i < num_local_elts; ++i)
    y_acc[i] += a * x_acc[i];
}

// Implement an action for the mul_add control point.
int
mul_add_action() {
  const double a = 12.34; // Arbitrary scalar value to multiply
  flecsi::execute<mul_add_task>(
    a, x_field(dist_vector_slot), y_field(dist_vector_slot));
  return 0;
}

// Define a task that adds up all values of Y and returns the sum.
double
reduce_y_task(one_field<double>::accessor<flecsi::rw> y_acc) {
  auto local_sum = reduceall(elt,
    accum,
    y_acc.span(),
    flecsi::exec::fold::sum,
    double,
    "reduce_y_task") {
    accum(elt);
  };
  return local_sum;
}

// Implement an action for the finalize control point.
int
finalize_action() {
  double sum = flecsi::reduce<reduce_y_task, flecsi::exec::fold::sum>(
    y_field(dist_vector_slot))
                 .get();
  flog(info) << "The sum over all elements in the final vector is " << sum
             << std::endl;
  dist_vector_slot.deallocate();
  dist_vector_cslot.deallocate();
  return 0;
}

// Register each of the preceding actions with its eponymous control point.
// None of the variables declared below are ever used; they exist only for
// the side effects induced by declaration.
flaxpy::control::action<initialize_action, flaxpy::cp::initialize> init;
flaxpy::control::action<mul_add_action, flaxpy::cp::mul_add> ma;
flaxpy::control::action<finalize_action, flaxpy::cp::finalize> fin;

} // namespace

int
main(int argc, char ** argv) {
  // Initialize the FleCSI run-time system.
  auto status = flecsi::initialize(argc, argv);
  status = flaxpy::control::check_status(status);
  if(status != flecsi::run::status::success) {
    return status < flecsi::run::status::clean ? 0 : status;
  }
  flecsi::log::add_output_stream("clog", std::clog, true);

  // Execute our code control point by control point.
  status = flecsi::start(flaxpy::control::execute);

  // Finalize the FleCSI run-time system.
  flecsi::finalize();
  return status;
}
