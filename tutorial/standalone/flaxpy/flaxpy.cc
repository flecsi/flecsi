/*
 * Demonstrate how to compute Y = a*X + Y over a distributed array
 * using FleCSI.
 */

#include <flecsi/data.hh>
#include <flecsi/execution.hh>
#include <flecsi/flog.hh>
#include <flecsi/run/control.hh>
#include <flecsi/topo/index.hh>

// Let the user specify the vector length on the command line.
//
// In a larger program in which this definition appeared in a header
// file, it could be declared inline.
flecsi::program_option<std::size_t> vector_length("Flaxpy-specific Options",
  "length,l",
  "Specify the length of the vectors to add.",
  {{flecsi::option_default, 1000000}});

// Use our own namespace so as not to pollute the global namespace.
namespace flaxpy {

// Define a distributed vector type called dist_vector as a specialization
// of the "user" topology.
struct dist_vector
  : flecsi::topo::specialization<flecsi::topo::user, dist_vector> {
  // Return the number of indices to assign to each color.
  static coloring color() {
    // Specify one color per process, and assign the same number of
    // indices to each color.
    std::size_t nproc = flecsi::processes();
    std::size_t indexes_per_color = vector_length.value() / nproc;
    coloring * subvectors = new coloring(nproc, indexes_per_color);

    // Evenly distribute the leftover indices among the first colors.
    std::size_t leftovers = vector_length.value() % nproc;
    for(std::size_t i = 0; i < leftovers; ++i)
      (*subvectors)[i]++;
    return *subvectors;
  }
};

// Add two fields, x_field and y_field, to dist_vector.
//
// For clarity we specify flecsi::data::layout::dense as a template
// parameter, but this is in fact the default and would normally be
// omitted.
template<typename T>
using one_field = flecsi::field<T, flecsi::data::layout::dense>;
const one_field<double>::definition<dist_vector> x_field;
const one_field<double>::definition<dist_vector> y_field;

// Declare a coloring of the distributed vector.
dist_vector::slot dist_vector_slot;
dist_vector::cslot dist_vector_cslot;

// Define three FleCSI control-point identifiers.
enum class cp { initialize, mul_add, finalize };

// Overload "*" to convert a control-point identifier to a string.
//
// In a larger program in which this function appeared in a header
// file, it could be declared inline.
const char *
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

// Define a task that initializes the elements of the distributed vector.
void
initialize_vectors_task(one_field<double>::accessor<flecsi::wo> x_acc,
  one_field<double>::accessor<flecsi::wo> y_acc) {
  // Compute our starting offset into the global vector.
  std::size_t base = 0;
  flecsi::data::coloring_slot<dist_vector>::color_type num_indices_per_color =
    dist_vector_cslot.get();
  std::size_t current_color = flecsi::color();
  for(std::size_t i = 0; i < current_color; ++i)
    base += num_indices_per_color[i];

  // Arbitrarily initialize x[i] = i and y[i] = length - i.
  std::size_t num_local_elts =
    num_indices_per_color[flecsi::color()]; // Same as x_acc.span().size()
  forall(i, x_acc.span(), "init_x") { x_acc[i] = double(base + i); };
  forall(i, y_acc.span(), "init_y") {
    y_acc[i] = double(vector_length.value() - (base + i));
  };
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
  auto local_sum = reduceall(
    i, accum, y_acc.span(), flecsi::exec::fold::sum, double, "reduce_y_task") {
    accum(y_acc[i]);
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
control::action<initialize_action, cp::initialize> init;
control::action<mul_add_action, cp::mul_add> ma;
control::action<finalize_action, cp::finalize> fin;

} // namespace flaxpy

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
