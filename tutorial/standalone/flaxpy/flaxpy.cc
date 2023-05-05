/*
 * Demonstrate how to compute Y = a*X + Y over a distributed array
 * using FleCSI.
 */

#include <flecsi/data.hh>
#include <flecsi/execution.hh>
#include <flecsi/flog.hh>
#include <flecsi/run/control.hh>

// In a larger program, this namespace would typically appear in a header file,
// where the inline keywords are necessary.
namespace flaxpy {

// Let the user specify the vector length on the command line.
inline flecsi::program_option<std::size_t> vector_length(
  "Flaxpy-specific Options",
  "length,l",
  "Specify the length of the vectors to add.",
  {{flecsi::option_default, 1000000}});

// Return indices to assign to each color, divided as evenly as possible.
inline flecsi::util::equal_map
divide_indices_among_colors(flecsi::Color ncolors) {
  return {vector_length.value(), ncolors};
}

// Define a distributed vector type called dist_vector as a specialization
// of the "user" topology.
struct dist_vector
  : flecsi::topo::specialization<flecsi::topo::user, dist_vector> {
  // Return the number of indices to assign to each color.
  static coloring color() {
    // Specify one color per process, and distribute indices accordingly.
    std::vector<std::size_t> ret;
    for(auto c : divide_indices_among_colors(flecsi::processes()))
      ret.push_back(c.size());
    return ret;
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
struct control_policy : flecsi::run::control_base {
  using control_points_enum = cp;
  struct node_policy {};
  using control_points =
    list<point<cp::initialize>, point<cp::mul_add>, point<cp::finalize>>;
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
  // Arbitrarily initialize x[i] = i and y[i] = 0.  We use a forall
  // for the latter because it can run in parallel without access to
  // the index variable.
  auto p = x_acc.span().begin();
  for(size_t i :
    flaxpy::divide_indices_among_colors(flecsi::colors())[flecsi::color()])
    *p++ = i;
  forall(elt, y_acc.span(), "init_y") { elt = 0; };
}

// Implement an action for the initialize control point.
void
initialize_action(flaxpy::control_policy &) {
  dist_vector_cslot.allocate();
  dist_vector_slot.allocate(dist_vector_cslot.get());
  flecsi::execute<initialize_vectors_task>(
    x_field(dist_vector_slot), y_field(dist_vector_slot));
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
void
mul_add_action(flaxpy::control_policy &) {
  const double a = 12.34; // Arbitrary scalar value to multiply
  flecsi::execute<mul_add_task>(
    a, x_field(dist_vector_slot), y_field(dist_vector_slot));
}

// Define a task that adds up all values of Y and returns the sum.
double
reduce_y_task(one_field<double>::accessor<flecsi::ro> y_acc) {
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
void
finalize_action(flaxpy::control_policy &) {
  double sum = flecsi::reduce<reduce_y_task, flecsi::exec::fold::sum>(
    y_field(dist_vector_slot))
                 .get();
  flog(info) << "The sum over all elements in the final vector is " << sum
             << std::endl;
  dist_vector_slot.deallocate();
  dist_vector_cslot.deallocate();
}

// Register each of the preceding actions with its eponymous control point.
// None of the variables declared below are ever used; they exist only for
// the side effects induced by declaration.
flaxpy::control::action<initialize_action, flaxpy::cp::initialize> init;
flaxpy::control::action<mul_add_action, flaxpy::cp::mul_add> ma;
flaxpy::control::action<finalize_action, flaxpy::cp::finalize> fin;

} // namespace

// The main program largely delegates to the control model.
int
main(int argc, char ** argv) {
  // Initialize the FleCSI run-time system.
  flecsi::run::arguments args(argc, argv);
  const flecsi::run::dependencies_guard dg(args.dep);
  const flecsi::runtime run(args.cfg);
  flecsi::flog::add_output_stream("clog", std::clog, true);
  // Execute our code control point by control point.
  return run.main<flaxpy::control>(args.act);
}
