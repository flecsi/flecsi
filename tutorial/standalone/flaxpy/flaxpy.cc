/*
 * Demonstrate how to compute Y = a*X + Y over a distributed array
 * using FleCSI.
 */

#include <flecsi/data.hh>
#include <flecsi/execution.hh>
#include <flecsi/flog.hh>
#include <flecsi/runtime.hh>

// In a larger program, this namespace would typically appear in a header file,
// where the inline keywords are necessary.
namespace flaxpy {

// Let the user specify the vector length on the command line.
inline flecsi::program_option<flecsi::util::gid> vector_length(
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
  static coloring color(flecsi::Color nc) {
    std::vector<std::size_t> ret;
    for(auto c : divide_indices_among_colors(nc))
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
  using control_points =
    list<point<cp::initialize>, point<cp::mul_add>, point<cp::finalize>>;

  dist_vector::ptr dist_vector_ptr;

  auto & vector() {
    return *dist_vector_ptr;
  }
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
using one_field = flecsi::field<double, flecsi::data::layout::dense>;
const one_field::definition<flaxpy::dist_vector> x_field, y_field;

// Define a task that initializes the elements of the distributed vector.
void
initialize_vectors_task(flecsi::exec::accelerator s,
  one_field::accessor<flecsi::wo> x_acc,
  one_field::accessor<flecsi::wo> y_acc) noexcept {
  // Arbitrarily initialize x[i] = i and y[i] = 0.  We use a forall
  // for the latter because it can run in parallel without access to
  // the index variable.
  auto p = x_acc.span().begin();
  for(flecsi::util::id i :
    flaxpy::divide_indices_among_colors(s.launch().size)[s.launch().index])
    *p++ = i;
  s.executor().forall(elt, y_acc.span()) {
    elt = 0;
  };
}

// Implement an action for the initialize control point.
void
initialize_action(flaxpy::control_policy & policy) {
  auto & sch = policy.scheduler();
  // Specify one color per process.
  sch.allocate(policy.dist_vector_ptr,
    flaxpy::dist_vector::mpi_coloring(sch, sch.runtime().processes()));
  sch.execute<initialize_vectors_task>(
    flecsi::exec::on, x_field(policy.vector()), y_field(policy.vector()));
}

// Define a task that assigns Y <- a*X + Y.
void
mul_add_task(double a,
  one_field::accessor<flecsi::ro> x_acc,
  one_field::accessor<flecsi::rw> y_acc) noexcept {
  const auto num_local_elts = x_acc.span().size();
  for(flecsi::util::id i = 0; i < num_local_elts; ++i)
    y_acc[i] += a * x_acc[i];
}

// Implement an action for the mul_add control point.
void
mul_add_action(flaxpy::control_policy & policy) {
  const double a = 12.34; // Arbitrary scalar value to multiply
  policy.scheduler().execute<mul_add_task>(
    a, x_field(policy.vector()), y_field(policy.vector()));
}

// Define a task that adds up all values of Y and returns the sum.
double
reduce_y_task(flecsi::exec::accelerator s,
  one_field::accessor<flecsi::ro> y_acc) noexcept {
  const auto local_sum = s.executor().reduceall(
    elt, accum, y_acc.span(), flecsi::exec::fold::sum, double) {
    accum(elt);
  };
  return local_sum;
}

// Implement an action for the finalize control point.
void
finalize_action(flaxpy::control_policy & policy) {
  const double sum = policy.scheduler()
                       .reduce<reduce_y_task, flecsi::exec::fold::sum>(
                         flecsi::exec::on, y_field(policy.vector()))
                       .get();
  flog(info) << "The sum over all elements in the final vector is " << sum
             << std::endl;
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
  flecsi::getopt()(argc, argv);
  const flecsi::run::dependencies_guard dg;
  flecsi::runtime run;
  flecsi::flog::add_output_stream("clog", std::clog, true);
  // Execute our code control point by control point.
  return run.control<flaxpy::control>();
}
