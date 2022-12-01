#include <flecsi-config.h>

#include "flecsi/data.hh"
#include "flecsi/run/leg/context.hh"
#include "flecsi/run/leg/mapper.hh"

namespace flecsi {
// These must be defined with the full execution machinery available:
namespace data::leg {
mirror::mirror(size2 s)
  : rects({s.first, 2}), columns({2, 1}),
    part(rects,
      columns,
      halves::field(columns)
        .use([s](auto f) { execute<fill>(f, s.first); })
        .fid(),
      complete),
    width(s.second) {}
void
mirror::fill(halves::Field::accessor<wo> a, size_t c) {
  const Legion::coord_t clr = color();
  a[0] = {{0, clr}, {upper(c), clr}};
}
void
mirror::extend(field<std::size_t, single>::accessor<ro> r,
  halves::Field::accessor<wo> w,
  std::size_t width) {
  const Legion::coord_t c = color();
  w[0] = {{c, 0}, {c, upper(r)}};
  w[1] = {{c, static_cast<Legion::coord_t>(r)}, {c, upper(width)}};
}
} // namespace data::leg

namespace run {

using namespace boost::program_options;

/*----------------------------------------------------------------------------*
  Legion top-level task.
 *----------------------------------------------------------------------------*/

void
top_level_task(const Legion::Task *,
  const std::vector<Legion::PhysicalRegion> &,
  Legion::Context,
  Legion::Runtime *) {

  context_t & context_ = context_t::instance();

  context_.mpi_wait();
  /*
    Invoke the FleCSI runtime top-level action.
   */

  detail::data_guard(),
    Legion::Runtime::set_return_code((*context_.top_level_action_)());

  /*
    Finish up Legion runtime and fall back out to MPI.
   */

  context_.mpi_handoff();
} // top_level_task

//----------------------------------------------------------------------------//
// Implementation of context_t::initialize.
//----------------------------------------------------------------------------//

context_t::context_t(int argc, char ** argv, bool d)
  : dep_base{d ? opt(std::in_place, argc, argv) : std::nullopt},
    context(argc, argv, util::mpi::size(), util::mpi::rank()) {
  if(exit_status() != success) {
    dep.reset();
  } // if
} // initialize

//----------------------------------------------------------------------------//
// Implementation of context_t::start.
//----------------------------------------------------------------------------//

int
context_t::start(const std::function<int()> & action) {
  using namespace Legion;
  using util::mpi::test;

  /*
    Store the top-level action for invocation from the top-level task.
   */

  top_level_action_ = &action;

  /*
    Setup Legion top-level task.
   */

  const TaskID FLECSI_TOP_LEVEL_TASK_ID = Runtime::generate_static_task_id();
  Runtime::set_top_level_task_id(FLECSI_TOP_LEVEL_TASK_ID);

  {
    Legion::TaskVariantRegistrar registrar(
      FLECSI_TOP_LEVEL_TASK_ID, "runtime_driver");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(
      registrar, "runtime_driver");
  } // scope

  /*
    Arg 0: MPI has initial control (true).
    Arg 1: Number of MPI participants (1).
    Arg 2: Number of Legion participants (1).
   */

  handshake_ = Legion::Runtime::create_handshake(true, 1, 1);

  /*
    Register custom mapper.
   */

  Runtime::add_registration_callback(mapper_registration);

  /*
    Configure interoperability layer.
   */

  Legion::Runtime::configure_MPI_interoperability(context::process_);

  context::start();

  /*
    Legion command-line arguments.
   */

  std::vector<char *> largv;
  largv.push_back(argv_[0]);

  for(auto & arg : backend_args_) {
    largv.push_back(&arg[0]);
  }

  // FIXME: This needs to be gotten from Legion
  context::threads_per_process_ = 1;
  context::threads_ = context::processes_ * context::threads_per_process_;

  Runtime::start(largv.size(), largv.data(), true);

  while(true) {
    test(MPI_Barrier(MPI_COMM_WORLD));
    handshake_.mpi_handoff_to_legion();
    handshake_.mpi_wait_on_legion();
    test(MPI_Barrier(MPI_COMM_WORLD));
    if(!mpi_task_)
      break;
    mpi_task_();
    mpi_task_ = nullptr;
  }

  return Legion::Runtime::wait_for_shutdown();
} // context_t::start

} // namespace run
} // namespace flecsi
