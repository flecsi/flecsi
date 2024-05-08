#include "flecsi/run/leg/context.hh"
#include "flecsi/data.hh"
#include "flecsi/run/leg/mapper.hh"

namespace flecsi {
// To avoid a separate source file in data/leg:
namespace data::leg {
void
with_used::extend(field<std::size_t, single>::accessor<ro> r,
  used::Field::accessor<wo> w) {
  const Legion::coord_t c = color();
  w = {{c, 0}, {c, upper(r)}};
}
} // namespace data::leg

namespace run {

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

  detail::data_guard(), task_local_base::guard(),
    Legion::Runtime::set_return_code((*context_.top_level_action_)());

  /*
    Finish up Legion runtime and fall back out to MPI.
   */

  context_.mpi_handoff();
} // top_level_task

context_t::context_t(const config & c)
  : context(c, util::mpi::size(), util::mpi::rank()), argv(c.legion) {}

dependencies_guard::dependencies_guard(dependencies_config d)
  : init(d.mpi.size(), pointers(d.mpi).data()) {}

//----------------------------------------------------------------------------//
// Implementation of context_t::start.
//----------------------------------------------------------------------------//

int
context_t::start(const std::function<int()> & action, bool check_args) {
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

  // FIXME: This needs to be gotten from Legion
  context::threads_per_process_ = 1;
  context::threads_ = context::processes_ * context::threads_per_process_;

  {
    int argc = argv.size();
    auto args = pointers(argv);
    auto p = args.data();
    Runtime::initialize(&argc, &p, true); // can be done with start after cr-16
#ifdef GASNET_CONDUIT_MPI
    util::mpi::init::finalize = false;
#endif
    if(check_args && argc > 1)
      flog_fatal("unrecognized Legion option: " << p[1]);
    Runtime::start(argc, p, true);
  }

  while(true) {
    test(MPI_Barrier(MPI_COMM_WORLD));
    handshake_.mpi_handoff_to_legion();
    handshake_.mpi_wait_on_legion();
    test(MPI_Barrier(MPI_COMM_WORLD));
    if(!mpi_task_)
      break;
    task_local_base::guard(), mpi_task_();
    mpi_task_ = nullptr;
  }

  return Legion::Runtime::wait_for_shutdown();
} // context_t::start

} // namespace run
} // namespace flecsi
