#include "flecsi/run/leg/context.hh"
#include "flecsi/data.hh"
#include "flecsi/run/leg/mapper.hh"

namespace flecsi {
// To avoid a separate source file in data/leg:
namespace data::leg {
void
with_used::extend(exec::cpu s,
  prefixes_base::Field::accessor<ro> r,
  used::Field::accessor<wo> w) noexcept {
  const Legion::coord_t c = s.launch().index;
  w = {{c, 0}, {c, upper(r.get())}};
}
} // namespace data::leg

namespace run {

/*----------------------------------------------------------------------------*
  Legion top-level task.
 *----------------------------------------------------------------------------*/

void
context_t::top_level_task(const Legion::Task *,
  const std::vector<Legion::PhysicalRegion> &,
  Legion::Context,
  Legion::Runtime *) noexcept {

  context_t & context_ = instance();

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

  const TaskID top_id = Runtime::generate_static_task_id();
  Runtime::set_top_level_task_id(top_id);

  {
    Legion::TaskVariantRegistrar registrar(top_id, "runtime_driver");
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

  Legion::Runtime::configure_MPI_interoperability(process());

  context::start();

  // FIXME: This needs to be gotten from Legion
  context::threads_per_process_ = 1;
  threads_ = processes() * threads_per_process_;

  const auto param_clean = params.clean();

  Runtime::start(argv.size(), pointers(argv).data(), true, true, true);
#ifdef GASNET_CONDUIT_MPI
  util::mpi::init::finalize = false;
#endif
  if(check_args)
    if(auto & args = Runtime::get_input_args(); args.argc > 1)
      flog_fatal("unrecognized Legion option: " << args.argv[1]);

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

void
param_locker::run() {
  task_idx i;
  const auto b = [&] {
    util::mpi::test(MPI_Bcast(&i, 1, util::mpi::type<decltype(i)>(), 0, comm));
  };
  if(!rank(comm)) {
    for(bool closed = false, done = false; !done;) {
      recv(i, MPI_ANY_SOURCE, 0, comm);
      if(!i) {
        closed = true;
        if(lease(), tasks.empty())
          done = true;
      }
      else if(lease(), [&, it = tasks.try_emplace(i).first] {
                const bool zero = !--it->second.ref;
                if(zero) {
                  tasks.erase(it);
                  if(closed && tasks.empty())
                    done = true;
                }
                return zero;
              }())
        b();
    }
    i = 0;
    b();
  }
  else
    while(b(), i)
      lease(), [&] {
        // If we haven't launched the task yet, make a blocking placeholder:
        const auto [it, nu] = tasks.try_emplace(i);
        if(!nu)
          tasks.erase(it);
      }();
}

} // namespace run
} // namespace flecsi
