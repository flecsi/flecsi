/*
    @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
   /@@/////  /@@          @@////@@ @@////// /@@
   /@@       /@@  @@@@@  @@    // /@@       /@@
   /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
   /@@////   /@@/@@@@@@@/@@       ////////@@/@@
   /@@       /@@/@@//// //@@    @@       /@@/@@
   /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
   //       ///  //////   //////  ////////  //

   Copyright (c) 2016, Triad National Security, LLC
   All rights reserved.
                                                                              */

#include <flecsi-config.h>

#include "flecsi/data.hh"
#include "flecsi/run/leg/context.hh"
#include "flecsi/run/leg/mapper.hh"

namespace flecsi {
namespace data::leg {
mirror::mirror(size2 s)
  : rects({s.first, 2}), columns({2, 1}),
    part(rects,
      columns,
      (execute<fill>(halves::field(columns), s.first), fid),
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
  Legion::Context ctx,
  Legion::Runtime * runtime) {

  context_t & context_ = context_t::instance();

  /*
    Initialize MPI interoperability.
   */

  context_.connect_with_mpi(ctx, runtime);
  context_.mpi_wait();
  /*
    Invoke the FleCSI runtime top-level action.
   */

  detail::data_guard(),
    context_.exit_status() = (*context_.top_level_action_)();

  /*
    Finish up Legion runtime and fall back out to MPI.
   */

  context_.mpi_handoff();
} // top_level_task

//----------------------------------------------------------------------------//
// Implementation of context_t::initialize.
//----------------------------------------------------------------------------//

int
context_t::initialize(int argc, char ** argv, bool dependent) {
  using util::mpi::test;

  if(dependent) {
    int version, subversion;
    test(MPI_Get_version(&version, &subversion));

#if defined(GASNET_CONDUIT_MPI)
    if(version == 3 && subversion > 0) {
      int provided;
      test(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));

      if(provided < MPI_THREAD_MULTIPLE) {
        std::cerr << "Your implementation of MPI does not support "
                     "MPI_THREAD_MULTIPLE which is required for use of the "
                     "GASNet MPI conduit with the Legion-MPI Interop!"
                  << std::endl;
        std::abort();
      } // if
    }
    else {
      // Initialize the MPI runtime
      test(MPI_Init(&argc, &argv));
    } // if
#else
    test(MPI_Init(&argc, &argv));
#endif
  } // if

  std::tie(context::process_, context::processes_) = util::mpi::info();

  auto status = context::initialize_generic(argc, argv, dependent);

  if(status != success && dependent) {
    test(MPI_Finalize());
  } // if

  return status;
} // initialize

//----------------------------------------------------------------------------//
// Implementation of context_t::finalize.
//----------------------------------------------------------------------------//

void
context_t::finalize() {
  context::finalize_generic();

#ifndef GASNET_CONDUIT_MPI
  if(context::initialize_dependent_) {
    util::mpi::test(MPI_Finalize());
  } // if
#endif
} // finalize

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

  auto iss = std::istringstream{backend_};
  std::vector<std::string> lsargv(std::istream_iterator<std::string>{iss},
    std::istream_iterator<std::string>());

  for(auto & arg : lsargv) {
    largv.push_back(&arg[0]);
  } // for

  // FIXME: This needs to be gotten from Legion
  context::threads_per_process_ = 1;
  context::threads_ = context::processes_ * context::threads_per_process_;

  /*
    Start Legion runtime.
   */

  {
    log::devel_guard("context");

    std::stringstream stream;

    stream << "Starting Legion runtime" << std::endl;
    stream << "\targc: " << largv.size() << std::endl;
    stream << "\targv: ";

    for(auto opt : largv) {
      stream << opt << " ";
    } // for

    stream << std::endl;

    flog_devel(info) << stream.str();
  } // scope

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

  Legion::Runtime::wait_for_shutdown();

  return context::exit_status();
} // context_t::start

//----------------------------------------------------------------------------//
// Implementation of context_t::connect_with_mpi.
//----------------------------------------------------------------------------//

void
context_t::connect_with_mpi(Legion::Context &, Legion::Runtime *) {
  LegionRuntime::Arrays::Rect<1> launch_bounds(
    LegionRuntime::Arrays::Point<1>(0),
    LegionRuntime::Arrays::Point<1>(processes_ - 1));

  context_t::instance().set_all_processes(launch_bounds);
} // context_t::connect_with_mpi

} // namespace run
} // namespace flecsi
