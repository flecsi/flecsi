// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUN_MPI_CONTEXT_HH
#define FLECSI_RUN_MPI_CONTEXT_HH

#include <flecsi-config.h>

#if !defined(FLECSI_ENABLE_MPI)
#error FLECSI_ENABLE_MPI not defined! This file depends on MPI!
#endif

#include "flecsi/run/local/context.hh"

#include <boost/program_options.hpp>
#include <mpi.h>

#include <map>

namespace flecsi::run {
/// \defgroup mpi-runtime MPI Runtime
/// Global state.
/// \ingroup runtime
/// \{

struct context_t : local::context {

  //--------------------------------------------------------------------------//
  //  Runtime.
  //--------------------------------------------------------------------------//

  /*!
    Documentation for this interface is in the top-level context type.
   */

  int start(const std::function<int()> &);

  /*!
    Documentation for this interface is in the top-level context type.
   */

  Color process() const {
    return process_;
  }

  Color processes() const {
    return processes_;
  }

  Color threads_per_process() const {
    return 1;
  }

  Color threads() const {
    return 0;
  }

  /*
    Documentation for this interface is in the top-level context type.
   */

  static int task_depth() {
    return 0;
  } // task_depth

  /*
    Documentation for this interface is in the top-level context type.
   */

  Color color() const {
    return process_;
  }

  /*
    Documentation for this interface is in the top-level context type.
   */

  Color colors() const {
    return processes_;
  }
};

/// \}
} // namespace flecsi::run

#endif
