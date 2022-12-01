// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUN_MPI_CONTEXT_HH
#define FLECSI_RUN_MPI_CONTEXT_HH

#include <flecsi-config.h>

#if !defined(FLECSI_ENABLE_MPI)
#error FLECSI_ENABLE_MPI not defined! This file depends on MPI!
#endif

#include "flecsi/run/context.hh"
#include "flecsi/util/mpi.hh"

#include <boost/program_options.hpp>
#include <mpi.h>

#include <map>
#include <optional>

namespace flecsi::run {
/// \defgroup mpi-runtime MPI Runtime
/// Global state.
/// \ingroup runtime
/// \{

struct context_t : context {

  //--------------------------------------------------------------------------//
  //  Runtime.
  //--------------------------------------------------------------------------//

  /*!
    Documnetation for this interface is in the top-level context type.
   */

  int initialize(int argc, char ** argv, bool dependent);

  /*!
    Documnetation for this interface is in the top-level context type.
   */

  void finalize();

  /*!
    Documnetation for this interface is in the top-level context type.
   */

  int start(const std::function<int()> &);

  /*
    Documnetation for this interface is in the top-level context type.
   */

  static int task_depth() {
    return 0;
  } // task_depth

  /*
    Documnetation for this interface is in the top-level context type.
   */

  Color color() const {
    return process_;
  }

  /*
    Documnetation for this interface is in the top-level context type.
   */

  Color colors() const {
    return processes_;
  }

private:
  struct dependent {
    dependent(int &, char **&);
    ~dependent();

  private:
    util::mpi::init mpi;
  };

  std::optional<dependent> dep;
};

/// \}
} // namespace flecsi::run

#endif
