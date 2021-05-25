// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUN_LOCAL_CONTEXT_HH
#define FLECSI_RUN_LOCAL_CONTEXT_HH

#include "flecsi/run/context.hh"

namespace flecsi::run::local {

// Common initialization and finalize operations for MPI and HPX
struct context : run::context {

  int initialize(int argc, char ** argv, bool dependent);

  void finalize();
};

} // namespace flecsi::run::local

#endif
