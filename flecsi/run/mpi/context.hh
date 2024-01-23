// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUN_MPI_CONTEXT_HH
#define FLECSI_RUN_MPI_CONTEXT_HH

#include "flecsi/run/context.hh"
#include "flecsi/util/mpi.hh"

#include <boost/program_options.hpp>
#include <mpi.h>

#include <map>

namespace flecsi {
namespace run {
/// \defgroup mpi-runtime MPI Runtime
/// Global state.
/// \ingroup runtime
/// \{

struct dependencies_guard {
  dependencies_guard(dependencies_config = {});

private:
  util::mpi::init mpi;
#ifdef FLECSI_ENABLE_KOKKOS
  Kokkos::ScopeGuard kokkos;
#endif
};

struct config : config_base {};

struct context_t : context {

  //--------------------------------------------------------------------------//
  //  Runtime.
  //--------------------------------------------------------------------------//

  context_t(const config &);

  int start(const std::function<int()> &, bool);

  static int task_depth() {
    return depth;
  } // task_depth

  Color color() const {
    return process_;
  }

  Color colors() const {
    return processes_;
  }

  static inline int depth;

  struct depth_guard {
    depth_guard() {
      ++depth;
    }
    ~depth_guard() {
      --depth;
    }
  };
};

/// \}
} // namespace run

template<class T>
struct task_local : private run::task_local_base {
  T & operator*() noexcept {
    return *cur();
  }
  T * operator->() noexcept {
    return &**this;
  }

private:
  void emplace() override {
    cur().emplace();
  }
  void reset() noexcept override {
    cur().reset();
  }

  std::optional<T> & cur() {
    return run::context_t::task_depth() ? task : outer;
  }

  std::optional<T> outer, task;
};

} // namespace flecsi

#endif
