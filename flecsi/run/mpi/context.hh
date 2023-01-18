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

namespace flecsi {
namespace run {
/// \defgroup mpi-runtime MPI Runtime
/// Global state.
/// \ingroup runtime
/// \{

struct dependencies_guard {
  dependencies_guard(arguments::dependent &);
  ~dependencies_guard();

private:
  dependencies_guard(arguments::dependent &, int, char **);

  util::mpi::init mpi;
};

struct context_t : context {

  //--------------------------------------------------------------------------//
  //  Runtime.
  //--------------------------------------------------------------------------//

  context_t(const arguments::config &, arguments::action &);

  /*!
    Documnetation for this interface is in the top-level context type.
   */

  int start(const std::function<int()> &);

  /*
    Documnetation for this interface is in the top-level context type.
   */

  static int task_depth() {
    return depth;
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

using runtime = run::context_t;

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
