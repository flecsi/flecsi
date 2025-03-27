// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUNTIME_HH
#define FLECSI_RUNTIME_HH

#include "flecsi/exec/fwd.hh"
#include "flecsi/run/backend.hh"
#include "flecsi/run/control.hh"
#include "flecsi/run/options.hh"

namespace flecsi {

/// \defgroup runtime Runtime Model
/// Environmental information and tools for organizing applications.
/// \ns{run}.
/// \code#include "flecsi/runtime.hh"\endcode
///
/// This header provides the features in the following additional \b deprecated
/// header:
/// - \ref control "flecsi/run/control.hh"
///
/// \{

/// FleCSI runtime state.
/// Only one can exist at a time.
///
/// \ns.
struct runtime {
  /// Construct from a configuration.
  ///
  /// As a debugging aid, if the \c FLECSI_SLEEP environment variable is set
  /// to an integer, wait for that number of seconds.
  explicit runtime(const run::config & c = {}) {
    auto & r = run::context::ctx;
    flog_assert(!r, "runtime already initialized");
    r.emplace(c);
    scheduler::instance.emplace(*this);
  }
  /// Immovable.
  runtime(runtime &&) = delete;
  ~runtime() {
    scheduler::instance.reset();
    run::context::ctx.reset();
  }

  /// Execute a control model.
  /// \tparam C control model
  /// \param aa arguments for \link run::control::invoke `C::invoke`\endlink
  /// \return resulting exit code
  template<class C, class... AA>
  int control(AA &&... aa) {
    auto & ctx = run::context::instance();
    return ctx.start(
      [&] { return C::invoke(*scheduler::instance, std::forward<AA>(aa)...); },
      true);
  }
  /// \deprecated Use a non-\c const \c runtime.
  template<class C, class... AA>
  [[deprecated("use a non-const runtime")]] int control(AA &&... aa) const {
    return const_cast<runtime &>(*this).control<C>(std::forward<AA>(aa)...);
  }
};

/*!
  Return the current process id.  \ns.
 */

inline Color
process() {
  return run::context::instance().process();
}

/*!
  Return the number of processes.  \ns.
 */

inline Color
processes() {
  return run::context::instance().processes();
}

/*!
  Return the number of threads per process.  \ns.
  \deprecated This information is unreliable and not useful for scheduling.
 */

[[deprecated]] inline Color
threads_per_process() {
  return run::context::instance().threads_per_process();
}

/*!
  Return the number of execution instances with which the runtime was
  invoked. In this context a \em thread is defined as an instance of
  execution, and does not imply any other properties. This interface can be
  used to determine the full subscription of the execution instances of the
  running process that invoked the FleCSI runtime.

  \ns.
  \deprecated This information is unreliable and not useful for scheduling.
 */

[[deprecated]] inline Color
threads() {
  return run::context::instance().threads();
}

/*!
  Return the color of the current execution instance. This function is only
  valid if invoked from within a non MPI task. For MPI task, use \c #process,
  which in that case equals to the color used from any topology.

  \ns.
 */

inline Color
color() {
  return run::context::instance().color();
}

/*!
  Return the number of colors of the current task invocation. This function is
  only valid if invoked from within a non MPI task. For MPI task, use \c
  #processes which in that case equals to the number of colors used from any
  topology.

  \ns.
 */

inline Color
colors() {
  return run::context::instance().colors();
}

/*!
  Return the mapping of shortened FleCSI task signatures to their full function
  signatures. Shortened names may be used by FleCSI to register tasks in some
  backends and provide more user-friendly names for debugging purposes. This map
  is empty if no such shortening took place.

  \ns.
 */

inline const std::map<std::string, std::string> &
task_names() {
  return run::context::instance().task_names();
}

/// \}

} // namespace flecsi

#endif
