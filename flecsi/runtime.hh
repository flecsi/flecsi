// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUNTIME_HH
#define FLECSI_RUNTIME_HH

#include "flecsi/run/backend.hh"
#include "flecsi/run/control.hh"

namespace flecsi {

/// \defgroup runtime Runtime Model
/// Environmental information and tools for organizing applications.
/// \code#include "flecsi/runtime.hh"\endcode
///
/// This header provides the features in the following additional \b deprecated
/// header:
/// - \ref control "flecsi/run/control.hh"
///
/// \{

/// FleCSI runtime state.
/// Only one can exist at a time.
struct runtime {
  /// Construct from a configuration.
  ///
  /// As a debugging aid, if the \c FLECSI_SLEEP environment variable is set
  /// to an integer, wait for that number of seconds.
  explicit runtime(const run::arguments::config & c) {
    auto & r = run::context::ctx;
    flog_assert(!r, "runtime already initialized");
    r.emplace(c);
  }
  /// Immovable.
  runtime(runtime &&) = delete;
  ~runtime() {
    run::context::ctx.reset();
  }

  /// Perform the \ref run::arguments::action::operation "operation"
  /// indicated by a command line.
  /// \tparam C control model
  /// \param aa arguments for \link run::control::invoke `C::invoke`\endlink
  ///   (if it is called)
  /// \return resulting exit code
  /// \warning The return value does not in general match that from
  ///   \c initialize or \c start.
  template<class C, class... AA>
  int main(run::arguments::action a, AA &&... aa) const {
    using A = decltype(a);
    auto & ctx = run::context::instance();
    ctx.check_config(a);
    if(!ctx.process())
      std::cerr << a.stderr;
    switch(a.op) {
#ifdef FLECSI_ENABLE_GRAPHVIZ
      case A::control_model:
        C::write_graph(a.program);
        break;
      case A::control_model_sorted:
        C::write_actions(a.program);
        break;
#endif
      case A::run:
        return ctx.start([&] { return C::invoke(std::forward<AA>(aa)...); });
      case A::help:
        break;
      default:
        return 1;
    }
    return 0;
  }
};

/*!
  Return the current process id.
 */

inline Color
process() {
  return run::context::instance().process();
}

/*!
  Return the number of processes.
 */

inline Color
processes() {
  return run::context::instance().processes();
}

/*!
  Return the number of threads per process.
 */

inline Color
threads_per_process() {
  return run::context::instance().threads_per_process();
}

/*!
  Return the number of execution instances with which the runtime was
  invoked. In this context a \em thread is defined as an instance of
  execution, and does not imply any other properties. This interface can be
  used to determine the full subscription of the execution instances of the
  running process that invoked the FleCSI runtime.
 */

inline Color
threads() {
  return run::context::instance().threads();
}

/*!
  Return the color of the current execution instance. This function is only
  valid if invoked from within a task.
 */

inline Color
color() {
  return run::context::instance().color();
}

/*!
  Return the number of colors of the current task invocation. This function is
  only valid if invoked from within a task.
 */

inline Color
colors() {
  return run::context::instance().colors();
}

/// \}

} // namespace flecsi

#endif
