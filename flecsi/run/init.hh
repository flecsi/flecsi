// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUN_INIT_HH
#define FLECSI_RUN_INIT_HH

#include "flecsi/run/backend.hh"

#include <functional>
#include <string>

namespace flecsi {

/// \addtogroup runtime
/// \{

inline std::string argv0;

/*!
  Perform FleCSI <code>\ref runtime</code> initialization (which see).
  If \em dependent is true, this call
  will also initialize any runtime on which FleCSI depends.

  The following options are interpreted in addition to any \c program_option
  objects:
  - \c \--Xbackend=arg

    Provide a command-line option to the backend.  May be used more than once.
  - \c \--backend-args=args

    Provide command-line options to the backend.
    May be used more than once; word splitting is applied.
  - \c \--flog-tags=tags

    Enable comma-separated output \a tags.
    \c all enables all and is the default; \c unscoped disables all.
    \c none disables normal Flog output entirely.
  - <tt>\--flog-verbose[=level]</tt>

    Enable verbose output if \a level is omitted or positive; suppress
    decorations if it is negative.  The default is 0.
  - \c \--flog-process=p

    Select output from process \a p (default 0), or from all if &minus;1.
  - \c \--control-model

    Write \c <em>program</em>-control-model.dot with the control points and
    the actions for each.
  - \c \--control-model-sorted

    Write \c <em>program</em>-control-model-sorted.dot containing linearized
    actions.

  The Flog options are recognized only when that feature is enabled.
  The control model options require Graphviz support and take effect via
  \c control::check_status.

  @param argc number of command-line arguments to process
  @param argv command-line arguments to process
  @param dependent A boolean telling FleCSI whether or not to initialize
                   runtimes on which it depends.

  @return An integer indicating the initialization status. This may be
          interpreted as a \em flecsi::run::status enumeration, e.g.,
          a value of 1 is equivalent to flecsi::run::status::help.
          Control model options take effect via \c control::check_status.

  \deprecated Construct a \c runtime object; to parse command-line arguments,
    also use \link getopt <code>getopt</code>\endlink.
 */
[[deprecated(
  "use flecsi::runtime and perhaps flecsi::getopt")]] [[nodiscard]] int
initialize(int argc, const char * const * argv, bool dependent = true);

/*!
  Perform FleCSI runtime start. This causes the runtime to begin execution
  of the top-level action.

  \param  action
          The top-level action, i.e., the entry point for FleCSI to begin
          execution.

  \return An integer indicating the finalization status. This will be
          either 0 for successful completion or an error code from
          flecsi::run::status.

  \deprecated Use \c runtime::control.
 */
[[deprecated("use flecsi::runtime")]] [[nodiscard]] inline int
start(const std::function<int()> & action) {
  return run::context::instance().start(action, false);
}

/*!
  Perform FleCSI runtime finalization. If FleCSI was initialized with the \em
  dependent flag set to true, FleCSI will also finalize any runtimes on which
  it depends.

  \deprecated Destroy a \c runtime object.
 */
[[deprecated("use flecsi::runtime")]] void finalize();

/*!
  Return the program name.
  Available only with \c initialize.

  \deprecated Check \c argv directly.
 */
[[deprecated("use argv")]] inline std::string const &
program() {
  return argv0;
}

/// \}

} // namespace flecsi

#endif
