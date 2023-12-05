// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUN_INIT_HH
#define FLECSI_RUN_INIT_HH

namespace flecsi {

/// \addtogroup runtime
/// \{

inline std::string argv0;

void finalize();

/*!
  Perform FleCSI <code>\ref runtime</code> initialization (which see).
  If \em dependent is true, this call
  will also initialize any runtime on which FleCSI depends.

  @param argc number of command-line arguments to process
  @param argv command-line arguments to process
  @param dependent A boolean telling FleCSI whether or not to initialize
                   runtimes on which it depends.

  @return An integer indicating the initialization status. This may be
          interpreted as a \em flecsi::run::status enumeration, e.g.,
          a value of 1 is equivalent to flecsi::run::status::help.
          Control model options take effect via \c control::check_status.

  \deprecated Construct a \c runtime object.
 */
[[deprecated("use flecsi::runtime")]] inline int
initialize(int argc, char ** argv, bool dependent = true) {
  run::arguments args(argc, argv);
  argv0 = args.act.program;
  const auto make = [](auto & o, auto & x) -> auto & {
    flog_assert(!o, "already initialized");
    return o.emplace(x);
  };
  if(dependent)
    make(run::dependent, args.dep);
  auto & ctx = make(run::context::ctx, args.cfg);
  ctx.check_config(args.act);
  const auto c = args.act.status();
  if(c) {
    if(!ctx.process())
      std::cerr << args.act.stderr;
    finalize();
  }
  return c;
}

/*!
  Perform FleCSI runtime start. This causes the runtime to begin execution
  of the top-level action.

  \param  action
          The top-level action, i.e., the entry point for FleCSI to begin
          execution.

  \return An integer indicating the finalization status. This will be
          either 0 for successful completion or an error code from
          flecsi::run::status.

  \deprecated Use \c runtime::main.
 */
[[deprecated("use flecsi::runtime")]] inline int
start(const std::function<int()> & action) {
  return run::context::instance().start(action, false);
}

/*!
  Perform FleCSI runtime finalization. If FleCSI was initialized with the \em
  dependent flag set to true, FleCSI will also finalize any runtimes on which
  it depends.

  \deprecated Destroy a \c runtime object.
 */
[[deprecated("use flecsi::runtime")]] inline void
finalize() {
  run::context::ctx.reset();
  run::dependent.reset();
}

/*!
  Return the program name.
  Available only with \c initialize.

  \deprecated Use \c act.program in \c run::arguments.
 */
[[deprecated("use run::arguments")]] inline std::string const &
program() {
  return argv0;
}

/// \}

} // namespace flecsi

#endif
