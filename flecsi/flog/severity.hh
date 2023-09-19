// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_LOG_SEVERITY_HH
#define FLECSI_LOG_SEVERITY_HH

#include "flecsi/config.hh"
#include "flecsi/flog/utils.hh"

#include <sstream>

namespace flecsi {
namespace flog {
/// \addtogroup flog
/// \{

inline std::string
verbose(const char * file, int line) {
  std::ostringstream ss;
  ss << timestamp() << " " << rstrip<'/'>(file) << ":" << line << " ";
  return std::move(ss).str();
}

#if defined(FLOG_ENABLE_MPI)
#define process_stamp " p" << state::instance().process()
#else
#define process_stamp ""
#endif

#define thread_stamp " t" << std::this_thread::get_id()

// Displays messages without decoration.
struct utility {
  static constexpr bool strip() {
    return false;
  }

  static bool format(std::stringstream &, const char *, int, bool) {
    return false;
  }
}; // struct utility

//----------------------------------------------------------------------------//
// trace
//----------------------------------------------------------------------------//

struct trace {
  static bool strip() {
    return flog::state::instance().strip_level() > 0;
  }

  static bool
  format(std::stringstream & ss, const char * file, int line, bool devel) {
    std::string label = devel ? "(devel) " : "";

    ss << FLOG_OUTPUT_CYAN("[trace ") << FLOG_OUTPUT_PURPLE(label);
    if(state::instance().verbose())
      ss << FLOG_OUTPUT_LTGRAY(verbose(file, line));
    ss << FLOG_OUTPUT_CYAN(state::active_tag_name());
    ss << FLOG_OUTPUT_GREEN(process_stamp);
    ss << FLOG_OUTPUT_LTBLUE(thread_stamp);
    ss << FLOG_OUTPUT_CYAN("] ") << std::endl;

    return false;
  }
}; // struct trace

//----------------------------------------------------------------------------//
// info
//----------------------------------------------------------------------------//

struct info {
  static bool strip() {
    return flog::state::instance().strip_level() > 1;
  }

  static bool
  format(std::stringstream & ss, const char * file, int line, bool devel) {
    std::string label = devel ? "(devel) " : "";

    ss << FLOG_OUTPUT_GREEN("[info ") << FLOG_OUTPUT_PURPLE(label);
    if(state::instance().verbose())
      ss << FLOG_OUTPUT_LTGRAY(verbose(file, line));
    ss << FLOG_OUTPUT_CYAN(state::active_tag_name());
    ss << FLOG_OUTPUT_GREEN(process_stamp);
    ss << FLOG_OUTPUT_LTBLUE(thread_stamp);
    ss << FLOG_OUTPUT_GREEN("] ") << std::endl;

    return false;
  } // info
}; // struct info

//----------------------------------------------------------------------------//
// warn
//----------------------------------------------------------------------------//

struct warn {
  static bool strip() {
    return flog::state::instance().strip_level() > 2;
  }

  static bool
  format(std::stringstream & ss, const char * file, int line, bool devel) {
    std::string label = devel ? "(devel) " : "";

    ss << FLOG_OUTPUT_BROWN("[warn ") << FLOG_OUTPUT_PURPLE(label);
    if(state::instance().verbose())
      ss << FLOG_OUTPUT_LTGRAY(verbose(file, line));
    ss << FLOG_OUTPUT_CYAN(state::active_tag_name());
    ss << FLOG_OUTPUT_GREEN(process_stamp);
    ss << FLOG_OUTPUT_LTBLUE(thread_stamp);
    ss << FLOG_OUTPUT_BROWN("] ") << std::endl << FLOG_COLOR_YELLOW;

    return true;
  }
}; // struct warn

//----------------------------------------------------------------------------//
// error
//----------------------------------------------------------------------------//

struct error {
  static bool strip() {
    return flog::state::instance().strip_level() > 3;
  }

  static bool
  format(std::stringstream & ss, const char * file, int line, bool devel) {
    std::string label = devel ? "(devel) " : "";

    ss << FLOG_OUTPUT_RED("[ERROR ") << FLOG_OUTPUT_PURPLE(label);
    if(state::instance().verbose())
      ss << FLOG_OUTPUT_LTGRAY(verbose(file, line));
    ss << FLOG_OUTPUT_CYAN(state::active_tag_name());
    ss << FLOG_OUTPUT_GREEN(process_stamp);
    ss << FLOG_OUTPUT_LTBLUE(thread_stamp);
    ss << FLOG_OUTPUT_RED("] ") << std::endl << FLOG_COLOR_LTRED;

    return true;
  }
}; // struct error

/// \}
} // namespace flog
} // namespace flecsi

#endif
