/*
    @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
   /@@/////  /@@          @@////@@ @@////// /@@
   /@@       /@@  @@@@@  @@    // /@@       /@@
   /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
   /@@////   /@@/@@@@@@@/@@       ////////@@/@@
   /@@       /@@/@@//// //@@    @@       /@@/@@
   /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
   //       ///  //////   //////  ////////  //

   Copyright (c) 2016, Triad National Security, LLC
   All rights reserved.
                                                                              */
#pragma once

/*! @file */

#include <flecsi-config.h>

#include "flecsi/log/utils.hh"

#include <sstream>

namespace flecsi {
namespace log {

inline std::string
verbose(const char * file, int line) {
  std::stringstream ss;
  ss << timestamp() << " " << rstrip<'/'>(file) << ":" << line << " ";
  return ss.str();
}

#if defined(FLOG_ENABLE_MPI)
#define process_stamp " p" << state::instance().process()
#else
#define process_stamp ""
#endif

#define thread_stamp " t" << std::this_thread::get_id()

//----------------------------------------------------------------------------//
// utility
//----------------------------------------------------------------------------//

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
  static constexpr bool strip() {
    return FLOG_STRIP_LEVEL > 0;
  }

  static bool
  format(std::stringstream & ss, const char * file, int line, bool devel) {
    std::string label = devel ? "(devel) " : "";
    std::string tag = state::instance().initialized()
                        ? state::instance().active_tag_name()
                        : "external";

    ss << FLOG_OUTPUT_CYAN("[trace ") << FLOG_OUTPUT_PURPLE(label);
    if(state::instance().verbose())
      ss << FLOG_OUTPUT_LTGRAY(verbose(file, line));
    ss << FLOG_OUTPUT_CYAN(tag);
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
  static constexpr bool strip() {
    return FLOG_STRIP_LEVEL > 1;
  }

  static bool
  format(std::stringstream & ss, const char * file, int line, bool devel) {
    std::string label = devel ? "(devel) " : "";
    std::string tag = state::instance().initialized()
                        ? state::instance().active_tag_name()
                        : "external";

    ss << FLOG_OUTPUT_GREEN("[info ") << FLOG_OUTPUT_PURPLE(label);
    if(state::instance().verbose())
      ss << FLOG_OUTPUT_LTGRAY(verbose(file, line));
    ss << FLOG_OUTPUT_CYAN(tag);
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
  static constexpr bool strip() {
    return FLOG_STRIP_LEVEL > 2;
  }

  static bool
  format(std::stringstream & ss, const char * file, int line, bool devel) {
    std::string label = devel ? "(devel) " : "";
    std::string tag = state::instance().initialized()
                        ? state::instance().active_tag_name()
                        : "external";

    ss << FLOG_OUTPUT_BROWN("[warn ") << FLOG_OUTPUT_PURPLE(label);
    if(state::instance().verbose())
      ss << FLOG_OUTPUT_LTGRAY(verbose(file, line));
    ss << FLOG_OUTPUT_CYAN(tag);
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
  static constexpr bool strip() {
    return FLOG_STRIP_LEVEL > 3;
  }

  static bool
  format(std::stringstream & ss, const char * file, int line, bool devel) {
    std::string label = devel ? "(devel) " : "";
    std::string tag = state::instance().initialized()
                        ? state::instance().active_tag_name()
                        : "external";

    ss << FLOG_OUTPUT_RED("[ERROR ") << FLOG_OUTPUT_PURPLE(label);
    if(state::instance().verbose())
      ss << FLOG_OUTPUT_LTGRAY(verbose(file, line));
    ss << FLOG_OUTPUT_CYAN(tag);
    ss << FLOG_OUTPUT_GREEN(process_stamp);
    ss << FLOG_OUTPUT_LTBLUE(thread_stamp);
    ss << FLOG_OUTPUT_RED("] ") << std::endl << FLOG_COLOR_LTRED;

    return true;
  }
}; // struct error

} // namespace log
} // namespace flecsi
