// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_LOG_UTILS_HH
#define FLECSI_LOG_UTILS_HH

#include "flecsi/config.hh"

#include <ctime>
#include <string>

/// \cond core
/// \addtogroup flog
/// \{
#define FLOG_COLOR_BLACK flecsi::flog::detail::use_color("\033[0;30m")
#define FLOG_COLOR_DKGRAY flecsi::flog::detail::use_color("\033[1;30m")
#define FLOG_COLOR_RED flecsi::flog::detail::use_color("\033[0;31m")
#define FLOG_COLOR_LTRED flecsi::flog::detail::use_color("\033[1;31m")
#define FLOG_COLOR_GREEN flecsi::flog::detail::use_color("\033[0;32m")
#define FLOG_COLOR_LTGREEN flecsi::flog::detail::use_color("\033[1;32m")
#define FLOG_COLOR_BROWN flecsi::flog::detail::use_color("\033[0;33m")
#define FLOG_COLOR_YELLOW flecsi::flog::detail::use_color("\033[1;33m")
#define FLOG_COLOR_BLUE flecsi::flog::detail::use_color("\033[0;34m")
#define FLOG_COLOR_LTBLUE flecsi::flog::detail::use_color("\033[1;34m")
#define FLOG_COLOR_PURPLE flecsi::flog::detail::use_color("\033[0;35m")
#define FLOG_COLOR_LTPURPLE flecsi::flog::detail::use_color("\033[1;35m")
#define FLOG_COLOR_CYAN flecsi::flog::detail::use_color("\033[0;36m")
#define FLOG_COLOR_LTCYAN flecsi::flog::detail::use_color("\033[1;36m")
#define FLOG_COLOR_LTGRAY flecsi::flog::detail::use_color("\033[0;37m")
#define FLOG_COLOR_WHITE flecsi::flog::detail::use_color("\033[1;37m")
#define FLOG_COLOR_PLAIN flecsi::flog::detail::use_color("\033[0m")

#define FLOG_OUTPUT_BLACK(s) FLOG_COLOR_BLACK << s << FLOG_COLOR_PLAIN
#define FLOG_OUTPUT_DKGRAY(s) FLOG_COLOR_DKGRAY << s << FLOG_COLOR_PLAIN
#define FLOG_OUTPUT_RED(s) FLOG_COLOR_RED << s << FLOG_COLOR_PLAIN
#define FLOG_OUTPUT_LTRED(s) FLOG_COLOR_LTRED << s << FLOG_COLOR_PLAIN
#define FLOG_OUTPUT_GREEN(s) FLOG_COLOR_GREEN << s << FLOG_COLOR_PLAIN
#define FLOG_OUTPUT_LTGREEN(s) FLOG_COLOR_LTGREEN << s << FLOG_COLOR_PLAIN
#define FLOG_OUTPUT_BROWN(s) FLOG_COLOR_BROWN << s << FLOG_COLOR_PLAIN
#define FLOG_OUTPUT_YELLOW(s) FLOG_COLOR_YELLOW << s << FLOG_COLOR_PLAIN
#define FLOG_OUTPUT_BLUE(s) FLOG_COLOR_BLUE << s << FLOG_COLOR_PLAIN
#define FLOG_OUTPUT_LTBLUE(s) FLOG_COLOR_LTBLUE << s << FLOG_COLOR_PLAIN
#define FLOG_OUTPUT_PURPLE(s) FLOG_COLOR_PURPLE << s << FLOG_COLOR_PLAIN
#define FLOG_OUTPUT_LTPURPLE(s) FLOG_COLOR_LTPURPLE << s << FLOG_COLOR_PLAIN
#define FLOG_OUTPUT_CYAN(s) FLOG_COLOR_CYAN << s << FLOG_COLOR_PLAIN
#define FLOG_OUTPUT_LTCYAN(s) FLOG_COLOR_LTCYAN << s << FLOG_COLOR_PLAIN
#define FLOG_OUTPUT_LTGRAY(s) FLOG_COLOR_LTGRAY << s << FLOG_COLOR_PLAIN
#define FLOG_OUTPUT_WHITE(s) FLOG_COLOR_WHITE << s << FLOG_COLOR_PLAIN
/// \}

namespace flecsi {
namespace flog {
/// \addtogroup flog
/// \{

/*!
  Create a timestamp.
 */

inline std::string
timestamp(bool underscores = false) {
  char stamp[14];
  time_t t = time(0);
  std::string format = underscores ? "%m%d_%H%M%S" : "%m%d %H:%M:%S";
  strftime(stamp, sizeof(stamp), format.c_str(), localtime(&t));
  return std::string(stamp);
} // timestamp

/*!
  Strip path from string up to last character C.

  @tparam C The character to strip.
 */

template<char C>
inline std::string
rstrip(const char * file) {
  std::string tmp(file);
  return tmp.substr(tmp.rfind(C) + 1);
} // rstrip

namespace detail {
inline const char * use_color(const char * c);
} // namespace detail

/// \}

} // namespace flog
} // namespace flecsi
/// \endcond

#endif
