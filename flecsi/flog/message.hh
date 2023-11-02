// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_LOG_MESSAGE_HH
#define FLECSI_LOG_MESSAGE_HH

#include "flecsi/config.hh"
#include "flecsi/flog/state.hh"
#include "flecsi/flog/types.hh"
#include "flecsi/flog/utils.hh"

#include <iostream>
#include <sstream>

/// \cond core
namespace flecsi {
namespace flog {
/// \addtogroup flog
/// \{

/*!
  The message type provides a basic log message type that is customized
  with a formatting policy.
 */

template<typename Policy>
struct message {

  message(const char * file, int line, bool devel = false)
    : file_(file), line_(line), devel_(devel) {
#if defined(FLOG_ENABLE_DEBUG)
    std::cerr << FLOG_COLOR_LTGRAY << "FLOG: log_message_t constructor " << file
              << " " << line << FLOG_COLOR_PLAIN << std::endl;
#endif
    if(!state::instance().active_process())
      ss_.clear(std::ios_base::badbit);
  }

  ~message() {
    if(Policy::strip() || !state::tag_enabled() ||
       !state::instance().active_process()) {
      return;
    } // if

#if defined(FLOG_ENABLE_DEBUG)
    std::cerr << FLOG_COLOR_LTGRAY << "FLOG: log_message_t destructor "
              << FLOG_COLOR_PLAIN << std::endl;
#endif

    if(clean_) {
      auto str = ss_.str();
      if(str.back() == '\n') {
        str.pop_back();
        ss_.str(std::move(str));
        ss_.seekp(0, ss_.end);
        ss_ << FLOG_COLOR_PLAIN << '\n';
      }
      else {
        ss_ << FLOG_COLOR_PLAIN;
      }
    } // if

#if defined(FLOG_ENABLE_MPI)
    assert(state::instance);
    state::instance().buffer_output(std::move(ss_).str());
#else
    std::cout << ss_.rdbuf();
#endif // FLOG_ENABLE_MPI
  }

  /*
    Make this type work like std::ostream.
   */

  template<typename T>
  message & operator<<(T const & value) {
    ss_ << value;
    return *this;
  }

  /*
    This handles basic manipulators like std::endl.
   */

  message & operator<<(
    ::std::ostream & (*basic_manipulator)(::std::ostream & stream)) {
    ss_ << basic_manipulator;
    return *this;
  }

  /*
    Invoke the policy's formatting method and return this message.
   */

  message & format() {
    clean_ = state::instance().verbose() >= 0 &&
             Policy::format(ss_, file_, line_, devel_);
    return *this;
  }

  /*
    Conversion to bool for ternary usage.
   */

  operator bool() const {
    return true;
  }

private:
  const char * file_;
  int line_;
  bool devel_;
  bool clean_{false};
  std::stringstream ss_;
}; // message

/// \}
} // namespace flog
} // namespace flecsi
/// \endcond

#endif
