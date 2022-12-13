// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_LOG_TAG_SCOPE_HH
#define FLECSI_LOG_TAG_SCOPE_HH

#include <flecsi-config.h>

#if defined(FLECSI_ENABLE_FLOG)

#include "flecsi/log/state.hh"

/// \cond core
namespace flecsi {
namespace log {
/// \addtogroup flog
/// \{

/*!
  This type sets the active tag id to the id passed to the constructor,
  stashing the current active tag. When the instance goes out of scope,
  the active tag is reset to the stashed value.
 */

struct tag_scope_t {
  tag_scope_t(size_t tag = 0) {
#if defined(FLOG_ENABLE_DEBUG)
    std::cerr << FLOG_COLOR_LTGRAY << "FLOG: activating tag " << tag
              << FLOG_COLOR_PLAIN << std::endl;
#endif

    // Warn users about externally-scoped messages
    if(state::instance)
      stash_ = std::exchange(state::active_tag(), tag);
    else {
      std::cerr
        << FLOG_COLOR_YELLOW << "FLOG: !!!WARNING You cannot use "
        << "tag guards for externally scoped messages!!! "
        << "This message will be active if FLOG_ENABLE_EXTERNAL is defined!!!"
        << FLOG_COLOR_PLAIN << std::endl;
    } // if
  } // tag_scope_t

  ~tag_scope_t() {
    if(state::instance)
      state::active_tag() = stash_;
  } // ~tag_scope_t

private:
  size_t stash_;

}; // tag_scope_t

/// \}
} // namespace log
} // namespace flecsi
/// \endcond

#endif // FLECSI_ENABLE_FLOG

#endif
