// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_LOG_TAG_SCOPE_HH
#define FLECSI_LOG_TAG_SCOPE_HH

#include <flecsi-config.hh>

#if defined(FLECSI_ENABLE_FLOG)

#include "flecsi/flog/state.hh"

#include <utility> // exchange

/// \cond core
namespace flecsi {
namespace flog {
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

    stash_ = std::exchange(state::active_tag(), tag);
  } // tag_scope_t

  ~tag_scope_t() {
    state::active_tag() = stash_;
  } // ~tag_scope_t

private:
  size_t stash_;

}; // tag_scope_t

/// \}
} // namespace flog
} // namespace flecsi
/// \endcond

#endif // FLECSI_ENABLE_FLOG

#endif
