// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_TOPOLOGY_SLOT_HH
#define FLECSI_DATA_TOPOLOGY_SLOT_HH

#include "flecsi/exec/fwd.hh"
#include "flecsi/flog.hh"
#include "flecsi/run/backend.hh"
#include "flecsi/run/context.hh"
#include "flecsi/util/constant.hh"

#include <optional>

namespace flecsi {
namespace data {
/// \addtogroup data
/// \{

struct convert_tag {}; // must be recognized as a task argument

/// A movable slot that holds a topology, constructed upon request.
/// Declare a task parameter as a \c topology_accessor to use the topology.
/// \note A \c specialization provides aliases for both these types.
/// \warning If a \c topology_slot outlives \c start or \c control, use \c
///   #deallocate before it returns.
/// \deprecated Store topology instances directly or under \c std::unique_ptr.
template<typename Topo>
struct topology_slot : convert_tag {
  using topology = typename Topo::topology;
  using coloring = typename Topo::coloring;

  /// Create the topology.
  /// \param c coloring
  /// \param aa further specialization-specific parameters
  template<typename... AA>
  topology & allocate(scheduler & s, const coloring & c, AA &&... aa) {
    data.emplace(s, c);
    Topo::initialize(*this, c, std::forward<AA>(aa)...);
    // TODO:  fix issues with automatic register
    // run::context::instance().add_topology<Topo>(*this);

    return get();
  }
  /// \deprecated Pass a \c scheduler.
  template<typename... AA>
  [[deprecated("pass a scheduler")]] topology & allocate(const coloring & c,
    AA &&... aa) {
    return allocate(*scheduler::instance, c, std::forward<AA>(aa)...);
  }

  /// Destroy the topology.
  void deallocate() {
    data.reset();
  } // deallocate

  /// Return whether or not this slot is allocated.
  bool is_allocated() const {
    return data.has_value();
  }

  /// Get the topology instance, which must exist.
  topology & get() {
    flog_assert(data, "topology not allocated");
    return *data;
  }
  const topology & get() const {
    return const_cast<topology_slot &>(*this).get();
  }

  /// Access a member of the topology instance, which must exist.
  topology * operator->() {
    return &*data;
  }
  const topology * operator->() const {
    return &*data;
  }

  /// Return the number of colors for the topology, which must exist.
  Color colors() const {
    return get().colors();
  }

private:
  util::move_optional<topology> data;
}; // struct topology_slot

/// \}
} // namespace data
} // namespace flecsi

#endif
