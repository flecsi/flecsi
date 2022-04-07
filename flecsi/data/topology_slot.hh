// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_TOPOLOGY_SLOT_HH
#define FLECSI_DATA_TOPOLOGY_SLOT_HH

#include "flecsi/flog.hh"
#include "flecsi/run/backend.hh"
#include "flecsi/run/context.hh"

#include <optional>

namespace flecsi {
namespace data {
/// \addtogroup data
/// \{

struct convert_tag {}; // must be recognized as a task argument

/// A slot that holds a topology, constructed upon request.
/// Declare a task parameter as a \c topology_accessor to use the topology.
/// \note A \c specialization provides aliases for both these types.
template<typename Topo>
struct topology_slot : convert_tag {
  using core = typename Topo::core;
  using coloring = typename Topo::coloring;

  /// Create the topology.
  /// \param coloring_reference coloring (perhaps from a \c coloring_slot)
  /// \param aa further specialization-specific parameters
  template<typename... AA>
  core & allocate(coloring const & coloring_reference, AA &&... aa) {
    data.emplace(coloring_reference);
    Topo::initialize(*this, coloring_reference, std::forward<AA>(aa)...);
    // TODO:  fix issues with automatic register
    // run::context::instance().add_topology<Topo>(*this);

    return get();
  }

  /// Destroy the topology.
  void deallocate() {
    data.reset();
  } // deallocate

  core & get() {
    flog_assert(data, "topology not allocated");
    return *data;
  }
  const core & get() const {
    return const_cast<topology_slot &>(*this).get();
  }

  core * operator->() {
    return &*data;
  }
  const core * operator->() const {
    return &*data;
  }

private:
  std::optional<core> data;
}; // struct topology_slot

/// \}
} // namespace data
} // namespace flecsi

#endif
