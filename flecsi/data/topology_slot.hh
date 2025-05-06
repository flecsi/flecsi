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

namespace detail {
template<class, class, class = void>
struct accepts_scheduler : std::false_type {};
template<class P, class... AA>
struct accepts_scheduler<P,
  util::types<AA...>,
  decltype(void(P::initialize(std::declval<scheduler &>(),
    std::declval<typename P::slot &>(),
    std::declval<const typename P::coloring &>(),
    std::declval<AA>()...)))> : std::true_type {};
} // namespace detail

/// A movable slot that holds a topology, constructed upon request.
/// Declare a task parameter as a \c topology_accessor to use the topology.
/// \note A \c specialization provides aliases for both these types.
/// \warning No topologies may exist outside of \c start or \c control.
///   If a \c
///   topology_slot outlives that function, use \c #deallocate before it
///   returns.
template<typename Topo>
struct topology_slot : convert_tag {
  using topology = typename Topo::topology;
  using coloring = typename Topo::coloring;

  /// Create the topology.
  /// \param c coloring (perhaps from an \link
  ///   topo::specialization::mpi_coloring `mpi_coloring`\endlink)
  /// \param aa further specialization-specific parameters
  template<typename... AA>
  topology & allocate(scheduler & s, const coloring & c, AA &&... aa) {
    data.emplace(s, c);
    if constexpr(detail::accepts_scheduler<Topo, util::types<AA...>>::value)
      Topo::initialize(s, *this, c, std::forward<AA>(aa)...);
    else
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

  topology & get() {
    flog_assert(data, "topology not allocated");
    return *data;
  }
  const topology & get() const {
    return const_cast<topology_slot &>(*this).get();
  }

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
