// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_COLORING_HH
#define FLECSI_DATA_COLORING_HH

#include <flecsi/execution.hh>

#include <optional>

namespace flecsi {
namespace data {
/// \addtogroup data
/// \{

/// A coloring object, constructed on request.
/// \tparam Topo specialization that defines\code
/// static coloring color(/* ... */);
/// \endcode
/// \note Usually accessed as \c Topo::cslot.
template<class Topo>
struct coloring_slot {
  using color_type = typename Topo::coloring;

  /// Create the coloring object in an MPI task.
  /// \param args arguments to \c Topo::color
  /// \return the created \c Topo::coloring object
  template<typename... ARGS>
  color_type & allocate(ARGS &&... args) {
    execute<task<ARGS...>, flecsi::mpi>(*this, std::forward<ARGS>(args)...);
    return get();
  } // allocate

  /// Destroy the coloring object.
  void deallocate() {
    coloring.reset();
  } // deallocate

  /// Get the coloring object.
  color_type & get() {
    return *coloring;
  }
  /// Get the coloring object.
  const color_type & get() const {
    return *coloring;
  }

private:
  template<class... AA>
  static void task(coloring_slot & s, AA &&... aa) {
    s.coloring.emplace(Topo::color(std::forward<AA>(aa)...));
  }

  std::optional<color_type> coloring;
};

/// \}
} // namespace data
} // namespace flecsi

#endif
