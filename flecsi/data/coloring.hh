// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_COLORING_HH
#define FLECSI_DATA_COLORING_HH

#include <flecsi/execution.hh>

#include <optional>

namespace flecsi {
namespace data {
/// \addtogroup data
/// \{

/// A \link topo::specialization::mpi_coloring `mpi_coloring`\endlink object,
/// constructed on request.
/// \note Usually accessed as \c Topo::cslot.
/// \deprecated Use \c mpi_coloring directly.
template<class Topo>
struct coloring_slot {
  using color_type = typename Topo::coloring;

  /// Create the \c mpi_coloring.
  /// \return the created \c Topo::coloring object
  template<typename... ARGS>
  [[deprecated("use mpi_coloring")]] color_type & allocate(ARGS &&... args) {
    emplace(std::forward<ARGS>(args)...);
    return get();
  } // allocate

  template<class... AA>
  void emplace(AA &&... aa) { // not deprecated
    execute<task<AA...>, flecsi::mpi>(*this, std::forward<AA>(aa)...);
  }

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
