// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

// High-level partition types used for copy plans and user-level topologies.

#ifndef FLECSI_DATA_COPY_HH
#define FLECSI_DATA_COPY_HH

#include "flecsi/data/topology.hh"

#if FLECSI_BACKEND == FLECSI_BACKEND_legion
#include "flecsi/data/leg/copy.hh"
#elif FLECSI_BACKEND == FLECSI_BACKEND_mpi
#include "flecsi/data/mpi/copy.hh"
#endif

/// \cond core
namespace flecsi::data {
/// \addtogroup topology-data
/// \{

#ifdef DOXYGEN // implemented per-backend
/// A prefix of each row in a region_base.
struct prefixes : partition, prefixes_base {
  /// Derive row lengths from a field accessor.  The partition hosting the
  /// field must have the same number of rows as \a r.
  /// \param r region to subdivide
  /// \param f field accessor for \c Field
  template<class F>
  prefixes(region_base & r, F f);

  /// Has the same effect as the constructor, reusing the same region_base.
  template<class F>
  void update(F);
};

/// A subset of each row in a region_base, expressed as a set of intervals.
struct intervals {
  /// Factory function for the interval type.
  static auto make(subrow, std::size_t r = color());
  /// Defined by the backend.
  using Value = decltype(make({}));

  /// Derive intervals from the field values in each row of a partition.
  /// \param r region of which to select a subset
  /// \param part must have the same number of rows as \a r
  /// \param id field of type \c Value
  intervals(region_base & r,
    const partition & part,
    field_id_t id,
    completeness = incomplete);
};

/// Performs a specified copy operation repeatedly.
struct copy_engine {
  /// Factory function for the \c Point type.
  /// \param r row
  /// \param i index (within row)
  static auto point(std::size_t r, std::size_t i);
  /// Defined by the backend.
  using Point = decltype(point(0, 0));

  /// Prepare to copy into intervals.
  /// \param id field of type \c Point referring into \a src; must not
  ///   be mutated while using this value
  copy_engine(const prefixes & src, const intervals & dest, field_id_t id);

  /// Copy fields from \a src to \a dest.
  void operator()(const std::vector<field_id_t> &) const;
};
#endif

/// \}
} // namespace flecsi::data
/// \endcond

#endif
