// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

// High-level partition types used for copy plans and user-level topologies.

#ifndef FLECSI_DATA_COPY_HH
#define FLECSI_DATA_COPY_HH

#include "flecsi/data/topology.hh"

#if FLECSI_BACKEND == FLECSI_BACKEND_legion
#include "flecsi/data/leg/copy.hh"
#endif

/// \cond core
namespace flecsi::data {
/// \addtogroup topology-data
/// \{

#ifdef DOXYGEN // implemented per-backend
/// A prefix of each row in a region_base.
struct prefixes : partition, prefixes_base {
  /// Derive row lengths from a field accessor.  The partition hosting the
  /// field must have the same number of rows as \a r and survive until this
  /// partition is updated or destroyed.
  /// \param r region to subdivide
  /// \param f field accessor for \c Field
  template<class F>
  prefixes(region_base & r, F f);

  /// Has the same effect as the constructor, reusing the same region_base.
  template<class F>
  void update(F);
};

/// A subset of each row in a prefixes, expressed as a set of intervals.
struct intervals {
  /// Factory function for the interval type.
  static auto make(subrow, std::size_t r = color());
  /// Defined by the backend.
  using Value = decltype(make({}));

  /// Derive intervals from the field values in each row of a partition.
  /// \param pfx partition of which to create a subset
  /// \param part must have the same number of rows as \a pfx and outlive this
  ///   value
  /// \param id field of type \c Value
  intervals(prefixes & pfx,
    const partition & part,
    field_id_t id,
    completeness = incomplete);
};

/// A set of elements in a prefixes for (not of!) each row.
struct points {
  /// Factory function for the point type.
  static auto make(std::size_t r, std::size_t);
  /// Defined by the backend.
  using Value = decltype(make(0, 0));

  /// Derive points from the field values in each row of a partition.
  /// \param pfx partition from which to select
  /// \param iv must have the same number of rows as \a pfx and outlive this
  ///   value
  /// \param id field of type \c Value (need not be unique)
  points(prefixes & pfx,
    const intervals & iv,
    field_id_t id,
    completeness = incomplete);
};

/// Performs a specified copy operation repeatedly.
struct copy_engine {
  /// Prepare to copy from points to intervals.
  /// \param src must outlive this value
  /// \param dest similarly
  /// \param id that used to create \a src; must not be mutated while using
  ///   this value
  copy_engine(const points & src, const intervals & dest, field_id_t id);

  /// Copy one field from \a src to \a dest.
  void operator()(field_id_t) const;
};
#endif

/// \}
} // namespace flecsi::data
/// \endcond

#endif
