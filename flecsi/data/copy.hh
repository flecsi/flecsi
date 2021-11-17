// High-level partition types used for copy plans and user-level topologies.

#ifndef FLECSI_DATA_COPY_HH
#define FLECSI_DATA_COPY_HH

#include "flecsi/data/topology.hh"

#if FLECSI_RUNTIME_MODEL == FLECSI_RUNTIME_MODEL_legion
#include "flecsi/data/leg/copy.hh"
#elif FLECSI_RUNTIME_MODEL == FLECSI_RUNTIME_MODEL_mpi
#include "flecsi/data/mpi/copy.hh"
#endif

namespace flecsi::data {

#ifdef DOXYGEN // implemented per-backend
// These types are movable but may not be copyable.

// A prefix of each row in a region_base.
struct prefixes : partition, prefixes_base {
  // Derives row lengths from the Field values accessed via the field accessor
  // argument.  The partition hosting the field must have the same number of
  // rows as the region and survive until this partition is updated or
  // destroyed.
  template<class F>
  prefixes(region_base &, F);

  // The same effect as the constructor, reusing the same region_base.
  template<class F>
  void update(F);
};

// A subset of each row in a prefixes, expressed as a set of intervals.
struct intervals {
  // The interval type is defined by the backend:
  static auto make(subrow, std::size_t r = color()); // "constructor"
  using Value = decltype(make({}));

  // Derives intervals from the field values (which should be of type Value)
  // in each row of the argument partition, which must have the same number of
  // rows as the region_base and must outlive this value.
  intervals(prefixes &,
    const partition &,
    field_id_t,
    completeness = incomplete);
};

// A set of elements in a prefixes for (not of!) each row.
struct points {
  static auto make(std::size_t r, std::size_t); // constructor, as above
  using Value = decltype(make(0, 0));

  // Derives points from the field values (which should be of type Value)
  // in each row of the intervals argument, which must have the same number of
  // rows as the region_base and must outlive this value.
  // The points need not be unique.
  points(prefixes &, const intervals &, field_id_t, completeness = incomplete);
};

struct copy_engine {
  // The field ID is the same as that to create src.  src and dest must
  // outlive this value; the field values must not be mutated while using it.
  copy_engine(const points & src, const intervals & dest, field_id_t);

  void operator()(field_id_t) const;
};

struct pointers : partition { // up to one source row number in each row
  static constexpr auto & field; // a flecsi::field<borrow::Value>::definition

  // The claims are not const but are not modified.
  pointers(prefixes &, topo::claims::core &);

  auto operator*(); // field reference onto this partition
};
#endif

} // namespace flecsi::data

#endif
