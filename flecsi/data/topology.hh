// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_TOPOLOGY_HH
#define FLECSI_DATA_TOPOLOGY_HH

#include "flecsi/data/backend.hh"
#include "flecsi/data/layout.hh"
#include "flecsi/data/privilege.hh"
#include "flecsi/util/constant.hh"
#include "flecsi/util/demangle.hh"

#include <map>
#include <set>

/// \cond core
namespace flecsi::data {
/// \defgroup topology-data Topology implementation
/// These types are movable but may not be copyable.
/// \ingroup data
/// \{

template<class, layout, class Topo, typename Topo::index_space>
struct field_reference;

#ifdef DOXYGEN // implemented per-backend
/// A rectangular abstract array.
struct region_base {
  /// Construct an array of several fields.
  /// \param s total size (perhaps much larger than what is allocated)
  /// \param f fields to define (not all of which need be allocated)
  /// \param n optional name for debugging
  region_base(size2 s, const fields & f, const char * n = nullptr);

  /// Get (bounding) size.
  size2 size() const;
};

/// Base class storing a prefix of each row in a \c region_base.
/// \note No constructors are specified.
struct partition {
  /// Get the number of subsets (also the number of rows).
  Color colors() const;
  /// Convenience function for simple topologies with just one partition.
  /// \return this object
  template<topo::single_space>
  partition & get_partition() {
    return *this;
  }
};

/// All of each row in a region_base.
struct rows : partition {
  /// Divide a region into rows.
  explicit rows(region_base &);
};

/// A selection of rows, typically of a \c data::prefixes object.
struct borrow : borrow_base {
  /// Select rows (or no row for \c nil).
  borrow(Claims);
  /// Get the number of selections (not the number of \e available rows).
  Color size() const;
};
#endif

// Adds backend-independent metadata.
struct region : region_base {
  using region_base::region_base;
  // key_type is a bit odd here, but we lack a generic single-type wrapper.
  template<class Topo, typename Topo::index_space S>
  region(size2 s, util::key_type<S, Topo>)
    : region_base(s,
        run::context::field_info_store<Topo, S>(),
        (util::type<Topo>() + '[' +
          std::to_string(static_cast<std::underlying_type_t<decltype(S)>>(S)) +
          ']')
          .c_str()) {}

  // Return whether a copy is needed.
  template<Privileges P>
  bool ghost(field_id_t i) {
    constexpr auto n = privilege_count(P);
    static_assert(n > 1, "need shared/ghost privileges");
    constexpr auto g = get_privilege(n - 1, P);
    constexpr bool gr = privilege_read(g),
                   sw = privilege_write(get_privilege(n - 2, P));
    // The logic here is constructed to allow a single read/write set access:
    // writing to ghosts, or reading from them without also writing to shared,
    // clears the dirty bit, and otherwise writing to shared sets it.
    // Otherwise, it retains its value (and we don't copy).
    return (privilege_write(g) || (!sw && gr)
               ? dirty.erase(i)
               : sw && !dirty.insert(i).second) &&
           gr;
  }

  // Perform a ghost copy if needed.
  template<Privileges P,
    class T,
    layout L,
    class Topo,
    typename Topo::index_space S>
  void ghost_copy(const field_reference<T, L, Topo, S> & f) {
    constexpr auto np = privilege_count(P);
    static_assert(np == Topo::template privilege_count<S>,
      "privilege-count mismatch between accessor and topology type");
    if constexpr(np > 1)
      if(ghost<P>(f.fid()))
        f.topology().ghost_copy(f);
  }

  template<topo::single_space> // for convenience for simple topologies
  region & get_region() {
    return *this;
  }

private:
  std::set<field_id_t> dirty;
};

template<class Topo, typename Topo::index_space Index = Topo::default_space()>
region
make_region(size2 s) {
  return {s, util::key_type<Index, Topo>()};
}

// Types ending in "ed" indicate that a region is bundled.
template<class P>
struct partitioned : region, P {
  template<class... TT>
  partitioned(region && r, TT &&... tt)
    : region(std::move(r)),
      P(static_cast<region &>(*this), std::forward<TT>(tt)...) {}
};

/// \}
} // namespace flecsi::data
/// \endcond

#endif
