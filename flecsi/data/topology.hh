// Copyright (c) 2020, Triad National Security, LLC
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
  const partition & get_partition() const {
    return *this;
  }
};

/// All of each row in a region_base.
struct rows : partition {
  /// Divide a region into rows.
  explicit rows(region_base &);
};

/// Read from what might be a device pointer.
/// The backend knows where field data is stored for the current task.
/// \param p pointer to field data
template<typename T>
T get_scalar_from_accessor(const T * p);
#endif

// Adds backend-independent metadata.
struct region : region_base {
  using region_base::region_base;
  // key_type is a bit odd here, but we lack a generic single-type wrapper.
  template<class Topo, typename Topo::index_space S>
  region(size2 s, util::key_type<S, Topo>)
    : region_base(s,
        run::context::instance().field_info_store<Topo, S>(),
        (util::type<Topo>() + '[' + std::to_string(S) + ']').c_str()) {}

  template<class D>
  void cleanup(field_id_t f, D d) {
    // We assume that creating the objects will be successful:
    destroy.insert_or_assign(f, std::move(d));
  }

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
    return (privilege_write(g) || !sw && gr ? dirty.erase(i)
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
  // Each field can have a destructor (for individual field values) registered
  // that is invoked when the field is recreated or the region is destroyed.
  struct finalizer {
    template<class F>
    finalizer(F f) : f(std::move(f)) {}
    finalizer(finalizer && o) noexcept {
      f.swap(o.f); // guarantee o.f is empty
    }
    ~finalizer() {
      if(f)
        f();
    }
    finalizer & operator=(finalizer o) noexcept {
      f.swap(o.f);
      return *this;
    }

  private:
    std::function<void()> f;
  };

  std::map<field_id_t, finalizer> destroy;
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
