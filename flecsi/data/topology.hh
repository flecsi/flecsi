/*
    @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
   /@@/////  /@@          @@////@@ @@////// /@@
   /@@       /@@  @@@@@  @@    // /@@       /@@
   /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
   /@@////   /@@/@@@@@@@/@@       ////////@@/@@
   /@@       /@@/@@//// //@@    @@       /@@/@@
   /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
   //       ///  //////   //////  ////////  //

   Copyright (c) 2020, Triad National Security, LLC
   All rights reserved.
                                                                              */
#pragma once

/*! @file */

#if !defined(__FLECSI_PRIVATE__)
#error Do not include this file directly!
#endif

#include "flecsi/data/backend.hh"
#include "flecsi/data/layout.hh"
#include "flecsi/data/privilege.hh"

#include <map>
#include <set>

namespace flecsi::data {
template<class, layout, class Topo, typename Topo::index_space>
struct field_reference;

#ifdef DOXYGEN // implemented per-backend
// It is not required that any of these types be movable.

// A rectangular abstract array.
struct region_base {
  region(size2, const fields &, const char * = nullptr);

  size2 size() const;

protected:
  // For convenience, we always use rw accessors for certain fields that
  // initially contain no constructed objects; this call indicates that no
  // initialization is needed.
  void vacuous(field_id_t);
};

// A prefix of each row in a region_base.
struct partition {
  // The row type is defined by the backend:
  static auto make_row(std::size_t i, std::size_t n); // "constructor"
  using row = decltype(make_row(0, 0));
  static std::size_t row_size(const row &); // "accessor"

  explicit partition(region_base &); // divides into rows
  // Derives row lengths from the field values (of type 'row') accessed via
  // the partition argument, which must have the same number of rows and have
  // length 1 on each row.  The argument partition must survive
  // until this partition is updated or destroyed.
  partition(region_base &,
    const partition &,
    field_id_t,
    completeness = incomplete);

  std::size_t colors() const;
  // The same effect as the constructor, reusing the same region_base.
  void update(const partition &, field_id_t, completeness = incomplete);
  template<topo::single_space> // for convenience for simple topologies
  const partition & get_partition(field_id_t) const {
    return *this;
  }
};

// A subset of each row in a region_base, expressed as a set of intervals.
struct intervals {
  static auto make(subrow, std::size_t r = color()); // constructor, as above
  using Value = decltype(make({}));

  // Derives intervals from the field values (which should be of type Value)
  // in each row of the argument partition, which must have the same number of
  // rows as the region_base and must outlive this value.
  intervals(const region_base &,
    const partition &,
    field_id_t,
    completeness = incomplete);
};

// A set of elements in a region_base for (not of!) each row.
struct points {
  static auto make(std::size_t r, std::size_t); // constructor, as above
  using Value = decltype(make(0, 0));

  // Derives points from the field values (which should be of type Value)
  // in each row of the intervals argument, which must have the same number of
  // rows as the region_base and must outlive this value.
  // The points need not be unique.
  points(const region_base &,
    const intervals &,
    field_id_t,
    completeness = incomplete);
};

struct copy_engine {
  // The field ID is the same as that to create src.  src and dest must
  // outlive this value; the field values must not be mutated while using it.
  copy_engine(const points & src, const intervals & dest, field_id_t);

  void operator()(field_id_t) const;
};
#endif

struct region : region_base {
  using region_base::region_base;

  template<class D>
  void cleanup(field_id_t f, D d, bool hard = true) {
    // We assume that creating the objects will be successful:
    if(hard)
      destroy.insert_or_assign(f, std::move(d));
    else if(destroy.try_emplace(f, std::move(d)).second)
      vacuous(f);
  }

  // Return whether a copy is needed.
  template<std::size_t P>
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
  template<std::size_t P,
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
  return {s, run::context::instance().get_field_info_store<Topo, Index>()};
}

template<class P>
struct partitioned : region, P {
  template<class... TT>
  partitioned(region && r, TT &&... tt)
    : region(std::move(r)),
      P(static_cast<region &>(*this), std::forward<TT>(tt)...) {}
};

} // namespace flecsi::data
