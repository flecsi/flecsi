/*
    @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
   /@@/////  /@@          @@////@@ @@////// /@@
   /@@       /@@  @@@@@  @@    // /@@       /@@
   /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
   /@@////   /@@/@@@@@@@/@@       ////////@@/@@
   /@@       /@@/@@//// //@@    @@       /@@/@@
   /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
   //       ///  //////   //////  ////////  //

   Copyright (c) 2016, Triad National Security, LLC
   All rights reserved.
                                                                              */
#pragma once

/*!  @file */

#if !defined(FLECSI_ENABLE_LEGION)
#error FLECSI_ENABLE_LEGION not defined! This file depends on Legion!
#endif

#include "flecsi/run/leg/mapper.hh"

#include <legion.h>

#include <unordered_map>

namespace flecsi {
namespace data {

struct prefixes;

enum disjointness { compute = 0, disjoint = 1, aliased = 2 };
constexpr auto
partitionKind(disjointness dis, completeness cpt) {
  return Legion::PartitionKind((dis + 2) % 3 + 3 * cpt);
}

static_assert(static_cast<Legion::coord_t>(logical_size) == logical_size,
  "logical_size too large for Legion");

namespace leg {
constexpr inline Legion::ProjectionID def_proj = 0;

inline auto &
run() {
  return *Legion::Runtime::get_runtime();
}
inline auto
ctx() {
  return Legion::Runtime::get_context();
}

// Legion uses a number of "handle" types that are non-owning identifiers for
// inaccessible objects maintained by the runtime.  By wrapping their deletion
// functions in a uniform interface, we can use normal RAII hereafter.

inline void
destroy(Legion::IndexSpace i) {
  run().destroy_index_space(ctx(), i, false, false);
}
inline void
destroy(Legion::IndexPartition p) {
  run().destroy_index_partition(ctx(), p, false, false);
}
inline void
destroy(Legion::FieldSpace f) {
  run().destroy_field_space(ctx(), f);
}
inline void
destroy(Legion::LogicalRegion r) {
  run().destroy_logical_region(ctx(), r);
}

template<class T>
struct unique_handle {
  unique_handle() = default;
  unique_handle(T t) : h(t) {}
  unique_handle(unique_handle && u) noexcept : h(std::exchange(u.h, {})) {}
  // Prevent erroneous conversion through T constructor:
  template<class U>
  unique_handle(unique_handle<U> &&) = delete;
  ~unique_handle() {
    if(ctx()) // does the Legion Context still exist?
      if(*this) // empty LogicalRegions, at least, cannot be deleted
        destroy(h);
  }
  unique_handle & operator=(unique_handle u) noexcept {
    std::swap(h, u.h);
    return *this;
  }
  explicit operator bool() {
    return h.exists();
  }
  operator T() const {
    return h;
  }

private:
  T h;
};

using unique_index_space = unique_handle<Legion::IndexSpace>;
using unique_index_partition = unique_handle<Legion::IndexPartition>;
using unique_field_space = unique_handle<Legion::FieldSpace>;
using unique_logical_region = unique_handle<Legion::LogicalRegion>;

using rect = Legion::Rect<2>;

// NB: n=0 works because Legion interprets inverted ranges as empty.
inline Legion::coord_t
upper(std::size_t n) {
  return static_cast<Legion::coord_t>(n) - 1;
}

template<class T>
const char *
name(const T & t, const char * def = nullptr) {
  // NB: retrieve_name aborts if no name is set.
  const void * ret;
  std::size_t sz;
  return run().retrieve_semantic_information(
           t, NAME_SEMANTIC_TAG, ret, sz, true)
           ? static_cast<const char *>(ret)
           : def;
}
template<class T>
auto
named0(const T & t, const char * n) {
  unique_handle ret(t);
  if(n)
    run().attach_name(t, n);
  return ret;
}
// Avoid non-type-erased unique_handle specializations:
inline auto
named(const Legion::IndexSpace & s, const char * n) {
  return named0(s, n);
}
inline auto
named(const Legion::IndexPartition & p, const char * n) {
  return named0(p, n);
}
inline auto
named(const Legion::LogicalRegion & r, const char * n) {
  return named0(r, n);
}
inline auto
named(const Legion::LogicalPartition & p, const char * n) {
  if(n)
    run().attach_name(p, n);
  return p;
}

struct region {
  region(size2 s, const fields & fs, const char * name = nullptr)
    : index_space(named(run().create_index_space(ctx(),
                          rect({0, 0}, {upper(s.first), upper(s.second)})),
        name)),
      field_space([&fs] { // TIP: IIFE (q.v.) allows statements here
        auto & r = run();
        const auto c = ctx();
        unique_field_space ret = r.create_field_space(c);
        Legion::FieldAllocator allocator = r.create_field_allocator(c, ret);
        for(auto const & fi : fs) {
          allocator.allocate_field(fi->type_size, fi->fid);
          r.attach_name(ret, fi->fid, fi->name.c_str());
        }
        return ret;
      }()),
      logical_region(
        named(run().create_logical_region(ctx(), index_space, field_space),
          name)) {}

  size2 size() const {
    const auto p = run().get_index_space_domain(index_space).hi();
    return size2(p[0] + 1, p[1] + 1);
  }

  unique_index_space index_space;
  unique_field_space field_space;
  unique_logical_region logical_region;
};

struct partition_base {
  unique_index_partition index_partition;
  Legion::LogicalPartition logical_partition;

  Legion::IndexSpace get_color_space() const {
    return run().get_index_partition_color_space_name(index_partition);
  }
  Legion::LogicalRegion root() const { // required for using privileges
    auto & r = run();
    Legion::LogicalPartition lp = logical_partition;
    while(true) {
      auto lr = r.get_parent_logical_region(lp);
      if(!r.has_parent_logical_partition(lr))
        return lr;
      lp = r.get_parent_logical_partition(lr);
    }
  }

  // NB: intervals and points are not advertised as deriving from this class.
  Color colors() const {
    return run().get_index_space_domain(get_color_space()).get_volume();
  }

protected:
  partition_base(const Legion::LogicalRegion & r, unique_index_partition ip)
    : index_partition(std::move(ip)),
      logical_partition(log(r, index_partition)) {}

  static Legion::LogicalPartition log(const Legion::LogicalRegion & r,
    const Legion::IndexPartition & p) {
    return named(run().get_logical_partition(r, p),
      (std::string(1, '{') + name(r, "?") + '/' + name(p, "?") + '}').c_str());
  }
};

} // namespace leg

// This type must be defined outside of namespace leg to support
// forward declarations
struct partition : leg::partition_base { // instead of "using partition ="
  using leg::partition_base::partition_base;

  template<topo::single_space>
  partition & get_partition(field_id_t) {
    return *this;
  }
};

namespace leg {

struct with_color { // for initialization order
  unique_index_space color_space;
};
struct rows : with_color, partition {
  explicit rows(const region & reg) : rows(reg.logical_region, reg.size()) {}

  // this constructor will create partition by rows with s.first being number
  // of colors and s.second the max size of the rows
  rows(const Legion::LogicalRegion & reg, size2 s)
    : with_color{run().create_index_space(ctx(),
        Legion::Rect<1>(0, upper(s.first)))},
      partition(reg, partition_rows(reg, upper(s.second))) {}

private:
  unique_index_partition partition_rows(const Legion::LogicalRegion & reg,
    Legion::coord_t hi) {
    // The type-erased version assumes a square transformation matrix
    return named(run().create_partition_by_restriction(
                   ctx(),
                   Legion::IndexSpaceT<2>(reg.get_index_space()),
                   Legion::IndexSpaceT<1>(color_space),
                   [&] {
                     Legion::Transform<2, 1> ret;
                     ret.rows[0].x = 1;
                     ret.rows[1].x = 0;
                     return ret;
                   }(),
                   {{0, 0}, {0, hi}},
                   DISJOINT_COMPLETE_KIND),
      (name(reg.get_index_space(), "?") + std::string(1, '=')).c_str());
  }

public:
  void update(const Legion::LogicalRegion & reg) {
    Legion::DomainPoint hi =
      run().get_index_space_domain(reg.get_index_space()).hi();
    auto ip = partition_rows(reg, hi[1]);

    logical_partition = log(reg, ip);
    index_partition = std::move(ip);
  }
};

/// \internal Common dependent partitioning facility.
/// \tparam R use ranges (\c rect instead of \c Point<2>)
/// \tparam D assume disjoint partitions
template<bool R = true, bool D = R>
struct partition : data::partition {
  partition(region & reg,
    const data::partition & src,
    field_id_t fid,
    completeness cpt = incomplete)
    : partition(reg.logical_region, reg.index_space, src, fid, cpt) {}

  partition(prefixes & reg,
    const data::partition & src,
    field_id_t fid,
    completeness cpt = {});

  partition remake(const data::partition & src,
    field_id_t fid,
    completeness cpt = incomplete) const {
    auto & r = run();
    return {r.get_parent_logical_region(logical_partition),
      r.get_parent_index_space(index_partition),
      src,
      fid,
      cpt};
  }

private:
  // We document that src must outlive this partitioning, although Legion is
  // said to support deleting its color space before our partition using it.
  partition(const Legion::LogicalRegion & reg,
    const Legion::IndexSpace & is,
    const data::partition & src,
    field_id_t fid,
    completeness cpt)
    : data::partition(reg,
        named(
          [& r = run()](auto &&... aa) {
            return R ? r.create_partition_by_image_range(
                         std::forward<decltype(aa)>(aa)...)
                     : r.create_partition_by_image(
                         std::forward<decltype(aa)>(aa)...);
          }(ctx(),
            is,
            src.logical_partition,
            src.root(),
            fid,
            src.get_color_space(),
            partitionKind(D ? disjoint : compute, cpt)),
          (name(src.logical_partition, "?") + std::string("->")).c_str())) {}
};

struct borrow : partition<true, false> {
  using Value = rect;

  static Value make(prefixes_base::row r,
    std::size_t c = run::context::instance().color()) {
    const Legion::coord_t i = c;
    return {{i, 0}, {i, upper(r)}};
  }

  using partition::partition;
};
} // namespace leg

using region_base = leg::region;
using leg::rows, leg::borrow;

} // namespace data
} // namespace flecsi
