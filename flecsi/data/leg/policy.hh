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

#if !defined(__FLECSI_PRIVATE__)
#error Do not include this file directly!
#endif

#if !defined(FLECSI_ENABLE_LEGION)
#error FLECSI_ENABLE_LEGION not defined! This file depends on Legion!
#endif

#include "flecsi/run/backend.hh"
#include "flecsi/run/leg/mapper.hh"

#include <legion.h>

#include <unordered_map>

namespace flecsi {
namespace data {

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
  run().destroy_index_space(ctx(), i);
}
// Legion seems to be buggy with destroying partitions:
inline void
destroy(Legion::IndexPartition) {}
inline void
destroy(Legion::FieldSpace f) {
  run().destroy_field_space(ctx(), f);
}
inline void
destroy(Legion::LogicalRegion r) {
  run().destroy_logical_region(ctx(), r);
}
inline void
destroy(Legion::LogicalPartition) {}

template<class T>
struct unique_handle {
  unique_handle() = default;
  unique_handle(T t) : h(t) {}
  unique_handle(unique_handle && u) noexcept : h(std::exchange(u.h, {})) {}
  // Prevent erroneous conversion through T constructor:
  template<class U>
  unique_handle(unique_handle<U> &&) = delete;
  ~unique_handle() {
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
using unique_logical_partition = unique_handle<Legion::LogicalPartition>;

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
  return named0(p, n);
}

struct region {
  region(size2 s, const fields & fs, const char * name = nullptr)
    : index_space(named(run().create_index_space(ctx(),
                          Legion::Rect<2>({0, 0},
                            Legion::Point<2>(upper(s.first), upper(s.second)))),
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

  bool poll_discard(field_id_t f) {
    return discard.erase(f);
  }

  unique_index_space index_space;
  unique_field_space field_space;
  unique_logical_region logical_region;

protected:
  void vacuous(field_id_t f) {
    discard.insert(f);
  }

private:
  std::set<field_id_t> discard;
};

struct partition_base {
  unique_index_partition index_partition;
  unique_logical_partition logical_partition;

  Legion::IndexSpace get_color_space() const {
    return run().get_index_partition_color_space_name(index_partition);
  }

  // NB: intervals and points are not advertised as deriving from this class.
  std::size_t colors() const {
    return leg::run().get_index_space_domain(get_color_space()).get_volume();
  }
  template<topo::single_space>
  const partition_base & get_partition(field_id_t) const {
    return *this;
  }

protected:
  partition_base(const region & r, unique_index_partition ip)
    : index_partition(std::move(ip)),
      logical_partition(log(r.logical_region, index_partition)) {}

  static unique_logical_partition log(const Legion::LogicalRegion & r,
    const Legion::IndexPartition & p) {
    return named(run().get_logical_partition(r, p),
      (std::string(1, '{') + name(r, "?") + '/' + name(p, "?") + '}').c_str());
  }
};

struct with_color { // for initialization order
  unique_index_space color_space;
};
struct rows : with_color, partition_base {
  explicit rows(const region & reg)
    : rows(reg, run().get_index_space_domain(reg.index_space).hi()) {}

private:
  // The type-erased version assumes a square transformation matrix.
  rows(const region & reg, Legion::DomainPoint hi)
    : with_color{run().create_index_space(ctx(), Legion::Rect<1>(0, hi[0]))},
      partition_base(reg,
        named(run().create_partition_by_restriction(
                ctx(),
                Legion::IndexSpaceT<2>(reg.index_space),
                Legion::IndexSpaceT<1>(color_space),
                [&] {
                  Legion::Transform<2, 1> ret;
                  ret.rows[0].x = 1;
                  ret.rows[1].x = 0;
                  return ret;
                }(),
                {{0, 0}, {0, hi[1]}},
                DISJOINT_COMPLETE_KIND),
          name(reg.index_space))) {}
};

template<bool R = true>
struct partition : partition_base {
  partition(region & reg,
    const partition_base & src,
    field_id_t fid,
    completeness cpt = incomplete)
    : partition_base(reg, part(reg.index_space, src, fid, cpt)) {}

protected:
  void update(const partition_base & src,
    field_id_t fid,
    completeness cpt = incomplete) {
    auto & r = run();
    auto ip = part(r.get_parent_index_space(index_partition), src, fid, cpt);
    logical_partition = log(r.get_parent_logical_region(logical_partition), ip);
    index_partition = std::move(ip); // can't fail
  }

private:
  // We document that src must outlive this partitioning, although Legion is
  // said to support deleting its color space before our partition using it.
  static unique_index_partition part(const Legion::IndexSpace & is,
    const partition_base & src,
    field_id_t fid,
    completeness cpt) {
    auto & r = run();

    Legion::Color part_color =
      r.get_index_partition_color(ctx(), src.index_partition);

    auto tag = flecsi::run::tag_index_partition(part_color);

    return named(
      [&r](auto &&... aa) {
        return R ? r.create_partition_by_image_range(
                     std::forward<decltype(aa)>(aa)...)
                 : r.create_partition_by_image(
                     std::forward<decltype(aa)>(aa)...);
      }(ctx(),
        is,
        src.logical_partition,
        r.get_parent_logical_region(src.logical_partition),
        fid,
        src.get_color_space(),
        partitionKind(R ? disjoint : compute, cpt),
        LEGION_AUTO_GENERATE_ID,
        0,
        tag),
      name(src.index_partition));
  }
};

} // namespace leg

using region_base = leg::region;
using partition = leg::partition_base;
using leg::rows;

template<typename T>
T
get_scalar_from_accessor(const T * ptr) {
  T tmp;
  if(Legion::Processor::get_executing_processor().kind() ==
     Legion::Processor::TOC_PROC) {
#if defined(__NVCC__) || defined(__CUDACC__)
    cudaMemcpy(&tmp, ptr, sizeof(T), cudaMemcpyDeviceToHost);
#else
    flog_assert(false, "Cuda should be enabled when using toc task");
#endif
  }
  else {
    tmp = *ptr;
  }
  return tmp;
}

} // namespace data
} // namespace flecsi
