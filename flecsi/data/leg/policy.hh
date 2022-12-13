// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_LEG_POLICY_HH
#define FLECSI_DATA_LEG_POLICY_HH

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
/// \defgroup legion-data Legion Data
/// Owning wrappers for Legion objects.
/// \ingroup data
/// \{

constexpr inline Legion::ProjectionID def_proj = 0;
constexpr inline int region_dimensions = 2;

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
struct shared_handle {
  shared_handle() = default;
  shared_handle(T t) : h(t) {}
  shared_handle(const shared_handle & s) noexcept : h(s.h) {
    run().create_shared_ownership(ctx(), h);
  }
  shared_handle(shared_handle && s) noexcept : h(std::exchange(s.h, {})) {}
  // Prevent erroneous conversion through T constructor:
  template<class U>
  shared_handle(const shared_handle<U> &) = delete;
  ~shared_handle() {
    if(ctx()) // does the Legion Context still exist?
      destroy(h);
  }
  shared_handle & operator=(shared_handle s) noexcept {
    std::swap(h, s.h);
    return *this;
  }
  explicit operator bool() {
    return h.exists();
  }
  operator T() const {
    return h;
  }
  const T * operator->() const {
    return &h;
  }

private:
  T h;
};

using shared_index_space = shared_handle<Legion::IndexSpace>;
using shared_index_partition = shared_handle<Legion::IndexPartition>;
using shared_field_space = shared_handle<Legion::FieldSpace>;
using shared_logical_region = shared_handle<Legion::LogicalRegion>;

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
  shared_handle ret(t);
  if(n)
    run().attach_name(t, n);
  return ret;
}
// Avoid non-type-erased shared_handle specializations:
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
    : logical_region([&] { // TIP: IIFE (q.v.) allows statements here
        auto & r = run();
        const auto c = ctx();
        shared_index_space index_space(
          named(r.create_index_space(c,
                  Legion::Rect<2>(
                    {0, 0}, Legion::Point<2>(upper(s.first), upper(s.second)))),
            name));
        shared_field_space field_space(r.create_field_space(c));
        Legion::FieldAllocator allocator =
          r.create_field_allocator(c, field_space);
        for(auto const & fi : fs) {
          allocator.allocate_field(fi->type_size, fi->fid);
          r.attach_name(field_space, fi->fid, fi->name.c_str());
        }

        return named(
          r.create_logical_region(c, index_space, field_space), name);
      }()) {}

  // Retain value semantics:
  region(region &&) = default;
  region & operator=(region &&) & = default;

  Legion::IndexSpace get_index_space() const {
    return logical_region->get_index_space();
  }

  Legion::FieldSpace get_field_space() const {
    return logical_region->get_field_space();
  }

  size2 size() const {
    const auto p = run().get_index_space_domain(get_index_space()).hi();
    return size2(p[0] + 1, p[1] + 1);
  }

  shared_logical_region logical_region;
};

struct partition_base {
  shared_index_partition index_partition;
  Legion::LogicalPartition logical_partition;

  // Retain value semantics:
  partition_base(partition_base &&) = default;
  partition_base & operator=(partition_base &&) & = default;

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
  template<topo::single_space>
  const partition_base & get_partition() const {
    return *this;
  }

protected:
  partition_base(const Legion::LogicalRegion & r, shared_index_partition ip)
    : index_partition(std::move(ip)),
      logical_partition(log(r, index_partition)) {}

  static Legion::LogicalPartition log(const Legion::LogicalRegion & r,
    const Legion::IndexPartition & p) {
    return named(run().get_logical_partition(r, p),
      (std::string(1, '{') + name(r, "?") + '/' + name(p, "?") + '}').c_str());
  }
};

struct rows : partition_base {
  explicit rows(const region & reg) : rows(reg.logical_region, reg.size()) {}

  // this constructor will create partition by rows with s.first being number
  // of colors and s.second the max size of the rows
  rows(const Legion::LogicalRegion & reg, size2 s)
    : partition_base(reg,
        partition_rows(reg,
          shared_index_space(run().create_index_space(ctx(),
            Legion::Rect<1>(0, upper(s.first)))),
          upper(s.second))) {}

private:
  shared_index_partition partition_rows(const Legion::LogicalRegion & reg,
    Legion::IndexSpace color_space,
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
    auto ip = partition_rows(reg, get_color_space(), hi[1]);

    logical_partition = log(reg, ip);
    index_partition = std::move(ip);
  }
};

/// Common dependent partitioning facility.
/// \tparam R use ranges (\c rect instead of \c Point<2>)
template<bool R = true>
struct partition : partition_base {
  partition(region & reg,
    const partition_base & src,
    field_id_t fid,
    completeness cpt = incomplete)
    : partition(reg.logical_region, reg.get_index_space(), src, fid, cpt) {}

  partition(prefixes & reg,
    const partition_base & src,
    field_id_t fid,
    completeness cpt = {});

  partition remake(const partition_base & src,
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
  partition(const Legion::LogicalRegion & reg,
    const Legion::IndexSpace & is,
    const partition_base & src,
    field_id_t fid,
    completeness cpt)
    : partition_base(reg,
        named(
          [&r = run()](auto &&... aa) {
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
            partitionKind(R ? disjoint : compute, cpt)),
          (name(src.logical_partition, "?") + std::string("->")).c_str())) {}
};

/// \}
} // namespace leg

using region_base = leg::region;
using partition = leg::partition_base;
using leg::rows;

template<typename T>
T
get_scalar_from_accessor(const T * ptr) {
  if(Legion::Processor::get_executing_processor().kind() ==
     Legion::Processor::TOC_PROC) {
#if defined(__NVCC__) || defined(__CUDACC__)
    T tmp;
    cudaMemcpy(&tmp, ptr, sizeof(T), cudaMemcpyDeviceToHost);
    return tmp;
#else
    flog_assert(false, "Cuda should be enabled when using toc task");
#endif
  }
  return *ptr;
}

} // namespace data
} // namespace flecsi

#endif
