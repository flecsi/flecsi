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

#include "flecsi/data/field_info.hh"
#include "flecsi/topo/core.hh" // single_space
#include "flecsi/util/array_ref.hh"

#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

namespace flecsi {
namespace data {

namespace mpi {
struct region {
  // The constructor is collectively called on all ranks with the same s,
  // and fs. s.first is number of rows while s.second is number of columns.
  // MPI may assume "number of rows" == number of ranks. The number of columns
  // is a placeholder for the number of data points to be stored in a
  // particular row (aka rank), it could be exact or in the case when we don't
  // know the exact number yet, a large number is used.
  //
  // Generic frontend code supplies `s` and `fs` (for information about fields),
  // requesting memory to be (notionally) reserved from backend. Here we only
  // create the underlying std::vector<> without actually allocating any memory.
  // The client code (mostly mpi::task_prologue) will call .get_storage() with
  // a field id and the number of elements on this rank, as determined by
  // partitioning of the field. We will then call .resize() on the
  // std::vector<>.
  region(size2 s, const fields & fs) : s(std::move(s)), fs(fs) {
    for(auto f : fs) {
      storages.emplace(f->fid, 0);
    }
  }

  size2 size() const {
    return s;
  }

  // The span is safe because it is used only within a user task while the
  // vectors are resized or destroyed only outside user tasks (though perhaps
  // during execute).
  template<class T>
  util::span<T> get_storage(field_id_t fid) {
    return get_storage<T>(fid, s.second);
  }

  template<class T>
  util::span<T> get_storage(field_id_t fid, std::size_t nelems) {
    auto & v = storages.at(fid);
    std::size_t nbytes = nelems * sizeof(T);
    if(nbytes > v.size())
      v.resize(nbytes);
    return {reinterpret_cast<T *>(v.data()), nelems};
  }

protected:
  void vacuous(field_id_t) {}

private:
  size2 s;
  fields fs; // fs[].fid is only unique within a region, i.e. r0.fs[].fid is
             // unrelated to r1.fs[].fid even if they have the same value.

  std::unordered_map<field_id_t, std::vector<std::byte>> storages;
};

struct partition {
  using row = std::size_t;
  static row make_row(std::size_t, std::size_t n) {
    return n;
  }
  static std::size_t row_size(const row & r) {
    return r;
  }

  explicit partition(region & r) : r(&r) {
    // This constructor is usually (almost always) called when r.s.second != 4G
    // meaning it has the actual value. In this case, r.s.second is the number
    // of elements in the partition on this rank (the same value on all ranks
    // though).
    nelems = r.size().second;
  }

  partition(region & r,
    const partition & other,
    field_id_t fid,
    completeness = incomplete)
    : r(&r) {
    // Constructor for the case when how the data is partitioned is stored
    // as a field in another region. Delegate to update()
    update(other, fid);
  }

  std::size_t colors() const {
    // number of rows, essentially the number of MPI ranks.
    return r->size().first;
  }

  template<typename T>
  auto get_storage(field_id_t fid) const {
    return r->get_storage<T>(fid, nelems);
  }

  void
  update(const partition & other, field_id_t fid, completeness = incomplete) {
    // The number of elements for each ranks is stored as a field of the `row`
    // data type on the `other` partition.
    const auto s = other.get_storage<row>(fid); // non-owning span
    flog_assert(
      s.size() == 1, "underlying partition must have size 1, not " << s.size());
    nelems = s[0];
  }

  template<topo::single_space>
  const partition & get_partition(field_id_t) const {
    return *this;
  }

  size_t size() const {
    return nelems;
  }

private:
  region * r;
  // number of elements in this partition on this particular rank.
  size_t nelems = 0;
};
} // namespace mpi

// For backend-agnostic interface:
using region_base = mpi::region;
using mpi::partition;

struct intervals {
  using Value = subrow;
  static Value make(subrow r, std::size_t = 0) {
    return r;
  }

  intervals(const region_base &,
    const partition &,
    field_id_t,
    completeness = incomplete) {}
};

struct points {
  using Value = std::pair<std::size_t, std::size_t>;
  static Value make(std::size_t r, std::size_t i) {
    return {r, i};
  }

  // FIXME: Just to silence the compiler, I have no idea what I am doing.
  points(const region_base &,
    const intervals &,
    field_id_t,
    completeness = incomplete) {}
};

inline void
launch_copy(mpi::region &,
  const points &,
  const intervals &,
  const field_id_t &,
  const field_id_t &) {}

} // namespace data
} // namespace flecsi
