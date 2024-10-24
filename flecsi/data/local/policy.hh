// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_LOCAL_POLICY_HH
#define FLECSI_DATA_LOCAL_POLICY_HH

#include "flecsi/data/field_info.hh"
#include "flecsi/exec/task_attributes.hh"
#include "flecsi/run/backend.hh"
#include "flecsi/util/array_ref.hh"
#include "flecsi/util/mpi.hh"

#include <cstddef>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace flecsi {
namespace data {

namespace local {
struct region_impl {
  // The constructor is collectively called on all ranks with the same s,
  // and fs. s.first is number of rows while s.second is number of columns.
  // This file assumes "number of rows" == number of ranks. The number of
  // columns is a placeholder for the number of data points to be stored in a
  // particular row (aka rank), it could be exact or in the case when we don't
  // know the exact number yet, a large number (flecsi::data::logical_size) is
  // used.
  //
  // Generic frontend code supplies `s` and `fs` (for information about fields),
  // requesting memory to be (notionally) reserved from backend. Here we only
  // create the underlying std::vector<> without actually allocating any memory.
  // The client code (mostly exec::task_prologue) will call .get_storage() with
  // a field id and the number of elements on this rank, as determined by
  // partitioning of the field. We will then call .resize() on the
  // std::vector<>.
  region_impl(size2 s, const fields & fs) : s(std::move(s)), fs(fs) {
    for(const auto & f : fs) {
      storages[f->fid];
    }
  }

  size2 size() const {
    return s;
  }

  // Specifies the correct const-qualified span object given access privilege
  template<class T, partition_privilege_t AccessPrivilege>
  using span_access = flecsi::util::span<privilege_const<T, AccessPrivilege>>;

  // The span is safe because it is used only within a user task while the
  // vectors are resized or destroyed only outside user tasks (though perhaps
  // during execute).
  template<class T,
    exec::task_processor_type_t ProcessorType =
      exec::task_processor_type_t::loc,
    partition_privilege_t AccessPrivilege = partition_privilege_t::ro>
  auto get_storage(field_id_t fid) {
    return get_storage<T, ProcessorType, AccessPrivilege>(fid, s.second);
  }

  template<class T,
    exec::task_processor_type_t ProcessorType =
      exec::task_processor_type_t::loc,
    partition_privilege_t AccessPrivilege = partition_privilege_t::ro>
  auto get_storage(field_id_t fid, std::size_t nelems) {
    using return_type = span_access<T, AccessPrivilege>;

    auto & v = storages.at(fid);
    std::size_t nbytes = nelems * sizeof(T);
    if(nbytes > v.size())
      v.resize(nbytes);

    return return_type{reinterpret_cast<typename return_type::pointer>(
                         v.data<ProcessorType, AccessPrivilege>()),
      nelems};
  }

  backend_storage & operator[](field_id_t fid) {
    return storages.at(fid);
  }

  template<partition_privilege_t AccessPrivilege = partition_privilege_t::ro>
  auto kokkos_view(field_id_t fid) {
    return storages.at(fid).kokkos_view<AccessPrivilege>();
  }

  auto get_field_info(field_id_t fid) const {
    for(auto & f : fs) {
      if(f->fid == fid)
        return f;
    }
    throw std::runtime_error("can not find field");
  }

private:
  size2 s; // (nranks, nelems)
  fields fs; // fs[].fid is only unique within a region, i.e. r0.fs[].fid is
             // unrelated to r1.fs[].fid even if they have the same value.

  std::unordered_map<field_id_t, backend_storage> storages;
};

struct region {
  using ref = std::shared_ptr<region_impl>;

  region(size2 s, const fields & fs, const char * = nullptr)
    : p(std::make_shared<region_impl>(s, fs)) {}
  region(region &&) = default;
  region & operator=(region &&) & = default;

  size2 size() const {
    return p->size();
  }

  void partition_notify() {}
  void partition_notify(field_id_t) {}

  decltype(auto) operator[](field_id_t fid) const {
    return (*p)[fid];
  }

  ref share() {
    return p;
  }

  region_impl & operator*() {
    return *p;
  }

  region_impl * operator->() {
    return p.get();
  }

private:
  ref p; // to preserve an address on move
};

struct partition_impl {

  Color colors() const {
    // number of rows, essentially the number of MPI ranks.
    return r->size().first;
  }

  decltype(auto) operator[](field_id_t fid) const {
    return (*r)[fid];
  }

  template<typename T,
    exec::task_processor_type_t ProcessorType =
      exec::task_processor_type_t::loc,
    partition_privilege_t AccessPrivilege = partition_privilege_t::ro>
  auto get_storage(field_id_t fid) const {
    return r->get_storage<T, ProcessorType, AccessPrivilege>(fid, nelems);
  }

  template<partition_privilege_t AccessPrivilege>
  auto get_raw_storage(field_id_t fid, std::size_t item_size) const {
    return r->get_storage<std::byte,
      exec::task_processor_type_t::loc,
      AccessPrivilege>(fid, nelems * item_size);
  }

private:
  region::ref r;

public:
  partition_impl(region & r) : r(r.share()) {}
  // number of elements in this partition on this particular rank.
  size_t nelems = 0;
};

// partition makes sure the embedded partition_impl stays stable even if the
// partition itself is moved
struct partition {
  using ref = std::shared_ptr<partition_impl>;

  partition(partition &&) = default;
  partition & operator=(partition &&) & = default;

  Color colors() const {
    // number of rows, essentially the number of MPI ranks.
    return p->colors();
  }

  decltype(auto) operator[](field_id_t fid) const {
    return (*p)[fid];
  }

  ref share() {
    return p;
  }

  partition_impl & operator*() {
    return *p;
  }

  partition_impl * operator->() const {
    return p.get();
  }

protected:
  partition(region & r) : p(std::make_shared<partition_impl>(r)) {}

  ref p; // to preserve an address on move
};

// forward declaration only
struct copy_engine;

} // namespace local

// forward declaration only
struct copy_engine;

// This type must be defined outside of namespace local to support
// forward declarations
struct partition : local::partition { // instead of "using partition ="

  using local::partition::partition;

  template<topo::single_space>
  partition & get_partition() {
    return *this;
  }
};

namespace local {
struct rows : data::partition {
  explicit rows(region & r) : partition(r) {
    // This constructor is usually (almost always) called when r.s.second != a
    // large number, meaning it has the actual value. In this case, r.s.second
    // is the number of elements in the partition on this rank (the same value
    // on all ranks though).
    (*this)->nelems = r.size().second;
  }
};

struct prefixes : data::partition, prefixes_base {
  template<class F>
  prefixes(region & r, F f) : partition(r) {
    // Constructor for the case when how the data is partitioned is stored
    // as a field in another region referenced by the "other' partition.
    // Delegate to update().
    update(std::move(f));
  }

  template<class F>
  void update(F f) {
    // The number of elements for each ranks is stored as a field of the
    // prefixes::row data type on the `other` partition.
    auto & part = f.get_partition();
    // Make sure storage is actually available
    part[f.fid()].synchronize();
    const auto s = part->template get_storage<row>(f.fid()); // non-owning span
    flog_assert(
      s.size() == 1, "underlying partition must have size 1, not " << s.size());
    (*this)->nelems = s[0];
  }

  size_t size() const {
    return (*this)->nelems;
  }
};
} // namespace local

// For backend-agnostic interface:
using region_base = local::region;
using rows = local::rows;
using prefixes = local::prefixes;

struct borrow : borrow_base {
  borrow(Claims c) {
    auto & ctx = run::context::instance();
    if(c.size() != ctx.processes())
      flog_fatal("backend limited: one selection per process needed");
    auto p = ctx.process();
    const Claim i = c[p];
    sel = i != nil;
    if(sel && i != p)
      flog_fatal("backend limited: no cross-color access");
  }

  Color size() const {
    return run::context::instance().processes();
  }

  bool selected() const {
    return sel;
  }

private:
  bool sel;
};

struct intervals {
  using Value = subrow; // [begin, end)
  static Value make(subrow r, std::size_t = 0) {
    return r;
  }

  intervals(region_base & r,
    const partition & p,
    field_id_t fid, // The field id for the metadata in the region in p.
    completeness = incomplete)
    : r(&*r) {
    // Make sure the task that is writing to the field has finished running
    p[fid].synchronize();
    // Called by upper layer, supplied with a region and a partition. There are
    // two regions involved. The region `r` has the storage for real field data
    // (e.g. density, pressure etc.) as the destination of the ghost copy. It
    // also contains the pairs of (rank, index) of shared entities on remote
    // peers. The region and associated storage in the partition `p` contains
    // metadata on which entities are local ghosts (destination of copy) on the
    // current rank. The metadata is in the form of [beginning index, ending
    // index), type aliased as Value, into the index space of the entity (e.g.
    // vertex, edge, cell). We thus need to use the p.get_storage() to get the
    // metadata, not the region.get_storage() which gives the real data. This
    // works the same way as how one partition stores the number of elements in
    // a particular 'shard' on a rank for another partition. In addition, we
    // also need to copy Values from the partition and save them locally. User
    // code might change it after this constructor returns. We can not use a
    // copy assignment directly here since metadata is an util::span while
    // ghost_ranges is a std::vector<>.
    ghost_ranges = to_vector(p->get_storage<Value,
                             exec::task_processor_type_t::loc,
                             partition_privilege_t::ro>(fid));

    // Get The largest value of `end index` in ghost_ranges (i.e. the upper
    // bound). This tells how much memory needs to be allocated for ghost
    // entities.
    if(auto iter = std::max_element(ghost_ranges.begin(),
         ghost_ranges.end(),
         [](Value x, Value y) { return x.second < y.second; });
       iter != ghost_ranges.end()) {
      max_end = iter->second;
    }
  }

private:
  // This member function is only called by copy_engine.
  friend copy_engine;
  friend local::copy_engine;

  template<typename T, partition_privilege_t AccessPrivilege>
  auto get_storage(field_id_t fid) const {
    return r->get_storage<T, exec::task_processor_type_t::loc, AccessPrivilege>(
      fid, max_end);
  }

  decltype(auto) operator[](field_id_t fid) const {
    return (*r)[fid];
  }

  local::region_impl * r;

  // Locally cached metadata on ranges of ghost index.
  std::vector<Value> ghost_ranges;
  std::size_t max_end = 0;
};

struct points {
  using Value = std::pair<std::size_t, std::size_t>; // (rank, index)
  static Value make(std::size_t r, std::size_t i) {
    return {r, i};
  }

  points(region_base & r,
    const intervals &,
    field_id_t,
    completeness = incomplete)
    : r(&*r) {}

private:
  // The region `r` contains field data of shared entities on this rank as
  // source to be copied to remote peers. We make copy_engine a friend to allow
  // direct access to the region.
  friend copy_engine;

  local::region_impl * r;
};

} // namespace data
} // namespace flecsi

#endif // FLECSI_DATA_LOCAL_POLICY_HH
