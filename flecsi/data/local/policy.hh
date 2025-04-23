// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_LOCAL_POLICY_HH
#define FLECSI_DATA_LOCAL_POLICY_HH

// Include this file only after a definition of data::backend_storage.

#include "flecsi/data/field_info.hh"
#include "flecsi/exec/task_attributes.hh"
#include "flecsi/run/backend.hh"
#include "flecsi/util/array_ref.hh"
#include "flecsi/util/mpi.hh"

#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace flecsi {
namespace data {

namespace local {
struct region_impl {
  // s.first is never used (anything used must match the count of ranks).
  // s.second is sometimes the placeholder logical_size.
  region_impl(size2 s, const fields & fs) : s(std::move(s)), fs(fs) {
    for(const auto & f : fs) {
      storages[f->fid]; // field memory allocated by get_storage
    }
  }

  size2 size() const {
    return s;
  }

  // Specifies the correct const-qualified span object given access privilege
  template<class T, privilege Priv>
  using span_access = flecsi::util::span<privilege_const<T, Priv>>;

  template<class T,
    privilege Priv = ro,
    exec::processor Proc = exec::processor::loc>
  auto get_storage(field_id_t fid) {
    return get_storage<T, Priv, Proc>(fid, s.second);
  }

  template<class T, // sometimes erased to be std::byte
    privilege Priv = ro,
    exec::processor Proc = exec::processor::loc>
  auto get_storage(field_id_t fid, std::size_t nelems) {
    using return_type = span_access<T, Priv>;

    auto & v = storages.at(fid);
    std::size_t nbytes = nelems * sizeof(T);
    if(nbytes > v.size())
      v.resize(nbytes);

    auto data_view = v.data<Priv, Proc>();

    flog_assert(nbytes <= data_view.size(),
      "Requested region size larger than allocation");

    return return_type{
      reinterpret_cast<privilege_const<T, Priv> *>(data_view.data()), nelems};
  }

  template<privilege Priv>
  auto current_data(field_id_t fid) {
    return storages.at(fid).template current_data<Priv>();
  }

  backend_storage & operator[](field_id_t fid) {
    return storages.at(fid);
  }

  auto get_field_info(field_id_t fid) const {
    for(auto & f : fs) {
      if(f->fid == fid)
        return f;
    }
    throw std::runtime_error("can not find field");
  }

private:
  size2 s;
  fields fs;

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
    return r->size().first;
  }

  decltype(auto) operator[](field_id_t fid) const {
    return (*r)[fid];
  }

  template<typename T,
    privilege Priv = ro,
    exec::processor Proc = exec::processor::loc>
  auto get_storage(field_id_t fid) const {
    return r->get_storage<T, Priv, Proc>(fid, nelems);
  }

  template<privilege Priv>
  auto get_raw_storage(field_id_t fid, std::size_t item_size) const {
    return r->get_storage<std::byte, Priv>(fid, nelems * item_size);
  }

  region_impl & get_region() {
    return *r;
  }

private:
  region::ref r;

public:
  partition_impl(region & r) : r(r.share()) {}

  void resize(std::size_t n) {
    if(n > r->size().second)
      throw std::out_of_range("partition larger than region");
    nelems = n;
  }

private:
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
    (*this)->resize(r.size().second);
  }
};

struct prefixes : data::partition, prefixes_base {
  template<class F>
  prefixes(region & r, F f) : partition(r) {
    update(std::move(f));
  }

  template<class F>
  void update(F f) {
    auto & part = f.get_partition();
    // Make sure storage is actually available
    part[f.fid()].synchronize();
    const auto s = part->template get_storage<size_request>(f.fid());
    flog_assert(
      s.size() == 1, "underlying partition must have size 1, not " << s.size());
    (*this)->resize(s[0]);
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

struct intervals_impl {
  using Value = subrow; // [begin, end)

  intervals_impl(region_base & r, const partition & p, field_id_t fid)
    : r(&*r) {
    // Make sure the task that is writing to the field has finished running
    p[fid].synchronize();
    // Eagerly read field data, which might legitimately change later.
    ghost_ranges = to_vector(p->get_storage<Value>(fid));
    if(auto iter = std::max_element(ghost_ranges.begin(),
         ghost_ranges.end(),
         [](Value x, Value y) { return x.second < y.second; });
       iter != ghost_ranges.end()) {
      max_end = iter->second;
    }
  }

  template<typename T, privilege Priv>
  auto get_storage(field_id_t fid) const {
    return r->get_storage<T, Priv>(fid, max_end);
  }

  decltype(auto) operator[](field_id_t fid) const {
    return (*r)[fid];
  }

  local::region_impl * r;

  // Locally cached metadata on ranges of ghost index.
  std::vector<Value> ghost_ranges;
  std::size_t max_end = 0; // size of prefix containing all ranges
};

struct intervals {
  using Value = intervals_impl::Value;
  static Value make(subrow r, std::size_t = 0) {
    return r;
  }

  using ref = std::shared_ptr<const intervals_impl>;

  intervals(region_base & r,
    const partition & p,
    field_id_t fid,
    completeness = incomplete)
    : ii(std::make_shared<const intervals_impl>(r, p, fid)) {}
  intervals(intervals &&) = default;
  intervals & operator=(intervals &&) & = default;

  ref share() const {
    return ii;
  }

private:
  ref ii; // for asynchronous use
};

} // namespace data
} // namespace flecsi

#endif // FLECSI_DATA_LOCAL_POLICY_HH
