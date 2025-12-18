// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_LOCAL_POLICY_HH
#define FLECSI_DATA_LOCAL_POLICY_HH

// Include this file only after a definition of data::backend_storage.

#include "flecsi/data/field_info.hh"
#include "flecsi/exec/task_attributes.hh"
#include "flecsi/run/backend.hh"

#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace flecsi {
namespace data {

namespace local {
// NB: These own the region_impl objects that contain the storage.
using storage_ptr = std::shared_ptr<backend_storage>;

struct field {
  field() = default;
  field(storage_ptr s, std::size_t n) : s(std::move(s)), n(n) {}

  explicit operator bool() const {
    return !!s;
  }

  backend_storage & storage() const {
    return *s;
  }
  template<class T,
    privilege Priv = ro,
    exec::processor Proc = exec::processor::loc>
  auto as() const {
    return s->as<T, Priv, Proc>(n);
  }

private:
  storage_ptr s;
  std::size_t n;
};

struct region_impl : std::enable_shared_from_this<region_impl> {
  // s.first is never used (anything used must match processes()).
  // s.second is sometimes the placeholder logical_size.
  region_impl(size2 s, const fields & fs) : s(std::move(s)), fs(fs) {
    for(const auto & f : fs) {
      storages[f->fid]; // field memory allocated by get_storage
    }
  }

  size2 size() const {
    return s;
  }

  field prefix(field_id_t f, std::size_t n) {
    return field({weak_from_this().lock(), &storages.at(f)}, n);
  }
  field operator[](field_id_t f) {
    return prefix(f, s.second);
  }

  auto get_field_info(field_id_t fid) const {
    for(auto & f : fs) {
      if(f->fid == fid)
        return f;
    }
    throw std::runtime_error("no such field");
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

struct partition {
  partition(region & r) : r(&*r) {}
  partition(partition &&) = default;
  partition & operator=(partition &&) & = default;

  Color colors() const {
    return r->size().first;
  }

  field operator[](field_id_t f) const {
    return r->prefix(f, nelems);
  }

  template<privilege Priv>
  auto get_raw_storage(field_id_t fid, std::size_t item_size) const {
    return r->prefix(fid, nelems * item_size).as<std::byte, Priv>();
  }

  region_impl & base() const {
    return *r;
  }

  void resize(std::size_t n) {
    if(n > r->size().second)
      throw std::out_of_range("partition larger than region");
    nelems = n;
  }

private:
  region_impl * r;
  size_t nelems = 0; // for this process
};

using storages = std::vector<field>;

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
    resize(r.size().second);
  }
};

struct prefixes : data::partition, prefixes_base {
  template<class F>
  prefixes(region & r, F f) : partition(r) {
    update(std::move(f));
  }

  template<class F>
  void update(F f) {
    const field fld = f.get_partition()[f.fid()];
    // Make sure storage is actually available
    fld.storage().synchronize();
    const auto s = fld.as<size_request>();
    flog_assert(
      s.size() == 1, "underlying partition must have size 1, not " << s.size());
    resize(s[0]);
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
    field_id_t fid,
    completeness = incomplete)
    : r(&*r) {
    const local::field f = p[fid];
    // Make sure the task that is writing to the field has finished running
    f.storage().synchronize();
    // Eagerly read field data, which might legitimately change later.
    ghost_ranges = to_vector(f.as<Value>());
    if(auto iter = std::max_element(ghost_ranges.begin(),
         ghost_ranges.end(),
         [](Value x, Value y) { return x.second < y.second; });
      iter != ghost_ranges.end()) {
      max_end = iter->second;
    }
  }

  auto operator[](field_id_t f) const {
    return r->prefix(f, max_end);
  }

  local::region_impl * r;

  // Locally cached metadata on ranges of ghost index.
  std::vector<Value> ghost_ranges;
  std::size_t max_end = 0; // size of prefix containing all ranges
};

} // namespace data
} // namespace flecsi

#endif // FLECSI_DATA_LOCAL_POLICY_HH
