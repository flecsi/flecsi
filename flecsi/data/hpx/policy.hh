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

#include <flecsi-config.h>

#if !defined(FLECSI_ENABLE_HPX)
#error FLECSI_ENABLE_HPX not defined! This file depends on HPX!
#endif

#include <hpx/modules/collectives.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/modules/serialization.hpp>

#include "flecsi/data/field_info.hh"
#include "flecsi/exec/hpx/future.hh"
#include "flecsi/run/backend.hh"
#include "flecsi/util/array_ref.hh"
#include "flecsi/util/hpx.hh"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace flecsi {
namespace data {

namespace hpx {
struct region_impl {
  // The constructor is collectively called on all ranks with the same s,
  // and fs. s.first is number of rows while s.second is number of columns.
  // HPX may assume "number of rows" == number of ranks. The number of columns
  // is a placeholder for the number of data points to be stored in a
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
    for(auto f : fs) {
      storages.emplace(f->fid, 0);
      futures.emplace(f->fid, ::hpx::shared_future<void>{});
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

  auto get_field_info(field_id_t fid) const {
    for(auto f : fs) {
      if(f->fid == fid)
        return f;
    }
    throw std::runtime_error("can not find field");
  }

  ::hpx::shared_future<void> & get_future(field_id_t fid) {
    return futures.at(fid);
  }

private:
  size2 s; // (nranks, nelems)
  fields fs; // fs[].fid is only unique within a region, i.e. r0.fs[].fid is
             // unrelated to r1.fs[].fid even if they have the same value.

  std::unordered_map<field_id_t, std::vector<std::byte>> storages;
  std::unordered_map<field_id_t, ::hpx::shared_future<void>> futures;
};

struct region {
  region(size2 s, const fields & fs, const char * = nullptr)
    : p(new region_impl(s, fs)) {}

  size2 size() const {
    return p->size();
  }

  region_impl & operator*() const {
    return *p;
  }

private:
  std::unique_ptr<region_impl> p; // to preserve an address on move
};

struct partition {
  Color colors() const {
    // number of rows, essentially the number of MPI ranks.
    return r->size().first;
  }

  template<typename T>
  auto get_storage(field_id_t fid) const {
    return r->get_storage<T>(fid, nelems);
  }

  auto & get_future(field_id_t fid) const {
    return r->get_future(fid);
  }

  template<topo::single_space>
  const partition & get_partition(field_id_t) const {
    return *this;
  }

private:
  region_impl * r;

protected:
  partition(region & r) : r(&*r) {}

  region_impl & get_base() const {
    return *r;
  }

  // number of elements in this partition on this particular rank.
  size_t nelems = 0;
};

} // namespace hpx

// This type must be defined outside of namespace hpx to support
// forward declarations
struct partition
  : flecsi::data::hpx::partition { // instead of "using partition ="
  using flecsi::data::hpx::partition::partition;
};

namespace hpx {

struct rows : partition {
  explicit rows(region & r) : partition(r) {
    // This constructor is usually (almost always) called when r.s.second != a
    // large number, meaning it has the actual value. In this case, r.s.second
    // is the number of elements in the partition on this rank (the same value
    // on all ranks though).
    nelems = r.size().second;
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
    const auto s =
      f.get_partition().template get_storage<row>(f.fid()); // non-owning span
    flog_assert(
      s.size() == 1, "underlying partition must have size 1, not " << s.size());
    nelems = s[0];
  }

  size_t size() const {
    return nelems;
  }

  using partition::get_base;
};
} // namespace hpx

// For backend-agnostic interface:
using region_base = flecsi::data::hpx::region;
using flecsi::data::hpx::rows, flecsi::data::hpx::prefixes;

struct intervals {
  using Value = subrow; // [begin, end)
  static Value make(subrow r, std::size_t = 0) {
    return r;
  }

  intervals(prefixes & pre,
    const partition & p,
    field_id_t fid, // The field id for the metadata in the region in p.
    completeness = incomplete)
    : r(&pre.get_base()) {
    // Called by upper layer, supplied with a region and a partition. There are
    // two regions involved. The region for `r` stores real field data
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
    ghost_ranges = to_vector(p.get_storage<Value>(fid));
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
  friend struct copy_engine;

  template<typename T>
  auto get_storage(field_id_t fid) const {
    return r->get_storage<T>(fid, max_end);
  }

  // We use a reference to region instead of pointer since it is not nullable
  // nor needs to be redirected after initialization.
  flecsi::data::hpx::region_impl * r;

  // Locally cached metadata on ranges of ghost index.
  std::vector<Value> ghost_ranges;
  std::size_t max_end = 0;
};

struct points {
  using Value = std::pair<std::size_t, std::size_t>; // (rank, index)
  static Value make(std::size_t r, std::size_t i) {
    return {r, i};
  }

  points(prefixes & p, const intervals &, field_id_t, completeness = incomplete)
    : r(&p.get_base()) {}

private:
  // The region `r` contains field data of shared entities on this rank as
  // source to be copied to remote peers. We make copy_engine a friend to allow
  // direct access to the region.
  friend struct copy_engine;

  flecsi::data::hpx::region_impl * r;
};

struct copy_engine {
  // One copy engine for each entity type i.e. vertex, cell, edge.
  copy_engine(const points & points,
    const intervals & intervals,
    field_id_t meta_fid /* for remote shared entities */)
    : source(points), destination(intervals) {
    // There is no information about the indices of local shared entities,
    // ranks and indices of the destination of copy i.e. (local source
    // index, {(remote dest rank, remote dest index)}). We need to do a shuffle
    // operation to reconstruct this info from {(local ghost index, remote
    // source rank, remote source index)}.
    auto remote_sources =
      destination.get_storage<const points::Value>(meta_fid);
    // Essentially a GroupByKey of remote_sources, keys are the remote source
    // ranks and values are vectors of remote source indices.
    SendPoints remote_shared_entities;
    for(const auto & [begin, end] : destination.ghost_ranges) {
      for(auto ghost_idx = begin; ghost_idx < end; ++ghost_idx) {
        const auto & shared = remote_sources[ghost_idx];
        remote_shared_entities[shared.first].emplace_back(shared.second);
        // We also group local ghost entities into
        // (src rank, { local ghost ids})
        ghost_entities[shared.first].emplace_back(ghost_idx);
      }
    }

    // Create the inverse mapping of
    // group_shared_entities. This creates a map from remote destination rank to
    // a vector of *local* source indices. This information is later used by
    // hpx::collectives::set().
    {
      auto comm = flecsi::run::context::instance().world_comm(
        "copy_engine" + std::to_string(meta_fid));
      auto f = util::hpx::all_to_allv(
        [&](int r, int) -> auto & {
          static const std::vector<std::size_t> empty;
          const auto i = remote_shared_entities.find(r);
          return i == remote_shared_entities.end() ? empty : i->second;
        },
        comm);

      std::size_t r = 0;
      for(auto & v : f.get()) {
        if(!v.empty())
          shared_entities.try_emplace(r, std::move(v));
        ++r;
      }
    }

    // We need to figure out the max local source index in order to give correct
    // nelems when calling region::get_storage().
    for(const auto & [rank, indices] : shared_entities) {
      max_local_source_idx = (std::max)(max_local_source_idx,
        *std::max_element(indices.begin(), indices.end()));
    }
    max_local_source_idx += 1;
  }

  // called with each field (and field_id_t) on the entity, for example, one
  // for pressure, temperature, density etc.
  void operator()(field_id_t data_fid) const {

    auto source_storage =
      source.r->get_storage<std::byte>(data_fid, max_local_source_idx);
    auto destination_storage = destination.get_storage<std::byte>(data_fid);
    auto type_size = source.r->get_field_info(data_fid)->type_size;

    auto comm = flecsi::run::context::instance().world_channel_comm();

    using buffer_type = ::hpx::serialization::serialize_buffer<std::byte>;
    using data_type = std::vector<buffer_type>;

    std::vector<::hpx::future<void>> requests;
    requests.reserve(ghost_entities.size() + shared_entities.size());

    for(const auto & [src_rank, ghost_indices] : ghost_entities) {
      auto f = ::hpx::collectives::get<data_type>(
        comm, ::hpx::collectives::that_site_arg(src_rank));
      auto fc = f.then(::hpx::launch::sync,
        [&, ghost_indices = std::move(ghost_indices)](auto && f) {
          const auto & data = f.get();
          std::size_t i = 0;
          for(auto ghost_idx : ghost_indices) {
            std::copy_n(data[i].data(),
              type_size,
              destination_storage.data() + ghost_idx * type_size);
            ++i;
          }
        });
      requests.push_back(std::move(fc));
    }

    for(const auto & [dst_rank, shared_indices] : shared_entities) {
      data_type send_buffers;
      send_buffers.reserve(shared_indices.size());
      for(auto shared_idx : shared_indices) {
        send_buffers.emplace_back(
          buffer_type(source_storage.data() + shared_idx * type_size,
            type_size,
            buffer_type::reference));
      }
      auto f = ::hpx::collectives::set(comm,
        ::hpx::collectives::that_site_arg(dst_rank),
        std::move(send_buffers));
      requests.push_back(std::move(f));
    }

    ::hpx::wait_all(requests);

    // rethrow exceptions, if any
    for(auto && f : requests)
      f.get();
  }

private:
  // (remote rank, { local indices })
  using SendPoints = std::map<Color, std::vector<std::size_t>>;

  const points & source;
  const intervals & destination;
  SendPoints ghost_entities; // (src rank,  { local ghost indices})
  SendPoints shared_entities; // (dest rank, { local shared indices})
  std::size_t max_local_source_idx = 0;
};

template<typename T>
T
get_scalar_from_accessor(const T * ptr) {
  return *ptr;
}
} // namespace data
} // namespace flecsi
