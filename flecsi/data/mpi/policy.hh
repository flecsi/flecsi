// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_MPI_POLICY_HH
#define FLECSI_DATA_MPI_POLICY_HH

#include "flecsi/data/field_info.hh"
#include "flecsi/exec/task_attributes.hh"
#include "flecsi/run/backend.hh"
#include "flecsi/util/array_ref.hh"
#include "flecsi/util/mpi.hh"

#include <cstddef>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace flecsi {
namespace data {
namespace mpi {
/// \defgroup mpi-data MPI Data
/// Direct data storage.
/// \ingroup data
/// \{

namespace detail {

#if defined(FLECSI_ENABLE_KOKKOS)
using host_view = Kokkos::
  View<std::byte *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

using host_const_view = Kokkos::View<const std::byte *,
  Kokkos::HostSpace,
  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

using device_view = Kokkos::View<std::byte *,
  Kokkos::DefaultExecutionSpace,
  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

using device_const_view = Kokkos::View<const std::byte *,
  Kokkos::DefaultExecutionSpace,
  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

using view_variant = std::variant<host_view, device_view>;
#endif

struct buffer {
  template<exec::task_processor_type_t ProcessorType =
             exec::task_processor_type_t::loc>
  std::byte * data() {
    return v.data();
  }

  std::size_t size() const {
    return v.size();
  }

#if defined(FLECSI_ENABLE_KOKKOS)
  auto kokkos_view() {
    return host_view{v.data(), v.size()};
  }

  auto kokkos_view() const {
    return host_const_view{v.data(), v.size()};
  }
#endif

  void resize(std::size_t size) {
    v.resize(size);
  }

private:
  std::vector<std::byte> v;
};

#if defined(FLECSI_ENABLE_KOKKOS)
using buffer_impl_loc = buffer;

struct buffer_impl_toc {
  buffer_impl_toc() = default;
  buffer_impl_toc(const buffer_impl_toc &) = delete;
  buffer_impl_toc & operator=(const buffer_impl_toc &) = delete;

  std::byte * data() {
    return ptr;
  }

  std::size_t size() const {
    return s;
  }

  void resize(std::size_t ns) {
    // Kokkos does require calling of kokkos_malloc when ptr == nullptr.
    ptr = static_cast<std::byte *>(
      ptr ? Kokkos::kokkos_realloc<Kokkos::DefaultExecutionSpace>(ptr, ns)
          : Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace>(ns));
    flog_assert(ptr != nullptr, "memory allocation failed");
    s = ns;
  }

  auto kokkos_view() {
    return device_view{ptr, s};
  }

  auto kokkos_view() const {
    return device_const_view{ptr, s};
  }

  ~buffer_impl_toc() {
    if(ptr != nullptr && Kokkos::is_initialized())
      Kokkos::kokkos_free<Kokkos::DefaultExecutionSpace>(ptr);
  }

private:
  std::size_t s = 0;
  std::byte * ptr = nullptr;
};

struct storage {
  template<exec::task_processor_type_t ProcessorType>
  std::byte * data() {
    // HACK to treat mpi processor type as loc
    if constexpr(ProcessorType == exec::task_processor_type_t::mpi)
      return data<exec::task_processor_type_t::loc>();

    if(is_on != ProcessorType) {
      if(is_on == exec::task_processor_type_t::loc) {
        transfer(toc_buffer, loc_buffer);
      }
      else {
        transfer(loc_buffer, toc_buffer);
      }
    }
    is_on = ProcessorType;

    if constexpr(ProcessorType == exec::task_processor_type_t::loc)
      return loc_buffer.data();
    else
      return toc_buffer.data();
  }

  view_variant kokkos_view() {
    if(is_on == exec::task_processor_type_t::loc) {
      return {loc_buffer.kokkos_view()};
    }
    else {
      return {toc_buffer.kokkos_view()};
    }
  }

  std::size_t size() const {
    if(is_on == exec::task_processor_type_t::loc)
      return loc_buffer.size();
    else
      return toc_buffer.size();
  }

  void resize(std::size_t size) {
    if(is_on == exec::task_processor_type_t::loc) {
      loc_buffer.resize(size);
    }
    else {
      toc_buffer.resize(size);
    }
  }

private:
  template<typename D, typename S>
  static void transfer(D & dst, const S & src) {
    // We need to resize buffer on the destination side such that we don't
    // attempt deep_copy to cause buffer overrun (Kokkos does check that).
    if(dst.size() < src.size())
      dst.resize(src.size());
    Kokkos::deep_copy(
      Kokkos::DefaultExecutionSpace{}, dst.kokkos_view(), src.kokkos_view());
  }

  exec::task_processor_type_t is_on = exec::task_processor_type_t::loc;
  // We don't need to worry about the case that ExecutionSpace is actually
  // HostSpace (e.g. OpenMP) since currently default_accelerator == toc only
  // when compiling for CUDA or HIP.
  // Why don't we use Kokkos_View directly? Ans: our region lives longer than
  // Kokkos.
  buffer_impl_loc loc_buffer;
  buffer_impl_toc toc_buffer;
};
#else // !defined(FLECSI_ENABLE_KOKKOS)
using storage = buffer;
#endif // defined(FLECSI_ENABLE_KOKKOS)
} // namespace detail

struct region_impl {
  // The constructor is collectively called on all ranks with the same s,
  // and fs. s.first is number of rows while s.second is number of columns.
  // MPI may assume "number of rows" == number of ranks. The number of columns
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
    for(const auto & f : fs) {
      storages[f->fid];
    }
  }

  size2 size() const {
    return s;
  }

  // The span is safe because it is used only within a user task while the
  // vectors are resized or destroyed only outside user tasks (though perhaps
  // during execute).
  template<class T,
    exec::task_processor_type_t ProcessorType =
      exec::task_processor_type_t::loc>
  util::span<T> get_storage(field_id_t fid) {
    return get_storage<T, ProcessorType>(fid, s.second);
  }

  template<class T,
    exec::task_processor_type_t ProcessorType =
      exec::task_processor_type_t::loc>
  util::span<T> get_storage(field_id_t fid, std::size_t nelems) {
    auto & v = storages.at(fid);
    std::size_t nbytes = nelems * sizeof(T);
    if(nbytes > v.size())
      v.resize(nbytes);
    return {reinterpret_cast<T *>(v.data<ProcessorType>()), nelems};
  }

#if defined(FLECSI_ENABLE_KOKKOS)
  auto kokkos_view(field_id_t fid) {
    return storages.at(fid).kokkos_view();
  }
#endif

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

  std::unordered_map<field_id_t, detail::storage> storages;
};

struct region {
  region(size2 s, const fields & fs, const char * = nullptr)
    : p(new region_impl(s, fs)) {}

  size2 size() const {
    return p->size();
  }

  region_impl & operator*() {
    return *p;
  }

  region_impl * operator->() {
    return p.get();
  }

private:
  std::unique_ptr<region_impl> p; // to preserve an address on move
};

struct partition {
  partition(partition &&) = default;
  partition & operator=(partition &&) & = default;

  Color colors() const {
    // number of rows, essentially the number of MPI ranks.
    return r->size().first;
  }

  template<typename T,
    exec::task_processor_type_t ProcessorType =
      exec::task_processor_type_t::loc>
  auto get_storage(field_id_t fid) const {
    return r->get_storage<T, ProcessorType>(fid, nelems);
  }

  auto get_raw_storage(field_id_t fid, std::size_t item_size) const {
    return r->get_storage<std::byte>(fid, nelems * item_size);
  }

  template<topo::single_space>
  partition & get_partition() {
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

} // namespace mpi

// This type must be defined outside of namespace mpi to support
// forward declarations
struct partition : mpi::partition { // instead of "using partition ="
  using mpi::partition::partition;
};

namespace mpi {

struct rows : data::partition {
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
/// \}
} // namespace mpi

// For backend-agnostic interface:
using region_base = mpi::region;
using mpi::rows, mpi::prefixes;

struct borrow : partition {
  using Value = std::pair<std::size_t, prefixes::row>; // [row,size]
  static Value make(prefixes::row r,
    std::size_t c = run::context::instance().color()) {
    return {c, r};
  }
  static std::size_t get_row(const Value & v) {
    return v.first;
  }
  static prefixes_base::row get_size(const Value & v) {
    return v.second;
  }

  borrow(region_base & r,
    const partition & p,
    field_id_t f,
    completeness = incomplete)
    : partition(r) {
    const auto s = p.get_storage<const Value>(f);
    switch(s.size()) {
      case 0:
        break;
      case 1: {
        auto & v = s[0];
        flog_assert(!v.second || v.first == run::context::instance().color(),
          "sorry: MPI backend does not implement cross-color access");
        nelems = v.second;
        break;
      }
      default:
        flog_fatal("underlying partition has size " << s.size() << " > 1");
    }
  }
};

struct copy_engine;

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
  friend copy_engine;

  template<typename T>
  auto get_storage(field_id_t fid) const {
    return r->get_storage<T>(fid, max_end);
  }

#if defined(FLECSI_ENABLE_KOKKOS)
  auto kokkos_view(field_id_t fid) const {
    return r->kokkos_view(fid);
  }
#endif

  mpi::region_impl * r;

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
  friend copy_engine;

  mpi::region_impl * r;
};

// Copy/Paste from cppreference.com to make std::visit looks more
// like pattern matching in ML or Haskell.
template<class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
// explicit deduction guide (not needed as of C++20)
template<class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

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

    // Create the inverse mapping of group_shared_entities. This creates a map
    // from remote destination rank to a vector of *local* source indices. This
    // information is later used by MPI_Send().
    {
      std::size_t r = 0;
      for(auto &v : util::mpi::all_to_allv([&](int r, int) -> auto & {
            static const std::vector<std::size_t> empty;
            const auto i = remote_shared_entities.find(r);
            return i == remote_shared_entities.end() ? empty : i->second;
          })) {
        if(!v.empty())
          shared_entities.try_emplace(r, std::move(v));
        ++r;
      }
    }

    // We need to figure out the max local source index in order to give correct
    // nelems when calling region::get_storage().
    for(const auto & [rank, indices] : shared_entities) {
      max_local_source_idx = std::max(max_local_source_idx,
        *std::max_element(indices.begin(), indices.end()));
    }
    max_local_source_idx += 1;
  }

  // called with each field (and field_id_t) on the entity, for example, one
  // for pressure, temperature, density etc.
  void operator()(field_id_t data_fid) const {
    using util::mpi::test;

#if defined(FLECSI_ENABLE_KOKKOS)
    auto source_view = source.r->kokkos_view(data_fid);
    auto destination_view = destination.kokkos_view(data_fid);
#else
    auto source_storage =
      source.r->get_storage<std::byte>(data_fid, max_local_source_idx);
    auto destination_storage = destination.get_storage<std::byte>(data_fid);
#endif

    auto type_size = source.r->get_field_info(data_fid)->type_size;

    std::vector<MPI_Request> requests;
    requests.reserve(ghost_entities.size() + shared_entities.size());

    std::vector<std::vector<std::byte>> recv_buffers;
    for(const auto & [src_rank, ghost_indices] : ghost_entities) {
      recv_buffers.emplace_back(ghost_indices.size() * type_size);
      requests.resize(requests.size() + 1);
      test(MPI_Irecv(recv_buffers.back().data(),
        int(recv_buffers.back().size()),
        MPI_BYTE,
        int(src_rank),
        0,
        MPI_COMM_WORLD,
        &requests.back()));
    }

    auto gather_copy = [type_size](std::byte * dst,
                         const std::byte * src,
                         const std::vector<std::size_t> & src_indices) {
      for(std::size_t i = 0; i < src_indices.size(); i++) {
        std::memcpy(
          dst + i * type_size, src + src_indices[i] * type_size, type_size);
      }
    };

    auto scatter_copy = [type_size](std::byte * dst,
                          const std::byte * src,
                          const std::vector<std::size_t> & dst_indices) {
      for(std::size_t i = 0; i < dst_indices.size(); i++) {
        std::memcpy(
          dst + dst_indices[i] * type_size, src + i * type_size, type_size);
      }
    };

#if defined(FLECSI_ENABLE_KOKKOS)
    // copy a std::vector<T> from host to device memory and return the device
    // version as a Kokkos::View
    auto device_copy = [](const auto & hvec, const std::string & label) {
      using T = typename std::decay_t<decltype(hvec)>::value_type;
      Kokkos::View<T *, Kokkos::DefaultExecutionSpace> dview{
        Kokkos::ViewAllocateWithoutInitializing(label), hvec.size()};
      Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
        dview,
        Kokkos::View<const T *, Kokkos::HostSpace>{hvec.data(), hvec.size()});
      return dview;
    };
#endif

    std::vector<std::vector<std::byte>> send_buffers;
    for(const auto & [dst_rank, shared_indices] : shared_entities) {
      requests.resize(requests.size() + 1);
      auto n_elements = shared_indices.size();
      auto n_bytes = n_elements * type_size;
      send_buffers.emplace_back(n_bytes);

#if defined(FLECSI_ENABLE_KOKKOS)
      // We can not capture shared_indices in the overloaded lambda inside
      // std::visit directly
      const auto & src_indices = shared_indices;
      std::visit(
        overloaded{[&](mpi::detail::host_view & src) {
                     gather_copy(
                       send_buffers.back().data(), src.data(), src_indices);
                   },
          [&](mpi::detail::device_view & src) {
            // copy shared indices from host to device
            auto shared_indices_device_view =
              device_copy(src_indices, "shared indices");

            // allocate gather buffer on device
            auto gather_buffer_device_view =
              Kokkos::View<std::byte *, Kokkos::DefaultExecutionSpace>{
                "gather", n_bytes};

            // copy shared values to gather buffer on device in parallel, for
            // each element
            Kokkos::parallel_for(
              n_elements, KOKKOS_LAMBDA(const auto & i) {
                // Yes, memcpy is supported on device as long as there is no
                // std:: qualifier.
                memcpy(gather_buffer_device_view.data() + i * type_size,
                  src.data() + shared_indices_device_view[i] * type_size,
                  type_size);
              });

            // copy gathered shared values to send_buffer on host to be sent
            // through MPI.
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
              mpi::detail::host_view{send_buffers.back().data(), n_bytes},
              gather_buffer_device_view);
          }},
        source_view);
#else
      gather_copy(
        send_buffers.back().data(), source_storage.data(), shared_indices);
#endif

      test(MPI_Isend(send_buffers.back().data(),
        int(send_buffers.back().size()),
        MPI_BYTE,
        int(dst_rank),
        0,
        MPI_COMM_WORLD,
        &requests.back()));
    }

    std::vector<MPI_Status> status(requests.size());
    test(MPI_Waitall(int(requests.size()), requests.data(), status.data()));

    // copy from intermediate receive buffer to destination storage
    auto recv_buffer = recv_buffers.begin();
    for(const auto & [src_rank, ghost_indices] : ghost_entities) {
#if defined(FLECSI_ENABLE_KOKKOS)
      auto n_elements = ghost_indices.size();
      // We can not capture ghost_indices in the overloaded lambda inside
      // std::visit directly
      const auto & dst_indices = ghost_indices;
      std::visit(
        overloaded{[&](mpi::detail::host_view & dst) {
                     scatter_copy(dst.data(), recv_buffer->data(), dst_indices);
                   },
          [&](mpi::detail::device_view & dst) {
            // copy ghost indices from host to device
            auto ghost_indices_device_view =
              device_copy(dst_indices, "ghost indices");

            // copy recv buffer from host to scatter buffer on device
            auto scatter_buffer_device_view =
              device_copy(*recv_buffer, "scatter");

            // copy ghost values from scatter buffer on device to field storage
            // in parallel, for each element
            Kokkos::parallel_for(
              n_elements, KOKKOS_LAMBDA(const auto & i) {
                memcpy(dst.data() + ghost_indices_device_view[i] * type_size,
                  scatter_buffer_device_view.data() + i * type_size,
                  type_size);
              });
          }},
        destination_view);
#else
      scatter_copy(
        destination_storage.data(), recv_buffer->data(), ghost_indices);
#endif
      recv_buffer++;
    }
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

} // namespace data
} // namespace flecsi

#endif
