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
// The "infinite" size used for resizable regions (backend-specific because it
// depends on Legion::coord_t for the Legion backend)
constexpr inline util::id logical_size = std::numeric_limits<util::id>::max();

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
using const_view_variant = std::variant<host_const_view, device_const_view>;
#endif

struct buffer {
  inline static constexpr exec::task_processor_type_t location =
    exec::task_processor_type_t::loc;

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
  inline static constexpr exec::task_processor_type_t location =
    exec::task_processor_type_t::toc;

  buffer_impl_toc & operator=(buffer_impl_toc &&) = delete;

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
  /// Describes where the data is currently up-to-date.
  enum class data_sync { loc, toc, both };

  template<exec::task_processor_type_t ProcessorType =
             exec::task_processor_type_t::loc,
    partition_privilege_t AccessPrivilege = partition_privilege_t::ro>
  privilege_const<std::byte, AccessPrivilege> * data() {

    const auto transfer_return = [this](auto & sync, auto & ret) {
      if(ProcessorType == sync.location)
        return sync.data();
      else {
        if(ret.size() < sync.size())
          ret.resize(sync.size());

        auto ret_view = Kokkos::subview(ret.kokkos_view(),
          std::pair<std::size_t, std::size_t>(0, sync.size()));

        // If wo is requested, we don't care what's there, so no need to copy
        if constexpr(AccessPrivilege != partition_privilege_t::wo)
          Kokkos::deep_copy(
            Kokkos::DefaultExecutionSpace{}, ret_view, sync.kokkos_view());

        if constexpr(AccessPrivilege == partition_privilege_t::ro)
          current_state = data_sync::both;
        else
          current_state =
            (current_state == data_sync::loc ? data_sync::toc : data_sync::loc);

        return ret.data();
      }
    };

    // HACK to treat mpi processor type as loc
    if constexpr(ProcessorType == exec::task_processor_type_t::mpi)
      return data<exec::task_processor_type_t::loc, AccessPrivilege>();

    switch(current_state) {
      case data_sync::both:
        if constexpr(ProcessorType == exec::task_processor_type_t::loc) {
          // If we're writing, we need to change the state
          if constexpr(AccessPrivilege != partition_privilege_t::ro)
            current_state = data_sync::loc;

          return loc_buffer.data();
        }
        else {
          // If we're writing, we need to change the state
          if constexpr(AccessPrivilege != partition_privilege_t::ro)
            current_state = data_sync::toc;

          return toc_buffer.data();
        }

      case data_sync::loc:
        return transfer_return(loc_buffer, toc_buffer);
      case data_sync::toc:
        return transfer_return(toc_buffer, loc_buffer);
    }

    return nullptr;
  }

  // Get the view into where the data is currently synced
  template<partition_privilege_t AccessPrivilege = partition_privilege_t::ro>
  std::conditional_t<privilege_write(AccessPrivilege),
    view_variant,
    const_view_variant>
  kokkos_view() {
    const auto qualify_buffer = [](auto & x) -> auto & {
      if constexpr(privilege_write(AccessPrivilege))
        return x;
      else
        return std::as_const(x);
    };

    // Kokkos::View only static asserts when you attempt to convert a view of
    // one address space to another, so the view_variant/const_view_variant can
    // not automatically resolve which constructor to use, thus the need for the
    // branching
    if(current_state == data_sync::loc) {
      this->data<exec::task_processor_type_t::loc, AccessPrivilege>();
      return qualify_buffer(loc_buffer).kokkos_view();
    }
    else {
      this->data<exec::task_processor_type_t::toc, AccessPrivilege>();
      return qualify_buffer(toc_buffer).kokkos_view();
    }
  }

  std::size_t size() const {
    if(current_state == data_sync::loc)
      return loc_buffer.size();

    return toc_buffer.size();
  }

  void resize(std::size_t size) {
    if(current_state == data_sync::loc || current_state == data_sync::both)
      loc_buffer.resize(size);

    if(current_state == data_sync::toc || current_state == data_sync::both)
      toc_buffer.resize(size);
  }

private:
  // Where the data is currently synced
  data_sync current_state = data_sync::both;

  // We don't need to worry about the case that ExecutionSpace is actually
  // HostSpace (e.g. OpenMP) since currently default_accelerator == toc only
  // when compiling for CUDA or HIP.
  // Why don't we use Kokkos_View directly? Ans: our region lives longer than
  // Kokkos.
  buffer_impl_loc loc_buffer;
  buffer_impl_toc toc_buffer;
};

#else // !defined(FLECSI_ENABLE_KOKKOS)

struct storage : buffer {
  template<exec::task_processor_type_t ProcessorType,
    partition_privilege_t AccessPrivilege>
  privilege_const<std::byte, AccessPrivilege> * data() {
    return buffer::data();
  }
};

#endif // defined(FLECSI_ENABLE_KOKKOS)

template<typename T>
struct typed_storage {
  void resize(std::size_t elements) {
    untyped.resize(elements * sizeof(T));
  }

  template<exec::task_processor_type_t ProcessorType =
             exec::task_processor_type_t::loc,
    partition_privilege_t AccessType = partition_privilege_t::ro>
  auto data() {
    return reinterpret_cast<privilege_const<T, AccessType> *>(
      untyped.data<ProcessorType, AccessType>());
  }

  // While this method is const qualified here, the underlying
  // type detail::storage does have state that might change depending upon
  // which task_processor_type_t it is called with, so the data is
  // guaranteed not to mutate, but the state not so much
  template<exec::task_processor_type_t ProcessorType =
             exec::task_processor_type_t::loc>
  const auto * data() const {
    return const_cast<typed_storage *>(this)->data<ProcessorType>();
  }

  std::size_t size() const {
    return untyped.size() / sizeof(T);
  }

private:
  storage untyped;
};

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

#if defined(FLECSI_ENABLE_KOKKOS)
  template<partition_privilege_t AccessPrivilege = partition_privilege_t::ro>
  auto kokkos_view(field_id_t fid) {
    return storages.at(fid).kokkos_view<AccessPrivilege>();
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

  void partition_notify() {}
  void partition_notify(field_id_t) {}

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

protected:
  region_impl * r;

  partition(region & r) : r(&*r) {}
  // number of elements in this partition on this particular rank.
  size_t nelems = 0;
};

} // namespace mpi

// This type must be defined outside of namespace mpi to support
// forward declarations
struct partition : mpi::partition { // instead of "using partition ="
  using mpi::partition::partition;

  template<topo::single_space>
  partition & get_partition() {
    return *this;
  }
};

struct copy_engine;

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

  friend copy_engine;
};
/// \}
} // namespace mpi

// For backend-agnostic interface:
using region_base = mpi::region;
using mpi::rows, mpi::prefixes;

struct borrow : borrow_base {
  borrow(Claims c) {
    auto & ctx = run::context::instance();
    if(c.size() != ctx.processes())
      flog_fatal("MPI backend limited: one selection per process needed");
    auto p = ctx.process();
    const Claim i = c[p];
    sel = i != nil;
    if(sel && i != p)
      flog_fatal("MPI backend limited: no cross-color access");
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
    ghost_ranges = to_vector(p.get_storage<Value,
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

  template<typename T, partition_privilege_t AccessPrivilege>
  auto get_storage(field_id_t fid) const {
    return r->get_storage<T, exec::task_processor_type_t::loc, AccessPrivilege>(
      fid, max_end);
  }

  mpi::region_impl * r;

  // Locally cached metadata on ranges of ghost index.
  std::vector<Value> ghost_ranges;
  std::size_t max_end = 0;
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
  using index_type = std::size_t;

  using Point = std::pair<index_type, index_type>; // (rank, index)
  static Point point(std::size_t r, std::size_t i) {
    return {r, i};
  }

  // One copy engine for each entity type i.e. vertex, cell, edge.
  copy_engine(const prefixes & src,
    const intervals & intervals,
    field_id_t meta_fid /* for remote shared entities */)
    : source(src), destination(intervals) {
    // There is no information about the indices of local shared entities,
    // ranks and indices of the destination of copy i.e. (local source
    // index, {(remote dest rank, remote dest index)}). We need to do a shuffle
    // operation to reconstruct this info from {(local ghost index, remote
    // source rank, remote source index)}.
    auto remote_sources =
      destination.get_storage<Point, partition_privilege_t::ro>(meta_fid);

    // Calculate the memory needed up front for the ghost_entities
    std::map<Color, std::size_t> mem_size;
    for(const auto & [begin, end] : destination.ghost_ranges) {
      for(auto ghost_idx = begin; ghost_idx < end; ++ghost_idx) {
        const auto & shared = remote_sources[ghost_idx];
        mem_size[shared.first]++;
      }
    }

    // allocate the memory needed
    for(auto & p : mem_size)
      ghost_entities[p.first].resize(std::exchange(p.second, 0));

    // Essentially a GroupByKey of remote_sources, keys are the remote source
    // ranks and values are vectors of remote source indices.
    std::map<Color, std::vector<index_type>> remote_shared_entities;
    for(const auto & [begin, end] : destination.ghost_ranges) {
      for(auto ghost_idx = begin; ghost_idx < end; ++ghost_idx) {
        const auto & shared = remote_sources[ghost_idx];
        remote_shared_entities[shared.first].emplace_back(shared.second);
        // We also group local ghost entities into
        // (src rank, { local ghost ids})

        ghost_entities[shared.first]
          .data<exec::task_processor_type_t::loc,
            flecsi::rw>()[mem_size[shared.first]++] = ghost_idx;
      }
    }

    // Create the inverse mapping of group_shared_entities. This creates a map
    // from remote destination rank to a vector of *local* source indices. This
    // information is later used by MPI_Send().
    {
      std::size_t r = 0;
      for(auto & v : util::mpi::all_to_allv([&](int r) -> auto & {
            static const std::vector<std::size_t> empty;
            const auto i = remote_shared_entities.find(r);
            return i == remote_shared_entities.end() ? empty : i->second;
          })) {
        if(!v.empty()) {
          shared_entities[r].resize(v.size());
          std::uninitialized_copy(v.begin(),
            v.end(),
            shared_entities[r]
              .data<exec::task_processor_type_t::loc, flecsi::rw>());
        }
        ++r;
      }
    }

    // We need to figure out the max local source index in order to give correct
    // nelems when calling region::get_storage().
    for(const auto & [rank, indices] : shared_entities) {
      max_local_source_idx = std::max(max_local_source_idx,
        *std::max_element(indices.data(), indices.data() + indices.size()));
      max_shared_indices_size =
        std::max(max_shared_indices_size, indices.size());
    }
    max_local_source_idx += 1;
  }

  // called with each field (and field_id_t) on the entity, for example, one
  // for pressure, temperature, density etc.
  void operator()(field_id_t data_fid) const {
    using util::mpi::test;

    auto type_size = source.r->get_field_info(data_fid)->type_size;

    auto gather_copy =
      [type_size](std::byte * dst,
        const std::byte * src,
        const mpi::detail::typed_storage<index_type> & src_indices) {
        for(std::size_t i = 0; i < src_indices.size(); i++) {
          std::memcpy(dst + i * type_size,
            src + src_indices.data()[i] * type_size,
            type_size);
        }
      };

    auto scatter_copy =
      [type_size](std::byte * dst,
        const std::byte * src,
        const mpi::detail::typed_storage<index_type> & dst_indices) {
        for(std::size_t i = 0; i < dst_indices.size(); i++) {
          std::memcpy(dst + dst_indices.data()[i] * type_size,
            src + i * type_size,
            type_size);
        }
      };

    std::vector<std::vector<std::byte>> recv_buffers;
    std::size_t max_scatter_buffer_size = 0;

    {
      std::vector<std::vector<std::byte>> send_buffers;
      util::mpi::auto_requests requests(
        ghost_entities.size() + shared_entities.size());

      for(const auto & [src_rank, ghost_indices] : ghost_entities) {
        recv_buffers.emplace_back(ghost_indices.size() * type_size);
        max_scatter_buffer_size =
          std::max(max_scatter_buffer_size, recv_buffers.back().size());
        test(MPI_Irecv(recv_buffers.back().data(),
          int(recv_buffers.back().size()),
          MPI_BYTE,
          int(src_rank),
          0,
          MPI_COMM_WORLD,
          requests()));
      }

#if defined(FLECSI_ENABLE_KOKKOS)
      // allocate gather buffer on device
      auto gather_buffer_device_view =
        Kokkos::View<std::byte *, Kokkos::DefaultExecutionSpace>{
          Kokkos::ViewAllocateWithoutInitializing("gather"),
          max_shared_indices_size * type_size};
#endif

      // shared_indices is created on the host, but is accessed from
      // the device. It will be copied to the device on the first iteration.
      // Shared data in the field storage is copied to the gather buffer
      // in parallel. It is then copied to the send buffer (on host) and
      // sent to the peer via MPI_Send.
      for(const auto & [dst_rank, shared_indices] : shared_entities) {
        auto n_elements = shared_indices.size();
        auto n_bytes = n_elements * type_size;
        send_buffers.emplace_back(n_bytes);

#if defined(FLECSI_ENABLE_KOKKOS)
        const auto & src_indices = shared_indices;
        std::visit(
          overloaded{[&](const mpi::detail::host_const_view & src) {
                       gather_copy(
                         send_buffers.back().data(), src.data(), src_indices);
                     },
            [&](const mpi::detail::device_const_view & src) {
              const auto * shared_indices_device_data =
                src_indices.data<exec::task_processor_type_t::toc>();

              Kokkos::parallel_for(
                n_elements, KOKKOS_LAMBDA(const auto & i) {
                  // Yes, memcpy is supported on device as long as there is no
                  // std:: qualifier.
                  memcpy(gather_buffer_device_view.data() + i * type_size,
                    src.data() + shared_indices_device_data[i] * type_size,
                    type_size);
                });

              auto gather_view = Kokkos::subview(gather_buffer_device_view,
                std::pair<std::size_t, std::size_t>(0, n_bytes));
              Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
                mpi::detail::host_view{send_buffers.back().data(), n_bytes},
                gather_view);
            }},
          source.r->kokkos_view<partition_privilege_t::ro>(data_fid));
#else
        gather_copy(send_buffers.back().data(),
          source.r
            ->get_storage<std::byte,
              exec::task_processor_type_t::loc,
              partition_privilege_t::ro>(data_fid, max_local_source_idx)
            .data(),
          shared_indices);
#endif

        test(MPI_Isend(send_buffers.back().data(),
          int(send_buffers.back().size()),
          MPI_BYTE,
          int(dst_rank),
          0,
          MPI_COMM_WORLD,
          requests()));
      }
    }

#if defined(FLECSI_ENABLE_KOKKOS)
    // Construct the view based off the maximum recv_buffer size
    auto scatter_buffer_device_view =
      Kokkos::View<std::byte *, Kokkos::DefaultExecutionSpace>{
        Kokkos::ViewAllocateWithoutInitializing("scatter"),
        max_scatter_buffer_size};
#endif

    // ghost_indices is created on the host, but is accessed from
    // the device. It will be copied to the device on the first iteration.
    // Ghost data is received from peers bia MPI_Recv into the
    // recv_buffers. It is then copied to the scatter_buffer (on device)
    // and eventually copied in parallel into the field's storage (on device).

    auto recv_buffer = recv_buffers.begin();
    for(const auto & [src_rank, ghost_indices] : ghost_entities) {
#if defined(FLECSI_ENABLE_KOKKOS)
      auto n_elements = ghost_indices.size();
      const auto & dst_indices = ghost_indices;
      std::visit(
        overloaded{[&](const mpi::detail::host_view & dst) {
                     scatter_copy(dst.data(), recv_buffer->data(), dst_indices);
                   },
          [&](const mpi::detail::device_view & dst) {
            auto scatter_view = Kokkos::subview(scatter_buffer_device_view,
              std::pair<std::size_t, std::size_t>(0, recv_buffer->size()));
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
              scatter_view,
              mpi::detail::host_view{recv_buffer->data(), recv_buffer->size()});

            const auto * ghost_indices_device_data =
              dst_indices.data<exec::task_processor_type_t::toc>();

            Kokkos::parallel_for(
              n_elements, KOKKOS_LAMBDA(const auto & i) {
                memcpy(dst.data() + ghost_indices_device_data[i] * type_size,
                  scatter_buffer_device_view.data() + i * type_size,
                  type_size);
              });
          }},
        destination.r->kokkos_view<partition_privilege_t::wo>(data_fid));
#else
      scatter_copy(
        destination.get_storage<std::byte, partition_privilege_t::wo>(data_fid)
          .data(),
        recv_buffer->data(),
        ghost_indices);
#endif
      recv_buffer++;
    }
  }

private:
  // (remote rank, { local indices })
  using SendPoints = std::map<Color, mpi::detail::typed_storage<index_type>>;

  const prefixes & source;
  const intervals & destination;
  SendPoints ghost_entities; // (src rank,  { local ghost indices})
  SendPoints shared_entities; // (dest rank, { local shared indices})
  std::size_t max_local_source_idx = 0, max_shared_indices_size = 0;
};

} // namespace data
} // namespace flecsi

#endif
