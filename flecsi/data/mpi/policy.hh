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
#include <stdexcept>
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
/// \ns{data::mpi}.
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
  inline static constexpr exec::processor location = exec::processor::loc;

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
  inline static constexpr exec::processor location = exec::processor::toc;

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

  template<privilege Priv = ro, exec::processor Proc = exec::processor::loc>
  privilege_const<std::byte, Priv> * data() {

    const auto transfer_return = [this](auto & sync, auto & ret) {
      if(Proc == sync.location)
        return sync.data();
      else {
        if(ret.size() < sync.size())
          ret.resize(sync.size());

        auto ret_view = Kokkos::subview(ret.kokkos_view(),
          std::pair<std::size_t, std::size_t>(0, sync.size()));

        // If wo is requested, we don't care what's there, so no need to copy
        if constexpr(Priv != wo)
          Kokkos::deep_copy(
            Kokkos::DefaultExecutionSpace{}, ret_view, sync.kokkos_view());

        if constexpr(Priv == ro)
          current_state = data_sync::both;
        else
          current_state =
            (current_state == data_sync::loc ? data_sync::toc : data_sync::loc);

        return ret.data();
      }
    };

    // HACK to treat mpi processor type as loc
    if constexpr(Proc == exec::processor::mpi)
      return data<Priv>();

    switch(current_state) {
      case data_sync::both:
        if constexpr(Proc == exec::processor::loc) {
          // If we're writing, we need to change the state
          if constexpr(Priv != ro)
            current_state = data_sync::loc;

          return loc_buffer.data();
        }
        else {
          // If we're writing, we need to change the state
          if constexpr(Priv != ro)
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
  template<privilege Priv = ro>
  std::conditional_t<privilege_write(Priv), view_variant, const_view_variant>
  kokkos_view() {
    const auto qualify_buffer = [](auto & x) -> auto & {
      if constexpr(privilege_write(Priv))
        return x;
      else
        return std::as_const(x);
    };

    // Kokkos::View only static asserts when you attempt to convert a view of
    // one address space to another, so the view_variant/const_view_variant can
    // not automatically resolve which constructor to use, thus the need for the
    // branching
    if(current_state == data_sync::loc) {
      this->data<Priv, exec::processor::loc>();
      return qualify_buffer(loc_buffer).kokkos_view();
    }
    else {
      this->data<Priv, exec::processor::toc>();
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
  buffer_impl_loc loc_buffer;
  buffer_impl_toc toc_buffer;
};

#else // !defined(FLECSI_ENABLE_KOKKOS)

struct storage : buffer {
  template<privilege Priv = ro, exec::processor Proc = exec::processor::loc>
  privilege_const<std::byte, Priv> * data() {
    return buffer::data();
  }
};

#endif // defined(FLECSI_ENABLE_KOKKOS)

template<typename T>
struct typed_storage {
  void resize(std::size_t elements) {
    untyped.resize(elements * sizeof(T));
  }

  template<privilege Priv = ro, exec::processor Proc = exec::processor::loc>
  auto data() {
    return reinterpret_cast<privilege_const<T, Priv> *>(
      untyped.data<Priv, Proc>());
  }

  // While this method is const qualified here, the underlying
  // type detail::storage does have state that might change depending upon
  // which exec::processor it is called with, so the data is
  // guaranteed not to mutate, but the state not so much
  template<exec::processor Proc = exec::processor::loc>
  const auto * data() const {
    return const_cast<typed_storage *>(this)->data<ro, Proc>();
  }

  std::size_t size() const {
    return untyped.size() / sizeof(T);
  }

private:
  storage untyped;
};

} // namespace detail

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

    return return_type{
      reinterpret_cast<typename return_type::pointer>(v.data<Priv, Proc>()),
      nelems};
  }

#if defined(FLECSI_ENABLE_KOKKOS)
  template<privilege Priv = ro>
  auto kokkos_view(field_id_t fid) {
    return storages.at(fid).kokkos_view<Priv>();
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
  size2 s;
  fields fs;

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
    return r->size().first;
  }

  template<typename T,
    privilege Priv = ro,
    exec::processor Proc = exec::processor::loc>
  auto get_storage(field_id_t fid) const {
    return r->get_storage<T, Priv, Proc>(fid, nelems);
  }

  template<privilege Priv>
  auto get_raw_storage(field_id_t fid, std::size_t item_size) const {
    return r->get_storage<std::byte, Priv, exec::processor::loc>(
      fid, nelems * item_size);
  }

protected:
  region_impl * r;

  partition(region & r) : r(&*r) {}

  void resize(std::size_t n) {
    if(n > r->size().second)
      throw std::out_of_range("partition larger than region");
    nelems = n;
  }

private:
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
    const auto s =
      f.get_partition().template get_storage<size_request>(f.fid());
    flog_assert(
      s.size() == 1, "underlying partition must have size 1, not " << s.size());
    resize(s[0]);
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
    // Eagerly read field data, which might legitimately change later.
    ghost_ranges = to_vector(p.get_storage<Value>(fid));
    if(auto iter = std::max_element(ghost_ranges.begin(),
         ghost_ranges.end(),
         [](Value x, Value y) { return x.second < y.second; });
       iter != ghost_ranges.end()) {
      max_end = iter->second;
    }
  }

private:
  friend copy_engine;

  template<typename T, privilege Priv>
  auto get_storage(field_id_t fid) const {
    return r->get_storage<T, Priv>(fid, max_end);
  }

  mpi::region_impl * r;

  // Locally cached metadata on ranges of ghost index.
  std::vector<Value> ghost_ranges;
  std::size_t max_end = 0; // size of prefix containing all ranges
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

  copy_engine(const prefixes & src, const intervals & intervals, field_id_t fid)
    : source(&src), destination(&intervals) {
    // The input comprises the color and index of shared elements stored at
    // each ghost element; reverse those pointers to know what to send where.

    auto remote_sources = intervals.get_storage<Point, ro>(fid);

    // Calculate the memory needed up front for the ghost_entities
    std::map<Color, std::size_t> mem_size;
    for(const auto & [begin, end] : intervals.ghost_ranges) {
      for(auto ghost_idx = begin; ghost_idx < end; ++ghost_idx) {
        const auto & shared = remote_sources[ghost_idx];
        mem_size[shared.first]++;
      }
    }

    for(auto & p : mem_size)
      ghost_entities[p.first].resize(std::exchange(p.second, 0));

    // Essentially a GroupByKey of remote_sources, keys are the remote source
    // ranks and values are vectors of remote source indices.
    std::map<Color, std::vector<index_type>> remote_shared_entities;
    for(const auto & [begin, end] : intervals.ghost_ranges) {
      for(auto ghost_idx = begin; ghost_idx < end; ++ghost_idx) {
        const auto & shared = remote_sources[ghost_idx];
        remote_shared_entities[shared.first].emplace_back(shared.second);
        // We also group local ghost entities into
        // (src rank, { local ghost ids})

        ghost_entities[shared.first].data<rw>()[mem_size[shared.first]++] =
          ghost_idx;
      }
    }

    // Create the inverse mapping of remote_shared_entities. This creates a map
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
          std::uninitialized_copy(
            v.begin(), v.end(), shared_entities[r].data<rw>());
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

  void operator()(const std::vector<field_id_t> & ff) const {
    using util::mpi::test;

    std::vector<std::vector<std::byte>> recv_buffers;
    std::size_t max_scatter_buffer_size = 0;

    {
      std::vector<std::vector<std::byte>> send_buffers;
      util::mpi::auto_requests requests(
        (ghost_entities.size() + shared_entities.size()) * ff.size());

      for(auto data_fid : ff) {
        auto type_size = source->r->get_field_info(data_fid)->type_size;

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
        std::optional<Kokkos::View<std::byte *, Kokkos::DefaultExecutionSpace>>
          gather_buffer_device_view;
#endif

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
                  src_indices.data<exec::processor::toc>();

                if(!gather_buffer_device_view)
                  gather_buffer_device_view.emplace(
                    Kokkos::ViewAllocateWithoutInitializing("gather"),
                    max_shared_indices_size * type_size);
                Kokkos::parallel_for(
                  n_elements, KOKKOS_LAMBDA(const auto & i) {
                    // Yes, memcpy is supported on device as long as there is no
                    // std:: qualifier.
                    memcpy(gather_buffer_device_view->data() + i * type_size,
                      src.data() + shared_indices_device_data[i] * type_size,
                      type_size);
                  });

                auto gather_view = Kokkos::subview(*gather_buffer_device_view,
                  std::pair<std::size_t, std::size_t>(0, n_bytes));
                Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
                  mpi::detail::host_view{send_buffers.back().data(), n_bytes},
                  gather_view);
              }},
            source->r->kokkos_view<ro>(data_fid));
#else
          gather_copy(send_buffers.back().data(),
            source->r->get_storage<std::byte>(data_fid, max_local_source_idx)
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
    }

#if defined(FLECSI_ENABLE_KOKKOS)
    std::optional<Kokkos::View<std::byte *, Kokkos::DefaultExecutionSpace>>
      scatter_buffer_device_view;
#endif

    // Copy recv_buffers to scatter_buffer_device_view and then in parallel
    // into the field's storage (on device).
    auto recv_buffer = recv_buffers.begin();
    for(auto data_fid : ff) {
      auto type_size = source->r->get_field_info(data_fid)->type_size;

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

      for(const auto & [src_rank, ghost_indices] : ghost_entities) {
#if defined(FLECSI_ENABLE_KOKKOS)
        auto n_elements = ghost_indices.size();
        const auto & dst_indices = ghost_indices;
        std::visit(
          overloaded{[&](const mpi::detail::host_view & dst) {
                       scatter_copy(
                         dst.data(), recv_buffer->data(), dst_indices);
                     },
            [&](const mpi::detail::device_view & dst) {
              if(!scatter_buffer_device_view)
                scatter_buffer_device_view.emplace(
                  Kokkos::ViewAllocateWithoutInitializing("scatter"),
                  max_scatter_buffer_size);
              auto scatter_view = Kokkos::subview(*scatter_buffer_device_view,
                std::pair<std::size_t, std::size_t>(0, recv_buffer->size()));
              Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
                scatter_view,
                mpi::detail::host_view{
                  recv_buffer->data(), recv_buffer->size()});

              const auto * ghost_indices_device_data =
                dst_indices.data<exec::processor::toc>();

              Kokkos::parallel_for(
                n_elements, KOKKOS_LAMBDA(const auto & i) {
                  memcpy(dst.data() + ghost_indices_device_data[i] * type_size,
                    scatter_buffer_device_view->data() + i * type_size,
                    type_size);
                });
            }},
          destination->r->kokkos_view<rw>(data_fid));
#else
        scatter_copy(destination->get_storage<std::byte, rw>(data_fid).data(),
          recv_buffer->data(),
          ghost_indices);
#endif
        recv_buffer++;
      }
    }
  }

private:
  // (remote rank, { local indices })
  using SendPoints = std::map<Color, mpi::detail::typed_storage<index_type>>;

  const prefixes * source;
  const intervals * destination;
  SendPoints ghost_entities; // (src rank,  { local ghost indices})
  SendPoints shared_entities; // (dest rank, { local shared indices})
  std::size_t max_local_source_idx = 0, max_shared_indices_size = 0;
};

} // namespace data
} // namespace flecsi

#endif
