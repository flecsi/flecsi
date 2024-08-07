// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_LOCAL_STORAGE_HH
#define FLECSI_DATA_LOCAL_STORAGE_HH

#include "flecsi/exec/task_attributes.hh"
#include "flecsi/run/backend.hh"
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

namespace local {
/// \defgroup local-data Backend Data
/// Direct data storage.
/// \ingroup data
/// \{
namespace detail {

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

struct buffer_impl_loc {
  inline static constexpr exec::task_processor_type_t location =
    exec::task_processor_type_t::loc;

  std::byte * data() {
    return v.data();
  }

  std::size_t size() const {
    return v.size();
  }

  auto kokkos_view() {
    return host_view{v.data(), v.size()};
  }

  auto kokkos_view() const {
    return host_const_view{v.data(), v.size()};
  }

  void resize(std::size_t size) {
    v.resize(size);
  }

private:
  std::vector<std::byte> v;
};

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

        // If wo is requested, we don't care what's there, so no need to copy
        if constexpr(AccessPrivilege != partition_privilege_t::wo)
          Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
            ret.kokkos_view(),
            sync.kokkos_view());

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

/// \}
} // namespace local

#ifdef DOXYGEN // implemented per-backend
/// Backend specific data storage.
/// \ingroup data
/// \{
struct backend_storage : local::detail::storage {
  /// Synchronize with all pending operations on this storage.
  ///
  /// \note This is implemented in the MPI and HPX backends. For the
  ///       MPI backend this operation is a no-op.
  void synchronize();
};
/// \}
#endif

} // namespace data
} // namespace flecsi

#endif // FLECSI_DATA_LOCAL_STORAGE_HH
