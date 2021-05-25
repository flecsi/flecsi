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
#include <vector>

namespace flecsi {
namespace data {
namespace local {
/// \defgroup local-data Backend Data
/// Direct data storage.
/// \ingroup data
/// \{
namespace detail {

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
    return Kokkos::View<std::byte *,
      Kokkos::HostSpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>(v.data(), v.size());
  }

  auto kokkos_view() const {
    return Kokkos::View<const std::byte *,
      Kokkos::HostSpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>(v.data(), v.size());
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
    return Kokkos::View<std::byte *,
      Kokkos::DefaultExecutionSpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ptr, s);
  }

  auto kokkos_view() const {
    return Kokkos::View<const std::byte *,
      Kokkos::DefaultExecutionSpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ptr, s);
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
    Kokkos::deep_copy(dst.kokkos_view(), src.kokkos_view());
  }

  exec::task_processor_type_t is_on = exec::task_processor_type_t::loc;
  // TODO: How to handle the case when ExecutionSpace is actually HostSpace
  //  (e.g. OpenMP)?
  // Why don't we use Kokkos_View directly? Ans: our region lives longer than
  // Kokkos.
  buffer_impl_loc loc_buffer;
  buffer_impl_toc toc_buffer;
};
#else // !defined(FLECSI_ENABLE_KOKKOS)
using storage = buffer;
#endif // defined(FLECSI_ENABLE_KOKKOS)
} // namespace detail
/// \}
} // namespace local

#ifdef DOXYGEN // implemented per-backend
/// Backend specific data storage.
/// \ingroup data
/// \{
struct backend_storage : local::detail::storage
{
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
