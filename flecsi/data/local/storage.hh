// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_LOCAL_STORAGE_HH
#define FLECSI_DATA_LOCAL_STORAGE_HH

#include "flecsi/data/privilege.hh"
#include "flecsi/exec/task_attributes.hh"
#include "flecsi/run/backend.hh"
#include "flecsi/util/mpi.hh"

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <numeric>
#include <unordered_map>
#include <utility>

namespace flecsi {
namespace data {
// The "infinite" size used for resizable regions (backend-specific because it
// depends on Legion::coord_t for the Legion backend)
constexpr inline util::id logical_size = std::numeric_limits<util::id>::max();

namespace local {
/// \defgroup local-data Backend Data
/// Direct data storage.
/// \ns{data::local}.
/// \ingroup data
/// \{
namespace detail {

template<typename T = std::byte>
struct storage {
  static_assert(!std::is_pointer_v<T>,
    "Kokkos::View<U**> would be multi-dimensional");

  enum sync { loc, toc, both };

  using host_view = Kokkos::View<T *, Kokkos::HostSpace>;
  using host_const_view = Kokkos::View<const T *, Kokkos::HostSpace>;
  using device_view = Kokkos::View<T *, Kokkos::DefaultExecutionSpace>;
  using device_const_view =
    Kokkos::View<const T *, Kokkos::DefaultExecutionSpace>;

  template<privilege Priv>
  using host_access =
    std::conditional_t<privilege_write(Priv), host_view, host_const_view>;

  template<privilege Priv>
  using device_access =
    std::conditional_t<privilege_write(Priv), device_view, device_const_view>;

  template<exec::processor P>
  void prefer(bool hard = false) {
    if(hard || current == both)
      data<rw, P>();
  }

  template<privilege Priv = ro, exec::processor Proc = exec::processor::loc>
  std::conditional_t<Proc == exec::processor::toc,
    device_access<Priv>,
    host_access<Priv>>
  data() {
    const auto transfer = [this](const auto & src, auto & ret, sync s) {
      if(current != both && current != s) {
        const auto n = src.extent(0);
        Kokkos::resize(Kokkos::WithoutInitializing, ret, n);

        // If wo is requested, we don't care what's there, so no need to copy
        if constexpr(Priv != privilege::wo) {
          Kokkos::deep_copy(Kokkos::subview(ret, std::pair(0 * n, n)), src);
          current = both;
        }
      }
      if(privilege_write(Priv))
        current = s;

      return ret;
    };

    if constexpr(Proc == exec::processor::toc)
      return transfer(loc_buffer, toc_buffer, toc);
    else
      return transfer(toc_buffer, loc_buffer, loc);
  }

  // NB: logically const, but can still transfer to Proc.
  template<exec::processor Proc = exec::processor::loc>
  auto data() const {
    return const_cast<storage<T> *>(this)->data<ro, Proc>();
  }

  auto data2() { // for coherent updates
    return std::pair(current == toc ? nullptr : loc_buffer.data(),
      current == loc ? nullptr : toc_buffer.data());
  }

  void resize(std::size_t size) {
    if(current == loc || current == both)
      Kokkos::resize(Kokkos::WithoutInitializing, loc_buffer, size);

    if(current == toc || current == both)
      Kokkos::resize(Kokkos::WithoutInitializing, toc_buffer, size);
  }

  std::size_t size() const {
    return (current == loc ? loc_buffer.extent(0) : toc_buffer.extent(0));
  }

private:
  sync current = both;
  host_view loc_buffer;
  device_view toc_buffer;
};

} // namespace detail

struct storage : detail::storage<> {
  template<class T, // sometimes erased to be std::byte
    privilege Priv = ro,
    exec::processor Proc = exec::processor::loc>
  auto as(std::size_t nelems) {
    using return_type = flecsi::util::span<privilege_const<T, Priv>>;

    std::size_t nbytes = nelems * sizeof(T);
    if(nbytes > size()) {
      if(Priv == ro)
        flog_fatal("reading uninitialized field");
      resize(nbytes);
    }
    else
      flog_assert(size() % sizeof(T) == 0,
        "Field access with wrong type. Requesting "
          << util::type<T>() << ", storage size = " << size()
          << ", nelems = " << nelems);

    return return_type(reinterpret_cast<typename return_type::pointer>(
                         data<Priv, Proc>().data()),
      nelems);
  }
};

/// \}
} // namespace local

#ifdef DOXYGEN // implemented per-backend
/// Backend specific data storage.
/// \ingroup local-data
struct backend_storage : local::storage {
  /// Synchronize with all pending operations on this storage.
  ///
  /// \note This is implemented in the MPI and HPX backends. For the
  ///       MPI backend this operation is a no-op.
  void synchronize();
};
#endif

} // namespace data
} // namespace flecsi

#endif // FLECSI_DATA_LOCAL_STORAGE_HH
