// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_LOCAL_STORAGE_HH
#define FLECSI_DATA_LOCAL_STORAGE_HH

#include "flecsi/exec/task_attributes.hh"
#include "flecsi/run/backend.hh"
#include "flecsi/util/mpi.hh"

#include <Kokkos_DualView.hpp>

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

  template<privilege Priv>
  std::variant<host_access<Priv>, device_access<Priv>> current_data() {
    if(current == toc)
      return device_access<Priv>(toc_buffer);
    else {
      if(privilege_write(Priv))
        current = loc;
      return host_access<Priv>(loc_buffer);
    }
  }

  template<exec::processor Proc = exec::processor::loc,
    privilege Priv = privilege::ro>
  std::conditional_t<Proc == exec::processor::toc,
    device_access<Priv>,
    host_access<Priv>>
  data() {
    const auto transfer_return = [this](auto & sync, auto & ret) {
      if(ret.extent(0) < sync.extent(0))
        Kokkos::resize(ret, sync.extent(0));

      auto ret_view = Kokkos::subview(
        ret, std::pair<std::size_t, std::size_t>(0, sync.extent(0)));

      // If wo is requested, we don't care what's there, so no need to copy
      if constexpr(Priv != privilege::wo)
        Kokkos::deep_copy(ret_view, sync);

      if constexpr(!privilege_write(Priv))
        current = both;
      else
        current = (current == loc ? toc : loc);

      return ret;
    };

    switch(current) {
      case loc:
        if constexpr(Proc == exec::processor::toc)
          return transfer_return(loc_buffer, toc_buffer);
        else
          return loc_buffer;
      case toc:
        if constexpr(Proc != exec::processor::toc)
          return transfer_return(toc_buffer, loc_buffer);
        else
          return toc_buffer;
      default:
        if constexpr(Proc == exec::processor::toc) {
          // If we're writing, we need to change the state
          if constexpr(privilege_write(Priv))
            current = toc;

          return toc_buffer;
        }
        else {
          // If we're writing, we need to change the state
          if constexpr(privilege_write(Priv))
            current = loc;

          return loc_buffer;
        }
    }
  }

  template<exec::processor Proc = exec::processor::loc>
  auto data() const {
    return const_cast<storage<T> *>(this)->data<Proc>();
  }

  void resize(std::size_t size) {
    if(current == loc || current == both)
      Kokkos::resize(loc_buffer, size);

    if(current == toc || current == both)
      Kokkos::resize(toc_buffer, size);
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

/// \}
} // namespace local

#ifdef DOXYGEN // implemented per-backend
/// Backend specific data storage.
/// \ingroup local-data
struct backend_storage : local::detail::storage {
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
