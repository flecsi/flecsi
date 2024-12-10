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

template<typename T = std::byte> // can not be pointer
struct storage {
  static_assert(!std::is_pointer_v<T>,
    "Kokkos::View<U**> would be multi-dimensional");
  // is this the default template arguments
  using dual_view_type = Kokkos::DualView<T *,
    Kokkos::LayoutLeft,
    Kokkos::Device<Kokkos::DefaultExecutionSpace,
      Kokkos::DefaultExecutionSpace::memory_space>>;
  using host_space = typename dual_view_type::host_mirror_space;
  using device_space = typename dual_view_type::execution_space;
  using host_view = typename dual_view_type::t_host;
  using host_const_view = typename dual_view_type::t_host_const;
  using device_view = typename dual_view_type::t_dev;
  using device_const_view = typename dual_view_type::t_dev_const;

  storage()
    : dual_view((std::stringstream()
                  << "dual_view-" << util::type<T>() << "-" << count++)
                  .str(),
        0) {}

  // In the case where the memory space is assignable from both host
  // and device the view types within the variant are equivalent for all
  // intents and purposes, so we form a variant containing only one of these
  // types. Otherwise, we need to construct a variant containing the two
  // distinct view types (with const-ness considered given the privilege)
  template<partition_privilege_t AccessPrivilege>
  using view_variant =
    std::conditional_t<Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace,
                         Kokkos::HostSpace>::accessible,
      std::variant<std::conditional_t<privilege_write(AccessPrivilege),
        host_view,
        host_const_view>>,
      std::variant<std::conditional_t<privilege_write(AccessPrivilege),
                     host_view,
                     host_const_view>,
        std::conditional_t<privilege_write(AccessPrivilege),
          device_view,
          device_const_view>>>;

  template<partition_privilege_t AccessPrivilege>
  view_variant<AccessPrivilege> current_data() {
    using variant_type = view_variant<AccessPrivilege>;
    // Kokkos::View uses static asserts instead of limiting
    // conversions which forces the std::in_place_index usage (see Kokkos#2127)
    if constexpr(std::variant_size_v<variant_type> == 2)
      if(!dual_view.template need_sync<device_space>())
        return variant_type(std::in_place_index<1>, dual_view.d_view);
    return variant_type(std::in_place_index<0>, dual_view.h_view);
  }

  // make a release note that we fixed bug in handling openmp with mpi backend
  template<exec::task_processor_type_t ProcessorType =
             exec::task_processor_type_t::loc,
    partition_privilege_t AccessPrivilege = partition_privilege_t::ro>
  auto data() {
    using space =
      std::conditional_t<ProcessorType == exec::task_processor_type_t::toc,
        device_space,
        host_space>;
    constexpr auto reading = privilege_read(AccessPrivilege);

    if constexpr(reading)
      dual_view.template sync<space>();

    if constexpr(privilege_write(AccessPrivilege)) {
      if constexpr(!reading)
        dual_view.clear_sync_state();
      dual_view.template modify<space>();
    }

    return Kokkos::View<privilege_const<T, AccessPrivilege> *, space>(
      dual_view.template view<space>());
  }

  template<exec::task_processor_type_t ProcessorType =
             exec::task_processor_type_t::loc,
    partition_privilege_t AccessPrivilege = ro,
    typename = std::enable_if_t<(AccessPrivilege == ro)>>
  auto data() const {
    return const_cast<storage<T> *>(this)
      ->data<ProcessorType, AccessPrivilege>();
  }

  std::size_t size() const {
    return dual_view.extent(0);
  }

  void resize(std::size_t size) {
    Kokkos::resize(dual_view, size);
  }

private:
  dual_view_type dual_view;
  inline static int count = 0;
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
