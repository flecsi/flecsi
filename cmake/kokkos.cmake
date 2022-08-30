macro(flecsi_enable_kokkos)
  find_package(Kokkos REQUIRED)

  get_target_property(KOKKOS_COMPILE_OPTIONS Kokkos::kokkoscore
    INTERFACE_COMPILE_OPTIONS)

  list(APPEND TPL_LIBRARIES Kokkos::kokkos)
endmacro()
