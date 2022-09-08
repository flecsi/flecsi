macro(flecsi_enable_kokkos)
  find_package(Kokkos REQUIRED)
  list(APPEND TPL_LIBRARIES Kokkos::kokkos)
endmacro()
