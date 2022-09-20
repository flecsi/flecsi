macro(flecsi_enable_kokkos target)
  find_package(Kokkos REQUIRED)
  target_link_libraries(${target} PUBLIC Kokkos::kokkos)
endmacro()
