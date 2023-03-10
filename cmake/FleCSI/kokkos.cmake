macro(flecsi_enable_kokkos target)
  find_package(Kokkos COMPONENTS separable_compilation REQUIRED)
  target_link_libraries(${target} PUBLIC Kokkos::kokkos)
endmacro()
