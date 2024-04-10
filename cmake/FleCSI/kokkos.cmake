macro(flecsi_enable_kokkos target)
  find_package(Kokkos COMPONENTS separable_compilation REQUIRED)
  target_link_libraries(${target} PUBLIC Kokkos::kokkos)

  if(Kokkos_ENABLE_CUDA AND CMAKE_BUILD_TYPE STREQUAL "Debug" AND
     CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
     CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 13.0.1)
    message(WARNING "Disabling GPU debuginfo for Clang > 13.0.1 due to "
                    "https://github.com/llvm/llvm-project/issues/58491")
    target_compile_options(${target} PUBLIC -Xarch_device -g0)
  endif()
endmacro()
