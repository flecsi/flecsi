function(_flecsi_check_kokkos_compiler)
  if(Kokkos_ENABLE_CUDA AND CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
     CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 13.0.1)
     message(WARNING "Disabling GPU debuginfo for Clang > 13.0.1 due to "
                     "https://github.com/llvm/llvm-project/issues/58491")
  endif()
endfunction()

macro(flecsi_enable_kokkos target)
  find_package(Kokkos COMPONENTS separable_compilation REQUIRED)
  target_link_libraries(${target} PUBLIC Kokkos::kokkos)

  if(Kokkos_ENABLE_CUDA)
    _flecsi_check_kokkos_compiler()
    target_compile_options(${target} PUBLIC
      $<$<AND:$<CXX_COMPILER_ID:Clang>,$<CONFIG:Debug>,$<VERSION_GREATER:$<CXX_COMPILER_VERSION>,13.0.1>>:-Xarch_device -g0>)
  endif()
endmacro()
