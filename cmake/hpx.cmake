macro(flecsi_enable_hpx target)

  if(NOT FLECSI_CMAKE_ENABLE_BOOST)
    message(ERROR "Boost is required for the HPX backend")
  endif()

  find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)

  target_compile_definitions(${target} PUBLIC ENABLE_HPX)
  target_include_directories(${target} SYSTEM PUBLIC ${HPX_INCLUDE_DIRS})
  target_link_libraries(${target} PUBLIC ${HPX_LIBRARIES})

  if(MSVC)
    target_compile_definitions(${target} PUBLIC
      _SCL_SECURE_NO_WARNINGS
      _CRT_SECURE_NO_WARNINGS
      _SCL_SECURE_NO_DEPRECATE
      _CRT_SECURE_NO_DEPRECATE
      _CRT_NONSTDC_NO_WARNINGS
      _HAS_AUTO_PTR_ETC=1
      _SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING
      _SILENCE_CXX17_ALLOCATOR_VOID_DEPRECATION_WARNING
      GTEST_LANG_CXX11=1
    )
  endif()

endmacro()
