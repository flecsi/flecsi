macro(flecsi_enable_openmp target)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
endmacro()
