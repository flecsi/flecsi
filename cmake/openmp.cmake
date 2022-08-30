macro(flecsi_enable_openmp)
  find_package(OpenMP REQUIRED COMPONENTS CXX)

  list(APPEND TPL_LIBRARIES OpenMP::OpenMP_CXX)
endmacro()
