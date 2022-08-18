option(ENABLE_OPENMP "Enable OpenMP Task Support" OFF)

if(ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)

  list(APPEND TPL_LIBRARIES OpenMP::OpenMP_CXX)
endif()
