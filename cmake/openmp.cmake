option(ENABLE_OPENMP "Enable OpenMP Task Support" OFF)

if(ENABLE_OPENMP)

  find_package(OpenMP)

  if(NOT OPENMP_FOUND)
      message(WARNING "OpenMP was requested but not found.")
  else()
    set (CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${OpenMP_Fortran_FLAGS}")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()

endif(ENABLE_OPENMP)
