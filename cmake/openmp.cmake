option(ENABLE_OPENMP "Enable OpenMP Task Support" OFF)

if(ENABLE_OPENMP)
  find_package(OpenMP)

  if(OpenMP_CXX_FOUND)
    list(APPEND TPL_LIBRARIES OpenMP::OpenMP_CXX)
  else()
    message(WARNING "OpenMP was requested but not found.")
  endif()
endif()
