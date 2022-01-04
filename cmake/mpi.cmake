#------------------------------------------------------------------------------#
#  @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
# /@@/////  /@@          @@////@@ @@////// /@@
# /@@       /@@  @@@@@  @@    // /@@       /@@
# /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
# /@@////   /@@/@@@@@@@/@@       ////////@@/@@
# /@@       /@@/@@//// //@@    @@       /@@/@@
# /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
# //       ///  //////   //////  ////////  //
#
# Copyright (c) 2016, Triad National Security, LLC
# All rights reserved
#------------------------------------------------------------------------------#

option(ENABLE_MPI "Enable MPI" OFF)
option(ENABLE_MPI_CXX_BINDINGS "Enable MPI C++ Bindings" OFF)
mark_as_advanced(ENABLE_MPI_CXX_BINDINGS)

if(ENABLE_MPI)
  find_package(Threads REQUIRED)
  list(APPEND TPL_LIBRARIES Threads::Threads)
  if(ENABLE_MPI_CXX_BINDINGS)
    find_package(MPI COMPONENTS C MPICXX REQUIRED)
    list(APPEND TPL_LIBRARIES MPI::MPI_CXX)
    set(MPI_LANGUAGE CXX)
  else()
    find_package(MPI COMPONENTS C REQUIRED)
    list(APPEND TPL_LIBRARIES MPI::MPI_C)
    set(MPI_LANGUAGE C)
  endif()

endif()
