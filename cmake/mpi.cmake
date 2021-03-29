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
option(ENABLE_MPI_THREAD_MULITPLE "Enable MPI_THREAD_MULTIPLE" OFF)
mark_as_advanced(ENABLE_MPI_CXX_BINDINGS)
mark_as_advanced(ENABLE_MPI_THREAD_MULITPLE)

if(ENABLE_MPI)
  if(ENABLE_MPI_CXX_BINDINGS)
    find_package(MPI COMPONENTS C MPICXX REQUIRED)
    list(APPEND TPL_LIBRARIES MPI::MPI_CXX)
    set(MPI_LANGUAGE CXX)
  else()
    find_package(MPI COMPONENTS C CXX REQUIRED)
    # These probably aren't needed anymore
    #list(APPEND TPL_DEFINES -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)
    list(APPEND TPL_LIBRARIES MPI::MPI_C)
    set(MPI_LANGUAGE C)
  endif()

endif()
