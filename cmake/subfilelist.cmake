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

function(make_subfilelist result directory)

  file(GLOB _CHILDREN RELATIVE ${directory} ${directory}/*)

  foreach(_CHILD ${_CHILDREN})
    if(NOT IS_DIRECTORY ${directory}/${_CHILD})
      list(APPEND _DIRLIST ${_CHILD})
    endif()
  endforeach()

  set(${result} ${_DIRLIST} PARENT_SCOPE)
endfunction(make_subfilelist)
