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

include(CMakeDependentOption)
include(colors)

option(ENABLE_DOCUMENTATION "Enable documentation" OFF)
mark_as_advanced(ENABLE_DOCUMENTATION)

if(ENABLE_DOCUMENTATION)
  add_custom_target(doc
    ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/.doc-dummy
  )

  if(ENABLE_SPHINX AND ENABLE_DOXYGEN)

    find_package(Git)

    if(NOT GIT_FOUND)
      message(FATAL_ERROR "Git is required for this target")
    endif()

    #--------------------------------------------------------------------------#
    # This target will work with multiple doxygen targets. However, because
    # sphinx is used for the main html page content, it will only work with
    # one sphinx target, i.e., the one named `sphinx`.
    #--------------------------------------------------------------------------#

    add_custom_target(deploy-documentation
      COMMAND
        make doc &&
        echo "Updating gh-pages" &&
          ([ -e gh-pages ] ||
            ${GIT_EXECUTABLE} clone --single-branch --branch gh-pages
              git@gitlab.lanl.gov:flecsi/flecsi-pages.git gh-pages &&
            cd gh-pages &&
            git rm -r . && git reset &&
            ${GIT_EXECUTABLE} remote rm origin &&
            ${GIT_EXECUTABLE} remote add origin
              git@github.com:flecsi/flecsi.git &&
              ${GIT_EXECUTABLE} fetch) &&
        echo "Updating Sphinx pages" &&
          cp -rT doc/sphinx gh-pages &&
        echo "Updating Doxygen pages" &&
          cp -rT doc/doxygen gh-pages/doxygen &&
        echo "Updated gh-pages are in ${CMAKE_BINARY_DIR}/gh-pages" &&
        echo "${FLECSI_Red}!!!WARNING WARNING WARNING!!!" &&
        echo "The gh-pages repository points to an EXTERNAL remote on github.com." &&
        echo "!!!MAKE SURE THAT YOU UNDERSTAND WHAT YOU ARE DOING BEFORE YOU PUSH!!!${FLECSI_ColorReset}"
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  endif()
endif()
