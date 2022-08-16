include(CMakeDependentOption)
include(colors)

option(ENABLE_DOCUMENTATION "Enable documentation" OFF)
mark_as_advanced(ENABLE_DOCUMENTATION)

if(ENABLE_DOCUMENTATION)
  add_custom_target(doc
    ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/.doc-dummy
  )

  if(ENABLE_SPHINX AND ENABLE_DOXYGEN)
    find_package(Git REQUIRED)

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
            ${GIT_EXECUTABLE} clone -q --single-branch --branch gh-pages
              git@github.com:flecsi/flecsi.git gh-pages &&
            cd gh-pages &&
            ${GIT_EXECUTABLE} rm -qr . && ${GIT_EXECUTABLE} reset -q &&
            ${GIT_EXECUTABLE} checkout .gitignore) &&
        echo "Updating pages" &&
          cp -rT doc gh-pages &&
          (cd gh-pages && ${GIT_EXECUTABLE} add -A .) &&
        echo "Updated gh-pages are in ${CMAKE_BINARY_DIR}/gh-pages" &&
        echo "${FLECSI_Red}!!!WARNING WARNING WARNING!!!" &&
        echo "The gh-pages repository points to an EXTERNAL remote on github.com." &&
        echo "!!!MAKE SURE THAT YOU UNDERSTAND WHAT YOU ARE DOING BEFORE YOU PUSH!!!${FLECSI_ColorReset}"
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  endif()
endif()
