include_guard(GLOBAL)

#------------------------------------------------------------------------------#
# Set a custom doc group target name
#------------------------------------------------------------------------------#

macro(flecsi_set_doc_target_name name)
  set(FLECSI_DOC_TARGET ${name})
endmacro()

#------------------------------------------------------------------------------#
# create doc group target ${FLECSI_DOC_TARGET}
# this collects all sphinx and doxygen targets as dependencies
# if no name is set, use default name "doc"
#------------------------------------------------------------------------------#

macro(_flecsi_define_doc_group_target)
  if(NOT DEFINED FLECSI_DOC_TARGET)
    set(FLECSI_DOC_TARGET doc PARENT_SCOPE)
    set(FLECSI_DOC_TARGET doc)
  endif()

  if(NOT TARGET ${FLECSI_DOC_TARGET})
    add_custom_target(${FLECSI_DOC_TARGET}
      ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/.doc-dummy
    )
  endif()
endmacro()

#------------------------------------------------------------------------------#
# Add deploy-docs target to given GITHUB_PAGES_REPO
#------------------------------------------------------------------------------#

function(flecsi_add_doc_deployment target)
  set(options)
  set(one_value_args GITHUB_PAGES_REPO)
  set(multi_value_args)

  cmake_parse_arguments(deploy "${options}" "${one_value_args}"
    "${multi_value_args}" ${ARGN})

  if(NOT DEFINED deploy_GITHUB_PAGES_REPO)
    message(FATAL_ERROR "Need to specify GITHUB_PAGES_REPO")
  endif()

  find_package(Git REQUIRED)
  include(FleCSI/colors)

  _flecsi_define_doc_group_target()

  #--------------------------------------------------------------------------#
  # This target will work with multiple doxygen targets. However, because
  # sphinx is used for the main html page content, it will only work with
  # one sphinx target, i.e., the one named `sphinx`.
  #--------------------------------------------------------------------------#

  add_custom_target(${target}
    COMMAND
      make ${FLECSI_DOC_TARGET} &&
      echo "Updating gh-pages" &&
        ([ -e gh-pages ] ||
          ${GIT_EXECUTABLE} clone -q --single-branch --branch gh-pages
            ${deploy_GITHUB_PAGES_REPO} gh-pages &&
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
endfunction()
