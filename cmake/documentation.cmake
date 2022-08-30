macro(flecsi_enable_documentation)
  add_custom_target(doc
    ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/.doc-dummy
  )
  set(FLECSI_CMAKE_ENABLE_DOCUMENTATION ON)
endmacro()

macro(flecsi_enable_docs_deployment GITHUB_PAGES_REPO)
  find_package(Git REQUIRED)
  include(colors)

  #--------------------------------------------------------------------------#
  # This target will work with multiple doxygen targets. However, because
  # sphinx is used for the main html page content, it will only work with
  # one sphinx target, i.e., the one named `sphinx`.
  #--------------------------------------------------------------------------#

  add_custom_target(deploy-docs
    COMMAND
      make doc &&
      echo "Updating gh-pages" &&
        ([ -e gh-pages ] ||
          ${GIT_EXECUTABLE} clone -q --single-branch --branch gh-pages
            ${GITHUB_PAGES_REPO} gh-pages &&
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
  set(FLECSI_CMAKE_ENABLE_DOCS_DEPLOYMENT ON)
endmacro()
