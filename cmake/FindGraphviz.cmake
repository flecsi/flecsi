find_program(Graphviz_dot_EXECUTABLE
  NAMES dot
  PATH_SUFFIXES bin
  DOC "Graphviz dot executable"
)

find_path(Graphviz_INCLUDE_DIR NAMES graphviz/cgraph.h)

find_library(Graphviz_cdt_LIBRARY NAMES cdt )
find_library(Graphviz_cgraph_LIBRARY NAMES cgraph )
find_library(Graphviz_gvc_LIBRARY NAMES gvc )

mark_as_advanced(
  Graphviz_INCLUDE_DIR
  Graphviz_cdt_LIBRARY
  Graphviz_cgraph_LIBRARY
  Graphviz_gvc_LIBRARY
  Graphviz_dot_EXECUTABLE
)

if(Graphviz_INCLUDE_DIR AND EXISTS "${Graphviz_INCLUDE_DIR}/graphviz/graphviz_version.h")
  file(STRINGS "${Graphviz_INCLUDE_DIR}/graphviz/graphviz_version.h" graphviz_version REGEX PACKAGE_VERSION)
  if(graphviz_version)
    string(REGEX
      REPLACE "#define.*PACKAGE_VERSION.*\"([.0-9]+).*\""
      "\\1"
      Graphviz_VERSION
      "${graphviz_version}")
  else()
    set(Graphviz_VERSION 0.0)
  endif()
else()
  set(Graphviz_VERSION 0.0)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Graphviz
  REQUIRED_VARS
    Graphviz_INCLUDE_DIR
	  Graphviz_cdt_LIBRARY
    Graphviz_cgraph_LIBRARY
    Graphviz_gvc_LIBRARY
    Graphviz_dot_EXECUTABLE
  VERSION_VAR
    Graphviz_VERSION
)

if(Graphviz_FOUND)
	set(Graphviz_LIBRARIES ${Graphviz_gvc_LIBRARY} ${Graphviz_cgraph_LIBRARY} ${Graphviz_cdt_LIBRARY})
	set(Graphviz_INCLUDE_DIRS ${Graphviz_INCLUDE_DIR})

  if(NOT TARGET Graphviz::cdt)
    add_library(Graphviz::cdt UNKNOWN IMPORTED)
    set_target_properties(Graphviz::cdt PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${Graphviz_INCLUDE_DIR}")
    set_property(TARGET Graphviz::cdt APPEND PROPERTY
        IMPORTED_LOCATION "${Graphviz_cdt_LIBRARY}")
  endif()

  if(NOT TARGET Graphviz::cgraph)
    add_library(Graphviz::cgraph UNKNOWN IMPORTED)
    set_target_properties(Graphviz::cgraph PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${Graphviz_INCLUDE_DIR}")
    set_property(TARGET Graphviz::cgraph APPEND PROPERTY
        IMPORTED_LOCATION "${Graphviz_cgraph_LIBRARY}")
    target_link_libraries(Graphviz::cgraph INTERFACE Graphviz::cdt)
  endif()

  if(NOT TARGET Graphviz::Graphviz)
    add_library(Graphviz::Graphviz UNKNOWN IMPORTED)
    set_target_properties(Graphviz::Graphviz PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${Graphviz_INCLUDE_DIR}")
    set_property(TARGET Graphviz::Graphviz APPEND PROPERTY
        IMPORTED_LOCATION "${Graphviz_gvc_LIBRARY}")
    target_link_libraries(Graphviz::Graphviz INTERFACE Graphviz::cgraph)
  endif()

  if(NOT TARGET Graphviz::dot)
    add_executable(Graphviz::dot IMPORTED GLOBAL)
    set_target_properties(Graphviz::dot
      PROPERTIES IMPORTED_LOCATION "${Graphviz_dot_EXECUTABLE}")
  endif()
else()
	set(Graphviz_LIBRARIES)
	set(Graphviz_INCLUDE_DIRS)
endif()

