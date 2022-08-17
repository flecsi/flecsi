option(ENABLE_GRAPHVIZ "Enable Graphviz" OFF)

if(ENABLE_GRAPHVIZ)
  find_package(Graphviz REQUIRED)

  if(NOT Graphviz_FOUND)
      message(FATAL_ERROR "Graphviz is required for this build configuration")
  endif()

  message(STATUS "Found Graphviz: ${Graphviz_INCLUDE_DIRS}")

  list(APPEND TPL_INCLUDES ${Graphviz_INCLUDE_DIRS})
  list(APPEND TPL_LIBRARIES ${Graphviz_LIBRARIES})
endif(ENABLE_GRAPHVIZ)
