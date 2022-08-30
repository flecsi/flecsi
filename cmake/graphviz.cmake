macro(flecsi_enable_graphviz)
  find_package(Graphviz REQUIRED)

  message(STATUS "Found Graphviz: ${Graphviz_INCLUDE_DIRS}")

  list(APPEND TPL_INCLUDES ${Graphviz_INCLUDE_DIRS})
  list(APPEND TPL_LIBRARIES ${Graphviz_LIBRARIES})
endmacro()
