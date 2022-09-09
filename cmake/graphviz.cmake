macro(flecsi_enable_graphviz target)
  find_package(Graphviz REQUIRED)

  message(STATUS "Found Graphviz: ${Graphviz_INCLUDE_DIRS}")

  target_include_directories(${target} SYSTEM PUBLIC ${Graphviz_INCLUDE_DIRS})
  target_link_libraries(${target} PUBLIC ${Graphviz_LIBRARIES})
endmacro()
