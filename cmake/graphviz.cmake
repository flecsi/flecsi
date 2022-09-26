macro(flecsi_enable_graphviz target)
  find_package(Graphviz REQUIRED)
  target_link_libraries(${target} PUBLIC Graphviz::Graphviz)
endmacro()
