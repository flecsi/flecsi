// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_GRAPHVIZ_HH
#define FLECSI_UTIL_GRAPHVIZ_HH

#include <flecsi-config.hh>

#include "flecsi/util/common.hh" // FILE

#if !defined(FLECSI_ENABLE_GRAPHVIZ)
#error FLECSI_ENABLE_GRAPHVIZ not defined! This file depends on Graphviz!
#endif

#include <graphviz/cgraph.h>

#include <stdexcept>
#include <utility>

/// \cond core
namespace flecsi {
namespace util {
/// \defgroup graphviz Graphviz Support
/// Wrapper for \c libcgraph.
/// \ingroup utils
/// \{

inline constexpr int ag_create = 1, ag_access = 0;

/// Class for creating Graphviz trees.
class graphviz
{
  static auto cc(const char * s) {
    return const_cast<char *>(s);
  }

public:
  ~graphviz() {
    if(graph_ != nullptr) {
      agclose(graph_);
    } // if
  } // ~graphviz

  explicit graphviz(const std::string & name) : graphviz(name.c_str()) {}
  explicit graphviz(const char * name)
    : graph_(agopen(cc(name), Agdirected, nullptr)) {
    const auto a = [&](int k, const char * n, const char * v) {
      agattr(graph_, k, cc(n), cc(v));
    };

    a(AGRAPH, "nodesep", ".5");

    // set default node attributes
    a(AGNODE, "label", "");
    a(AGNODE, "penwidth", "");
    a(AGNODE, "color", "black");
    a(AGNODE, "shape", "ellipse");
    a(AGNODE, "style", "");
    a(AGNODE, "fillcolor", "lightgrey");
    a(AGNODE, "fontcolor", "black");

    // set default edge attributes
    a(AGEDGE, "dir", "forward");
    a(AGEDGE, "label", "");
    a(AGEDGE, "penwidth", "");
    a(AGEDGE, "color", "black");
    a(AGEDGE, "style", "");
    a(AGEDGE, "fillcolor", "black");
    a(AGEDGE, "fontcolor", "black");
    a(AGEDGE, "headport", "c");
    a(AGEDGE, "tailport", "c");
    a(AGEDGE, "arrowsize", "0.75");
    a(AGEDGE, "arrowhead", "normal");
    a(AGEDGE, "arrowtail", "normal");
  }
  graphviz(graphviz && g) : graph_(std::exchange(g.graph_, {})) {}

  explicit operator bool() const {
    return graph_;
  }

  /// Add a node to the graph.
  /// \param label if non-null, be used for the display name of the node.
  Agnode_t * add_node(const char * name, const char * label = nullptr) {
    Agnode_t * node = agnode(graph_, cc(name), ag_create);

    if(label != nullptr) {
      agset(node, cc("label"), cc(label));
    } // if

    return node;
  } // add_node

  Agnode_t * node(const char * name) const {
    return agnode(graph_, cc(name), ag_access);
  } // node

  /// Set a node attribute.
  void
  set_node_attribute(Agnode_t * node, const char * attr, const char * value) {
    agset(node, cc(attr), cc(value));
  } // set_node_attribute

  /// Add an edge to the graph.
  Agedge_t * add_edge(Agnode_t * parent, Agnode_t * child) {
    return agedge(graph_, parent, child, nullptr, ag_create);
  } // add_edge

  void
  set_edge_attribute(Agedge_t * edge, const char * attr, const char * value) {
    agset(edge, cc(attr), cc(value));
  } // set_edge_attribute

  void write(const std::string & name) const {
    write(name.c_str());
  } // write

  void write(const char * name) const {
    if(agwrite(graph_, FILE(name, "w")))
      throw std::runtime_error("could not write graph");
  } // write

private:
  Agraph_t * graph_;

}; // class graphviz

/// \}
} // namespace util
} // namespace flecsi
/// \endcond

#endif
