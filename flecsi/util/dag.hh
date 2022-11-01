// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_DAG_HH
#define FLECSI_UTIL_DAG_HH

#include <flecsi-config.h>

#include "flecsi/flog.hh"

#if defined(FLECSI_ENABLE_GRAPHVIZ)
#include "flecsi/util/graphviz.hh"
#endif

#include <algorithm>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <regex>
#include <sstream>
#include <vector>

namespace flecsi {
namespace util {
/// \addtogroup utils
/// \{

namespace dag_impl {

/*
 */

template<typename NodePolicy>
struct node : NodePolicy, std::list<node<NodePolicy> const *> {

  template<typename... Args>
  node(std::string const & label, Args &&... args)
    : NodePolicy(std::forward<Args>(args)...), label_(label) {
    const void * address = static_cast<const void *>(this);
    std::stringstream ss;
    ss << address;
    identifier_ = ss.str();

    // Strip arguments.
    label_ = std::regex_replace(label_, std::regex("\\([^\\)]*\\)"), "()");

    // Strip unit test wrapper.
    label_ = std::count(label_.begin(), label_.end(), '<')
               ? std::regex_replace(
                   std::regex_replace(label, std::regex("^[^<]*<"), ""),
                   std::regex(std::regex(",.*")),
                   "")
               : label_;
  }

  std::string const & identifier() const {
    return identifier_;
  }

  std::string const & label() const {
    return label_;
  }

private:
  std::string identifier_;
  std::string label_;
};

} // namespace dag_impl

/*!
  Basic DAG type.
 */

template<typename NodePolicy>
struct dag : std::vector<dag_impl::node<NodePolicy> *> {

  using node_type = dag_impl::node<NodePolicy>;
  using sorted_type = std::vector<const node_type *>;

  dag(const char * label = "empty") : label_(label) {}

  /*!
    @return the DAG label.
   */

  std::string const & label() const {
    return label_;
  }

  /*!
    Topological sort using Kahn's algorithm.

    @return A valid sequence of the nodes in the DAG.
   */

  sorted_type sort() const {
    sorted_type sorted;

    // Create a temporary list of the nodes.
    std::list<node_type *> nodes;
    for(auto n : *this) {
      nodes.push_back(n);
    } // for

    // Create a tally of the number of dependencies of each node
    // in the graph. Remove nodes from the temporary list that do not
    // have any depenendencies.
    std::queue<node_type *> q;
    for(auto n = nodes.begin(); n != nodes.end();) {
      if((*n)->size() == 0) {
        q.push(*n);
        n = nodes.erase(n);
      }
      else {
        ++n;
      } // if
    } // for

    size_t count{0};
    while(!q.empty()) {
      const auto root = q.front();
      sorted.push_back(root);
      q.pop();

      for(auto n = nodes.begin(); n != nodes.end();) {
        auto it = std::find_if((*n)->begin(),
          (*n)->end(),
          [&root](const auto p) { return root == p; });

        if(it != (*n)->end()) {
          (*n)->erase(it);

          if(!(*n)->size()) {
            q.push(*n);
            n = nodes.erase(n);
          } // if
        }
        else {
          ++n;
        } // if
      } // for

      ++count;
    } // while

    flog_assert(count == this->size(), "sorting failed. This is not a DAG!!!");

    return sorted;
  } // sort

#if defined(FLECSI_ENABLE_GRAPHVIZ)
  /*!
    Add the DAG to the given graphviz graph.
   */

  void add(graphviz & gv, const char * color = "#c5def5") const {
    std::map<uintptr_t, Agnode_t *> node_map;

    // Find the common leading substring of the nodes under this dag.
    std::string cmmn;
    if(this->size()) {
      // Return substring up to last occurance of "::" (exclusive) or
      // empty string.
      auto strip_ns = [](std::string const & s) {
        auto pos = s.find_last_of(':');
        return std::count(s.begin(), s.end(), ':') % 2 == 0 &&
                   pos != std::string::npos
                 ? s.substr(0, pos - 1) /* assumes namespace separator "::" */
                 : "";
      }; // strip_ns

      auto n = this->begin();
      cmmn = strip_ns((*n++)->label());
      for(; n != this->end(); ++n) {
        auto current = (*n)->label();
        while(cmmn.size() && current.find(cmmn) == std::string::npos) {
          cmmn = strip_ns(cmmn);
        } // while
      } // for
      cmmn = cmmn.size() ? cmmn + "::" : cmmn;
    } // if

    for(auto n : *this) {
      // Strip the common leading substring.
      std::string label = std::regex_replace(n->label(), std::regex(cmmn), "");

      auto * node = gv.add_node(n->identifier().c_str(), label.c_str());
      node_map[uintptr_t(n)] = node;

      gv.set_node_attribute(node, "color", "black");
      gv.set_node_attribute(node, "style", "filled");
      gv.set_node_attribute(node, "fillcolor", color);
    } // for

    for(auto n : *this) {
      for(auto e : *n) {
        auto * edge =
          gv.add_edge(node_map[uintptr_t(e)], node_map[uintptr_t(n)]);
        gv.set_edge_attribute(edge, "penwidth", "1.5");
      } // for
    } // for
  } // add

#endif // FLECSI_ENABLE_GRAPHVIZ

private:
  std::string label_;
}; // struct dag

/// \}
} // namespace util
} // namespace flecsi

#endif
