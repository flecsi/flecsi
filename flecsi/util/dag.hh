// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_DAG_HH
#define FLECSI_UTIL_DAG_HH

#include "flecsi/config.hh"
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
    : NodePolicy(std::forward<Args>(args)...),
      label_(util::strip_return_type(util::strip_parameter_list(label))) {
    const void * address = static_cast<const void *>(this);
    std::stringstream ss;
    ss << address;
    identifier_ = ss.str();

    // strip unit test wrapper
    constexpr char wrapper_pfx[] = "flecsi::util::unit::action";
    if(label_.rfind(wrapper_pfx, 0) == 0) {
      auto pos = label_.rfind("(", label_.rfind(","));
      label_ = label_.substr(sizeof(wrapper_pfx), pos - sizeof(wrapper_pfx));
    }
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

    // Find the common leading scopes of the nodes under this dag.
    std::string cmmn;
    if(this->size()) {
      auto get_ns = [](const std::string & text) -> std::string {
        std::regex nsreg("^[\\w:]+");
        std::smatch match;
        if(std::regex_search(text, match, nsreg)) {
          auto pos = match.str().rfind("::");
          if(pos != std::string::npos) {
            return text.substr(0, pos);
          }
        }
        return "";
      };

      auto is_ident = [](const char c) { return std::isalnum(c) || c == '_'; };

      auto n = this->begin();
      cmmn = get_ns((*n++)->label());
      for(; n != this->end(); ++n) {
        auto current = (*n)->label();
        while(cmmn.size() && (!(current.rfind(cmmn, 0) == 0) ||
                               is_ident(current[cmmn.length()]))) {
          cmmn = get_ns(cmmn);
        } // while
      } // for
      cmmn = cmmn.size() ? cmmn + "::" : cmmn;
    } // if

    for(auto n : *this) {
      // Strip the common leading substring.
      std::string label = n->label().substr(cmmn.length());

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
