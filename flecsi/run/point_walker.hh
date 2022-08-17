// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUN_POINT_WALKER_HH
#define FLECSI_RUN_POINT_WALKER_HH

#include <flecsi-config.h>

#include "flecsi/flog.hh"
#include "flecsi/util/constant.hh"
#include "flecsi/util/dag.hh"

#if defined(FLECSI_ENABLE_GRAPHVIZ)
#include "flecsi/util/graphviz.hh"
#endif

#include <vector>

/// \cond core
namespace flecsi {
namespace run_impl {
/// \addtogroup control
/// \{

/*!
  Allow users to define cyclic control points. Cycles can be nested.

  @tparam Predicate     A predicate function that determines when
                        the cycle should end.
  @tparam ControlPoints A variadic list of control points within the cycle.
 */

template<bool (*Predicate)(), typename... ControlPoints>
struct cycle {

  using type = util::types<ControlPoints...>;

  static bool predicate() {
    return Predicate();
  } // run

}; // struct cycle

template<class F, class... TT>
void
walk(F && f, util::types<TT...> *) {
  (f.template visit_type<TT>(), ...);
}
template<class T, class F>
void
walk(F && f) {
  walk(std::forward<F>(f), static_cast<T *>(nullptr));
}

template<class T>
struct to_types {
  using type = T;
};
template<class... TT>
struct to_types<std::tuple<TT...>> {
  using type = util::types<TT...>;
};
template<class P>
using to_types_t = typename to_types<typename P::control_points>::type;

/*
  Helper type to initialize dag labels.
 */

template<typename ControlPolicy>
struct init_walker {

  using control_points_enum = typename ControlPolicy::control_points_enum;
  using dag_map = typename ControlPolicy::dag_map;

  init_walker(dag_map & registry) : registry_(registry) {}

  template<typename ElementType>
  void visit_type() const {

    if constexpr(std::is_same<typename ElementType::type,
                   control_points_enum>::value) {
      registry_.try_emplace(ElementType::value, *ElementType::value);
    }
    else {
      walk<typename ElementType::type>(*this);
    } // while
  } // visit_type

private:
  dag_map & registry_;

}; // struct init_walker

/*!
  The point_walker class allows execution of statically-defined
  control points.
 */

template<typename ControlPolicy>
struct point_walker {

  using control_points_enum = typename ControlPolicy::control_points_enum;
  using sorted_type = typename ControlPolicy::sorted_type;

  point_walker(const sorted_type & sorted, int & exit_status)
    : sorted_(sorted), exit_status_(exit_status) {}

  /*!
    Handle the tuple type \em ElementType.

    @tparam ElementType The tuple element type. This can either be a
                        control_points_enum or a \em cycle. Cycles are defined
                        by the specialization and must conform to the interface
                        used in the appropriate visit_type method.
   */

  template<typename ElementType>
  void visit_type() const {

    if constexpr(std::is_same<typename ElementType::type,
                   control_points_enum>::value) {

      // This is not a cycle -> execute each action for this control point.
      for(auto & node : sorted_.at(ElementType::value)) {
        exit_status_ |= node->execute();
      } // for
    }
    else {
      // This is a cycle -> create a new control point walker to recurse
      // the cycle.
      while(ElementType::predicate()) {
        point_walker walker(sorted_, exit_status_);
        walk<typename ElementType::type>(walker);
      } // while
    } // if
  } // visit_type

private:
  const sorted_type & sorted_;
  int & exit_status_;

}; // struct point_walker

#if defined(FLECSI_ENABLE_GRAPHVIZ)

template<typename ControlPolicy>
struct point_writer {
  using control_points_enum = typename ControlPolicy::control_points_enum;
  using dag_map = typename ControlPolicy::dag_map;
  using graphviz = flecsi::util::graphviz;

  static constexpr const char * colors[4] = {"#77c3ec",
    "#b8e2f2",
    "#4eb2e0",
    "#9dd9f3"};

  point_writer(const dag_map & registry,
    graphviz & gv,
    Agnode_t *& b,
    Agnode_t *& l,
    int depth = 0)
    : registry_(registry), gv_(gv), begin(b), last(l), depth_(depth) {}

  template<typename ElementType>
  void visit_type() const {
    if constexpr(std::is_same<typename ElementType::type,
                   control_points_enum>::value) {
      auto & dag = registry_.at(ElementType::value);

      auto * root = gv_.add_node(dag.label().c_str(), dag.label().c_str());
      set_begin(root);
      gv_.set_node_attribute(root, "shape", "box");
      gv_.set_node_attribute(root, "style", "rounded");

      if(last) {
        auto * edge = gv_.add_edge(last, root);
        gv_.set_edge_attribute(edge, "color", "#1d76db");
        gv_.set_edge_attribute(edge, "fillcolor", "#1d76db");
        gv_.set_edge_attribute(edge, "style", "bold");
      } // if
      last = root;

      dag.add(gv_, colors[static_cast<size_t>(ElementType::value) % 4]);

      for(auto & n : dag) {
        if(n->size() == 0) {
          auto * edge = gv_.add_edge(root, gv_.node(n->identifier().c_str()));
          gv_.set_edge_attribute(edge, "penwidth", "1.5");
        } // if
      } // for
    }
    else {
      Agnode_t * b = nullptr;
      walk<typename ElementType::type>(
        point_writer(registry_, gv_, b, last, depth_ - 1));
      if(!b)
        return;
      set_begin(b);

      auto * edge = gv_.add_edge(last, b);

      gv_.set_edge_attribute(edge, "label", " cycle");
      gv_.set_edge_attribute(edge, "color", "#1d76db");
      gv_.set_edge_attribute(edge, "fillcolor", "#1d76db");
      gv_.set_edge_attribute(edge, "style", "dashed,bold");

      if(b == last) {
        gv_.set_edge_attribute(edge, "dir", "back");
      }
      else {
        if(depth_ < 0) {
          gv_.set_edge_attribute(edge, "tailport", "ne");
          gv_.set_edge_attribute(edge, "headport", "se");
        }
        else {
          gv_.set_edge_attribute(edge, "tailport", "e");
          gv_.set_edge_attribute(edge, "headport", "e");
        }
      } // if
    } // if
  } // visit_type

  static void write(const dag_map & registry, graphviz & gv) {
    Agnode_t *begin = nullptr, *last = nullptr;
    walk<typename ControlPolicy::control_points>(
      point_writer(registry, gv, begin, last));
  }

  static void write_sorted(const typename ControlPolicy::sorted_type & sorted,
    graphviz & gv) {
    std::vector<Agnode_t *> nodes;

    for(auto cp : sorted) {
      for(auto n : cp.second) {
        auto * node = gv.add_node(n->identifier().c_str(), n->label().c_str());
        nodes.push_back(node);

        gv.set_node_attribute(node, "color", "black");
        gv.set_node_attribute(node, "style", "filled");
        gv.set_node_attribute(
          node, "fillcolor", colors[static_cast<size_t>(cp.first) % 4]);
      } // for
    } // for

    for(size_t n{1}; n < nodes.size(); ++n) {
      auto * edge = gv.add_edge(nodes[n - 1], nodes[n]);
      gv.set_edge_attribute(edge, "penwidth", "1.5");
    } // for
  } // write_sorted

private:
  void set_begin(Agnode_t * b) const {
    if(!begin)
      begin = b;
  }

  const dag_map & registry_;
  graphviz & gv_;
  Agnode_t *&begin, *&last;
  int depth_;

}; // struct point_writer

#endif // FLECSI_ENABLE_GRAPHVIZ

/// \}
} // namespace run_impl
} // namespace flecsi
/// \endcond

#endif
