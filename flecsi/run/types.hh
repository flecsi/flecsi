/*
    @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
   /@@/////  /@@          @@////@@ @@////// /@@
   /@@       /@@  @@@@@  @@    // /@@       /@@
   /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
   /@@////   /@@/@@@@@@@/@@       ////////@@/@@
   /@@       /@@/@@//// //@@    @@       /@@/@@
   /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
   //       ///  //////   //////  ////////  //

   Copyright (c) 2016, Triad National Security, LLC
   All rights reserved.
                                                                              */
#pragma once

/*! @file */

#include <flecsi-config.h>

#include "flecsi/flog.hh"
#include "flecsi/util/constant.hh"
#include "flecsi/util/dag.hh"
#include "flecsi/util/tuple_walker.hh"

#if defined(FLECSI_ENABLE_GRAPHVIZ)
#include "flecsi/util/graphviz.hh"
#endif

#include <map>
#include <vector>

namespace flecsi {
namespace run_impl {

template<auto P>
struct control_point : util::constant<P> {};
template<auto P>
struct meta_point : util::constant<P> {};

/*!
  Allow users to define cyclic control points. Cycles can be nested.

  @tparam Predicate     A predicate function that determines when
                        the cycle should end.
  @tparam ControlPoints A variadic list of control points within the cycle.

  @ingroup control
 */

template<bool (*Predicate)(), typename... ControlPoints>
struct cycle {

  using type = std::tuple<ControlPoints...>;
  static constexpr size_t last = std::tuple_size<type>::value - 1;

  template<size_t E, typename T>
  struct recurse;

  template<size_t E, bool (*P)(), typename... Ps>
  struct recurse<E, cycle<P, Ps...>> {
    using type = std::tuple<Ps...>;
    static constexpr const auto & value =
      recurse<E, typename std::tuple_element<E, type>::type>::value;
  };

  template<size_t E, auto Value>
  struct recurse<E, control_point<Value>> {
    static constexpr const auto & value = Value;
  };

  template<size_t E, auto Value>
  struct recurse<E, meta_point<Value>> {
    static constexpr const auto & value = Value;
  };

  static constexpr auto & begin =
    recurse<0, typename std::tuple_element<0, type>::type>::value;
  static constexpr auto & end =
    recurse<last, typename std::tuple_element<last, type>::type>::value;

  static bool predicate() {
    return Predicate();
  } // run

}; // struct cycle

/*
  Utility type for meta point search.
 */

template<class C>
struct search {
  template<class T>
  constexpr void visit(C, T *) {}
  template<auto P>
  constexpr void visit(C cp, meta_point<P> *) {
    if(cp == P)
      set(true);
  }
  template<auto P>
  constexpr void visit(C cp, control_point<P> *) {
    if(cp == P)
      set(false);
  }
  template<class... TT>
  constexpr void visit(C cp, std::tuple<TT...> *) {
    (visit(cp, static_cast<TT *>(nullptr)), ...);
  }
  template<bool (*P)(), class... TT>
  constexpr void visit(C cp, cycle<P, TT...> *) {
    visit(cp, static_cast<typename cycle<P, TT...>::type *>(nullptr));
  }

  constexpr operator bool() const {
    if(!found)
      throw "control point is not defined";
    return result;
  }

private:
  constexpr void set(bool v) {
    if(found)
      throw "control point occurs multiple times";
    found = true;
    result = v;
  }
  bool found = false;
  bool result = false; // until C++20
};

template<class T, class C>
constexpr bool
is_meta(C cp) {
  search<C> ret;
  ret.visit(cp, static_cast<T *>(nullptr));
  return ret;
}

/*
  Helper type to initialize dag labels.
 */

template<typename P>
struct init_walker : public util::tuple_walker<init_walker<P>> {

  using control_points_enum = typename P::control_points_enum;
  using dag = util::dag<typename P::control_node>;

  init_walker(std::map<control_points_enum, dag> & registry)
    : registry_(registry) {}

  template<typename ElementType>
  void visit_type() {

    if constexpr(std::is_same<typename ElementType::type,
                   control_points_enum>::value) {
      registry_.try_emplace(ElementType::value, *ElementType::value);
    }
    else {
      init_walker(registry_).template walk_types<typename ElementType::type>();
    } // while
  } // visit_type

private:
  std::map<control_points_enum, dag> & registry_;

}; // struct init_walker

/*!
  The point_walker class allows execution of statically-defined
  control points.

  @ingroup control
 */

template<typename P>
struct point_walker : public util::tuple_walker<point_walker<P>> {

  using control_points_enum = typename P::control_points_enum;
  using node_type = typename P::node_type;

  point_walker(
    std::map<control_points_enum, std::vector<node_type const *>> sorted,
    int & exit_status)
    : sorted_(sorted), exit_status_(exit_status) {}

  /*!
    Handle the tuple type \em ElementType.

    @tparam ElementType The tuple element type. This can either be a
                        control_points_enum or a \em cycle. Cycles are defined
                        by the specialization and must conform to the interface
                        used in the appropriate visit_type method.
   */

  template<typename ElementType>
  void visit_type() {

    if constexpr(std::is_same<typename ElementType::type,
                   control_points_enum>::value) {

      // This is not a cycle -> execute each action for this control point.
      for(auto & node : sorted_[ElementType::value]) {
        exit_status_ |= node->execute();
      } // for
    }
    else {
      // This is a cycle -> create a new control point walker to recurse
      // the cycle.
      while(ElementType::predicate()) {
        point_walker walker(sorted_, exit_status_);
        walker.template walk_types<typename ElementType::type>();
      } // while
    } // if
  } // visit_type

private:
  std::map<control_points_enum, std::vector<node_type const *>> sorted_;
  int & exit_status_;

}; // struct point_walker

#if defined(FLECSI_ENABLE_GRAPHVIZ)

template<typename P>
struct point_writer : public flecsi::util::tuple_walker<point_writer<P>> {
  using control_points_enum = typename P::control_points_enum;
  using control_points = typename P::control_points;
  using node_type = typename P::node_type;
  using dag = util::dag<typename P::control_node>;
  using graphviz = flecsi::util::graphviz;

  static constexpr const char * colors[4] = {"#77c3ec",
    "#b8e2f2",
    "#4eb2e0",
    "#9dd9f3"};

  point_writer(std::map<control_points_enum, dag> registry,
    graphviz & gv,
    int depth = 0)
    : registry_(registry), gv_(gv), depth_(depth) {}

  template<typename ElementType>
  void visit_type() {
    if constexpr(std::is_same<typename ElementType::type,
                   control_points_enum>::value) {
      auto & dag = registry_[ElementType::value];

      auto * root = gv_.add_node(dag.label().c_str(), dag.label().c_str());
      gv_.set_node_attribute(root, "shape", "box");

      if constexpr(is_meta<control_points>(ElementType::value)) {
        gv_.set_node_attribute(root, "style", "rounded,dashed");
        gv_.set_node_attribute(root, "color", "#777777");
      }
      else {
        gv_.set_node_attribute(root, "style", "rounded");
      }

      if(size_t(ElementType::value) > 0) {
        auto & last = registry_[static_cast<control_points_enum>(
          static_cast<size_t>(ElementType::value) - 1)];
        auto * last_node = gv_.node(last.label().c_str());
        auto * edge = gv_.add_edge(last_node, root);
        gv_.set_edge_attribute(edge, "color", "#1d76db");
        gv_.set_edge_attribute(edge, "fillcolor", "#1d76db");
        gv_.set_edge_attribute(edge, "style", "bold");
      } // if

      dag.add(gv_, colors[static_cast<size_t>(ElementType::value) % 4]);

      for(auto & n : dag) {
        if(n->size() == 0) {
          auto * edge = gv_.add_edge(root, gv_.node(n->identifier().c_str()));
          gv_.set_edge_attribute(edge, "penwidth", "1.5");
        } // if
      } // for
    }
    else {
      point_writer(registry_, gv_, depth_ - 1)
        .template walk_types<typename ElementType::type>();

      auto & begin = registry_[ElementType::begin];
      auto & end = registry_[ElementType::end];

      auto * edge = gv_.add_edge(
        gv_.node(end.label().c_str()), gv_.node(begin.label().c_str()));

      gv_.set_edge_attribute(edge, "label", " cycle");
      gv_.set_edge_attribute(edge, "color", "#1d76db");
      gv_.set_edge_attribute(edge, "fillcolor", "#1d76db");
      gv_.set_edge_attribute(edge, "style", "dashed,bold");

      if constexpr(ElementType::begin == ElementType::end) {
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

  static void write_sorted(
    std::map<control_points_enum, std::vector<node_type const *>> sorted,
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
  std::map<control_points_enum, dag> registry_;
  graphviz & gv_;
  int depth_;

}; // struct point_writer

#endif // FLECSI_ENABLE_GRAPHVIZ

} // namespace run_impl
} // namespace flecsi
