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

template<auto P>
struct control_point : util::constant<P> {};
template<auto P>
struct meta_point : util::constant<P> {};

/*!
  Allow users to define cyclic control points. Cycles can be nested.

  @tparam Predicate     A predicate function that determines when
                        the cycle should end.
  @tparam ControlPoints A variadic list of control points within the cycle.
 */

template<auto Predicate, typename... ControlPoints>
struct cycle {

  using type = util::types<ControlPoints...>;

  static constexpr auto predicate = Predicate;
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
  constexpr void visit(C cp, util::types<TT...> *) {
    (visit(cp, static_cast<TT *>(nullptr)), ...);
  }
  template<auto P, class... TT>
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
struct init_walker {

  using control_points_enum = typename P::control_points_enum;
  using dag_map = typename P::dag_map;

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

template<typename P>
struct point_walker {

  using control_points_enum = typename P::control_points_enum;
  using sorted_type = typename P::sorted_type;
  using policy_type = typename P::policy_type;

  point_walker(const sorted_type & sorted,
    int & exit_status,
    policy_type * policy = nullptr)
    : sorted_(sorted), exit_status_(exit_status), policy_(policy) {}

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
        if constexpr(P::is_control_base_policy)
          node->execute(policy_);
        else
          exit_status_ |= node->execute(policy_);
      } // for
    }
    else {
      // This is a cycle -> create a new control point walker to recurse
      // the cycle.
      auto test = [this]() {
        if constexpr(P::is_control_base_policy)
          return ElementType::predicate(*policy_);
        else
          return ElementType::predicate();
      };
      while(test()) {
        point_walker walker(sorted_, exit_status_, policy_);
        walk<typename ElementType::type>(walker);
      } // while
    } // if
  } // visit_type

private:
  const sorted_type & sorted_;
  int & exit_status_;
  policy_type * policy_;
}; // struct point_walker

#if defined(FLECSI_ENABLE_GRAPHVIZ)

template<typename P>
struct point_writer {
  using control_points_enum = typename P::control_points_enum;
  using control_points = typename P::control_points;
  using dag_map = typename P::dag_map;
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

      if constexpr(is_meta<control_points>(ElementType::value)) {
        gv_.set_node_attribute(root, "style", "rounded,dashed");
        gv_.set_node_attribute(root, "color", "#777777");
      }
      else {
        gv_.set_node_attribute(root, "style", "rounded");
      }

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
    walk<control_points>(point_writer(registry, gv, begin, last));
  }

  static void write_sorted(const typename P::sorted_type & sorted,
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
