// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUN_CONTROL_HH
#define FLECSI_RUN_CONTROL_HH

#include "flecsi/config.hh"
#include "flecsi/flog.hh"
#include "flecsi/run/init.hh"
#include "flecsi/run/types.hh"
#include "flecsi/util/constant.hh"
#include "flecsi/util/dag.hh"
#include "flecsi/util/demangle.hh"

#include <functional>
#include <map>
#include <vector>

namespace flecsi {
namespace run {
/// \defgroup control Control Model
/// Types for defining, extending, and executing control-flow graphs.
/// \ingroup runtime
/// \{

/// A control point for application use.
/// \tparam CP control point enumerator
/// \deprecated Use \c control_base::point.
template<auto CP>
using control_point = run_impl::control_point<CP>;

/*!
  A control-flow cycle.
  \tparam P tested before each iteration
  \tparam CP \c control_point or \c cycle types
  \deprecated Use \c control_base::cycle.
 */
template<bool (*P)(), typename... CP>
using cycle = run_impl::cycle<P, CP...>;

/*!
  Base class for providing default implementations for optional interfaces.
 */

struct control_base {
  /// A control point for application use.
  /// \tparam CP control point enumerator
  template<auto CP>
  using point = run_impl::control_point<CP>;

  /// A control point for specialization use.
  /// \tparam CP control point enumerator
  template<auto CP>
  using meta = run_impl::meta_point<CP>;

  /// A control-flow cycle.
  /// \tparam P of type `bool (*)(user_policy&)` tested before each iteration,
  /// where `user_policy` inherits from control_base.  This provides access to
  /// the control policy instance during policy execution.
  /// \tparam CP \c point or \c cycle types
  template<auto P, typename... CP>
  using cycle = run_impl::cycle<P, CP...>;

  /// Type for specifying control points
  /// \tparam TT pack of \c point, \c meta, or \c cycle
  template<class... TT>
  using list = util::types<TT...>;

  /*!
    Exception class for control points.
  */
  struct exception {
    int code; /// status code
  };

  struct node_policy {};
};

#ifdef DOXYGEN
/// An example control policy that is not really implemented.
/// Inheriting from \c control_base is required for using \c control::invoke.
struct control_policy : control_base {
  /// The labels for the control-flow graph.
  enum control_points_enum {};
  /// The control-flow graph.
  /// Each element is a \c control_base::point or a \c control_base::cycle.
  using control_points = list<>;
  /// Base class for control point objects.
  /// Optional; must not be defined if \c control::invoke is used.
  struct node_policy {};
};

/// A control policy must provide names for its control points.
inline const char * operator*(control_policy::control_points_enum);
#endif

/*!
  The control type provides a control model for specifying a
  set of control points as a coarse-grained control flow graph,
  with each node of the graph specifying a set of actions as a
  directed acyclic graph (DAG). The actions under a control point
  DAG are topologically sorted to respect dependency edges, which can
  be specified through the dag interface.

  If Graphviz support is enabled, the control flow graph and its DAG nodes
  can be written to a graphviz file that can be compiled and viewed using
  the \em dot program.

  \tparam P policy type like \c control_policy
 */

template<typename P>
struct control {

  using policy_type = P;

  static constexpr bool is_control_base_policy =
    std::is_base_of_v<control_base, P>;

  using target_type =
    std::conditional_t<is_control_base_policy, void (*)(P &), int (*)()>;

private:
  using control_points = run_impl::to_types_t<P>;
  using control_points_enum = typename P::control_points_enum;
  using node_policy = typename P::node_policy;

  using point_walker = run_impl::point_walker<control>;
  friend point_walker;

  using init_walker = run_impl::init_walker<control>;
  friend init_walker;

#if defined(FLECSI_ENABLE_GRAPHVIZ)
  using point_writer = run_impl::point_writer<control>;
  friend point_writer;
#endif

  /*
    Control node type. This just adds an executable target.
   */

  struct control_node : node_policy {

    template<typename... Args>
    control_node(target_type target, Args &&... args)
      : node_policy(std::forward<Args>(args)...), target_(target) {}

    std::conditional_t<is_control_base_policy, void, int> execute(P * p) const {
      if constexpr(is_control_base_policy)
        target_(*p);
      else
        return target_();
    }

  private:
    target_type target_;
  }; // struct control_node

  using dag = util::dag<control_node>;

  /*
    Use the node type that is defined by the specialized DAG.
   */

  using node_type = typename dag::node_type;

public:
  using sorted_type = std::map<control_points_enum, typename dag::sorted_type>;
  using dag_map = std::map<control_points_enum, dag>;

private:
  /*
    Initialize the control point dags. This is necessary in order to
    assign labels in the case that no actions are registered at one
    or more control points.
  */
  control() {
    run_impl::walk<control_points>(init_walker(registry_));
  }

  static control & instance() {
    static control c;
    return c;
  }

  /*
    Return the dag at the given control point.
  */
  dag & control_point_dag(control_points_enum cp) {
    registry_.try_emplace(cp, *cp);
    return registry_[cp];
  }

  /*
    Return a map of the sorted dags under each control point.
  */
  sorted_type sort() const {
    sorted_type sorted;
    for(auto & d : registry_) {
      sorted.try_emplace(d.first, d.second.sort());
    }
    return sorted;
  }

  /*
    Run the control model.
  */
  [[nodiscard]] int run(P * p) const {
    int status{flecsi::run::status::success};
    run_impl::walk<control_points>(point_walker(sort(), status, p));
    return status;
  } // run

  /*
    Output a graph of the control model.
  */
#if defined(FLECSI_ENABLE_GRAPHVIZ)
  void write(std::string_view p, const char * file) const {
    flecsi::util::graphviz gv(std::string(p) + " control model");
    point_writer::write(registry_, gv);
    gv.write(file);
  } // write

  void write_sorted(std::string_view p, const char * file) const {
    flecsi::util::graphviz gv(std::string(p) + " actions");
    point_writer::write_sorted(sort(), gv);
    gv.write(file);
  } // write_sorted
#endif

  dag_map registry_;
  std::conditional_t<is_control_base_policy, std::nullptr_t, P> policy_;

public:
  /// Return the control policy object.
  /// It is default-initialized when the control model is first used
  /// (typically to register an action) and
  /// destroyed at the end of the program.
  /// This function cannot be used if \c P inherits from \c control_base.
  /// \deprecated derive \c P from \c control_base and use the parameter
  ///   passed to each action
  static P & state() {
    static_assert(!is_control_base_policy);
    return instance().policy_;
  }

  /*!
    The action type provides a mechanism to add execution elements to the
    FleCSI control model.  The \c P::node_policy structure that gets
    instantiated was intended originally for sharing of data across actions.
    This usage is deprecated, and \c P::node_policy now should be defined as an
    empty \c struct.

    \tparam T function to call, of type `void(P&)` if \c P inherits from
      \c control_base and `int()` otherwise
    @tparam CP The control point under which this action is executed.
    @tparam M  Boolean indicating whether or not the action is a meta action.
               This is intended for specialization developers; application
               developers should omit this parameter (defaulting it to
               \c false).
   */

  template<target_type T, control_points_enum CP, bool M = false>
  struct [[nodiscard]] action {

    template<target_type, control_points_enum, bool>
    friend struct action;

    /*!
      Add a function to be executed under the specified control point.
     */

    action() : node_(util::symbol<*T>(), T) {
      static_assert(M == run_impl::is_meta<control_points>(CP),
        "you cannot use this interface for internal control points!");
      instance().control_point_dag(CP).push_back(&node_);
    }

    /*!
      Add a function to be executed under the specified control point.
      \deprecated Use the argument-less constructor.

      @param args   A variadic list of arguments that are forwarded to the
                    user-defined node type, as specified in the control policy
     */

    template<typename... Args>
    [[deprecated("actions should not take arguments")]] action(Args &&... args)
      : node_(util::symbol<*T>(), T, std::forward<Args>(args)...) {
      static_assert(M == run_impl::is_meta<control_points>(CP),
        "you cannot use this interface for internal control points!");
      instance().control_point_dag(CP).push_back(&node_);
    }

    /*
      Dummy type used to force namespace scope execution.
     */

    struct dependency {};

    /*!
      Add a dependency on the given action.

      \tparam V must be the same as \a ControlPoint
      @param from The upstream node in the dependency.
     */

    template<target_type U, control_points_enum V>
    dependency add(action<U, V> const & from) {
      static_assert(CP == V,
        "you cannot add dependencies between actions under different control "
        "points");
      node_.push_back(&from.node_);
      return {};
    }

  protected:
    node_type node_;
  }; // struct action

  /// An action registration on a \c meta_point for a specialization.
  /// \tparam T function
  /// \tparam CP control point enumerator
  template<target_type T, control_points_enum CP>
  using meta = action<T, CP, true>;

private:
  // Perform all the work for invoke.
  // The reason for this factorization is so execute and invoke can
  // complain differently about a non-empty node_policy (via a
  // deprecation warning and a static_assert, respectively).
  template<class... AA>
  [[nodiscard]] static int do_invoke(AA &&... aa) {
    if constexpr(is_control_base_policy) {
      try {
        P pol(std::forward<AA>(aa)...);
        return instance().run(&pol);
      }
      catch(control_base::exception e) {
        return e.code;
      }
    }
    else {
      static_assert(!sizeof...(aa), "arguments allowed only with control_base");
      return instance().run(nullptr);
    }
  }

public:
  /*!
    Execute the control model. This method does a topological sort of the
    actions under each of the control points to determine a non-unique, but
    valid ordering, and executes the actions.  An object of type \c P is
    initialized from \a aa and passed to each action; it is
    destroyed before this function returns.
    \c control_base::exception can be thrown for early
    termination.
    The control policy must inherit from \c control_base and
    must not define a \c node_policy,
    or the code will fail to compile.
    \return code from a thrown \c control_base::exception or 0
   */
  template<class... AA>
  [[nodiscard]] static int invoke(AA &&... aa) {
    static_assert(
      is_control_base_policy, "control policy must inherit from control_base");
    static_assert(std::is_same_v<typename policy_type::node_policy,
                    control_base::node_policy>,
      "control policy must not define node_policy");
    return do_invoke(std::forward<AA>(aa)...);
  }

  /// Perform the same operation as \c #invoke with no arguments.
  /// \c P need not inherit from \c control_base, if it does not, no object of
  /// it is created and actions are called with no arguments.
  /// \c P may define \c node_policy.
  /// \return `invoke()` if \c P inherits from \c control_base, otherwise the
  ///   bitwise or of return values of executed actions
  /// \deprecated Call \c #invoke directly or use \c runtime::control.
  [[deprecated("use invoke")]] [[nodiscard]] static int execute() {
    return do_invoke();
  }

  /*!
    Process control model command-line options.
    \param s initialization status from \c initialize
    \return status of control model output if requested, else \a s
    \deprecated Call \c #write_graph or \c #write_actions as needed.
    \see \c #status
   */
  [[deprecated("use flecsi::runtime")]] [[nodiscard]] static int check_status(
    int s) {
#if defined(FLECSI_ENABLE_GRAPHVIZ)
    // If a confused client calls this without having called initialize,
    // argv0 will be empty, which is a mild form of failure.
    switch(s) {
      case flecsi::run::status::control_model:
        write_graph(argv0, (argv0 + "-control-model.dot").c_str());
        break;
      case flecsi::run::status::control_model_sorted:
        write_actions(argv0, (argv0 + "-control-model-sorted.dot").c_str());
        break;
      default:
        break;
    } // switch
#endif
    return s;
  } // check_status

#ifdef FLECSI_ENABLE_GRAPHVIZ
  /// Write a Dot graph of control points and actions.
  /// \param p program name for graph title
  /// \param f output file
  static void write_graph(std::string_view p, const char * f) {
    instance().write(p, f);
  }
  /// Write a Dot graph of the sorted sequence of actions.
  /// \param p program name for graph title
  /// \param f output file
  static void write_actions(std::string_view p, const char * f) {
    instance().write_sorted(p, f);
  }
#endif
}; // struct control

struct call_policy : control_base {
  enum control_points_enum { single };
  using control_points = list<point<single>>;

  template<class F>
  explicit call_policy(F && f) : f(std::forward<F>(f)) {}
  int operator()() const {
    return f();
  }

private:
  std::function<int()> f;
};

inline const char *
operator*(call_policy::control_points_enum) {
  return "single";
}

/// A trivial control model that calls a single function.
/// Its control policy object can be constructed from any callable with the
/// signature `int()`.
using call = control<call_policy>;

/// \}
} // namespace run
} // namespace flecsi

#endif
