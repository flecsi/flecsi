// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

// Forward declarations for task execution, for use by templated code on which
// the definition of reduce depends.  Other clients must include execution.hh.

#ifndef FLECSI_EXEC_FWD_HH
#define FLECSI_EXEC_FWD_HH

#include "flecsi/exec/task_attributes.hh"
#include "flecsi/util/types.hh" // Color

#include <memory>
#include <optional>
#include <utility>

namespace flecsi {
struct scheduler;

namespace topo {
template<class, class>
struct topology;
}

/// A topology instance.
/// Pass to a task expecting a \c topology_accessor.
/// \tparam Topo specialization
/// \note A \c specialization provides aliases for both these types.
/// \warning No topologies may exist outside of \c start or \c control.
///
/// \ns.
/// \ingroup data
template<class Topo>
#ifdef DOXYGEN
struct topology {
  /// A topology can be constructed from its \c coloring type.
  topology(scheduler &, const typename Topo::coloring &);
  /// Immovable.
  topology(topology &&) = delete; // some internal topologies are movable

  /// Return the number of colors over which the topology is partitioned.
  Color colors() const;
};
#else
using topology = topo::topology<Topo, typename Topo::base>;
#endif

/// \addtogroup execution
/// \{

/*!
  Execute a reduction task.

  @tparam Reduction  The reduction operation type.
  \return a \ref future providing the reduced return value

  \see \c execute about parameter and argument types.

  \ns.
 */
template<auto & Task,
  class Reduction,
  TaskAttributes Attributes = flecsi::loc | flecsi::leaf,
  typename... Args>
[[nodiscard]] auto reduce(Args &&...);

/*!
  Execute a task.

  @tparam TASK          The user task.
    Its parameters must be copyable or a reference to a const, movable type.
    Any that is a pointer must be to a const type or to a function.
    If \a ATTRIBUTES specifies an MPI task, parameters need merely be movable.
  @tparam ATTRIBUTES    The task attributes mask.
  @tparam ARGS The user-specified task arguments, implicitly converted to the
    parameter types for \a TASK.
    Certain FleCSI-defined parameter types accept particular, different
    argument types that serve as selectors for information stored by the
    backend; each type involved documents the correspondence.
    If a parameter and its argument are each a (reference to a) \c std::vector
    or a \c std::tuple (of the same size), their elements are treated as
    separate parameters/arguments (with the unusual corollary that a
    `std::vector<int>` matches a parameter of type `std::vector<long>`).
  \return a \ref future providing the value(s) returned from the task

  \ns.
 */

template<auto & TASK,
  TaskAttributes ATTRIBUTES = flecsi::loc | flecsi::leaf,
  typename... ARGS>
auto
execute(ARGS &&... args) {
  return reduce<TASK, void, ATTRIBUTES>(std::forward<ARGS>(args)...);
} // execute

struct runtime;

/// Launches tasks according to their execution-space (template) parameters.
/// An instance is passed to control-model actions that accept it.
/// \note MPI tasks cannot use this interface.
struct scheduler {
  explicit scheduler(flecsi::runtime & r) : r(r) {}
  /// Immovable.
  scheduler(scheduler &&) = delete;

  /// Launch a variant of a reduction task.
  template<class, class R, class... AA>
  auto reduce(AA &&...);
  /// Launch a variant of a task.
  /// \tparam V like \c task_class
  template<class V, class... AA>
  auto execute(AA &&... aa) {
    return reduce<V, void>(std::forward<AA>(aa)...);
  }
  /// Execute a variant of a test task.
  template<class V, class... AA>
  [[nodiscard]] int test(AA &&... aa);

  /// Launch a reduction task.
  /// \tparam R reduction operation
  /// \return a \ref future providing the reduced return value
  /// \see \c execute about parameter and argument types.
  template<auto &, class R, class... AA>
  auto reduce(AA &&...);
  /// Launch a task.
  template<auto & F, class... AA>
  auto execute(AA &&... aa) {
    return reduce<F, void>(std::forward<AA>(aa)...);
  }
  /// Execute a test task.
  template<auto &, class... AA>
  [[nodiscard]] int test(AA &&...);

  /// Create a topology instance with specialization support.
  /// Calls the specialization's \c initialize on the topology instance.
  /// \param p where to store the topology
  /// \param c coloring (perhaps from an \link
  ///   topo::specialization::mpi_coloring `mpi_coloring`\endlink)
  /// \param aa further specialization-specific parameters
  /// \return the new instance
  template<class T, class... AA>
  auto & allocate(std::unique_ptr<topology<T>> & p,
    const typename T::coloring & c,
    AA &&... aa);

  // Will become a non-static member of runtime in 3.
  static std::optional<scheduler> instance;

  /// Get the runtime (which created this scheduler).
  const flecsi::runtime & runtime() const {
    return r;
  }

private:
  flecsi::runtime & r;
};
inline std::optional<scheduler> scheduler::instance;

#ifdef DOXYGEN
/// Example task class which is not really implemented.
struct task_class {
  /// A task can be a static member function.
  /// Parameters and return type can vary.
  /// \param s execution space, if desired
  static void task(exec::cpu s) noexcept;

  /// A task can be a static member function template.
  /// Parameters and return type can vary.
  /// An unspecified specialization that is not deleted is launched.
  /// \tparam S execution \ref space
  ///   (\c exec::cpu, \c exec::gpu, or \c exec::omp)
  /// \param s only allowed signature variation among specializations
  template<class S>
  static void task(S s) noexcept;
};
#endif

namespace exec {
// Learn from backend whether a trace is supported, active, and not skipped.
inline bool is_tracing();
} // namespace exec

/// \}
} // namespace flecsi

#endif
