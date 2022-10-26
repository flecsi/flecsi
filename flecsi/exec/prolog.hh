// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

// Backend-independent task argument handling.

#ifndef FLECSI_EXEC_PROLOG_HH
#define FLECSI_EXEC_PROLOG_HH

#include "flecsi/data/topology_slot.hh"
#include "flecsi/flog.hh"
#include "flecsi/util/demangle.hh"

namespace flecsi {
inline flog::devel_tag task_prologue_tag("task_prologue");
}

// task_prologue is implemented per backend:
#if FLECSI_BACKEND == FLECSI_BACKEND_legion
#include "flecsi/exec/leg/task_prologue.hh"
#elif FLECSI_BACKEND == FLECSI_BACKEND_mpi
#include "flecsi/exec/mpi/task_prologue.hh"
#endif

/// \cond core
namespace flecsi::exec {
/// \addtogroup execution
/// \{

#ifdef DOXYGEN // implemented per-backend
/// Handling for low-level special task parameters/arguments.
/// The exact member function signatures may vary between backends.
/// \tparam Proc for the task being executed
template<task_processor_type_t Proc>
struct task_prologue {
protected:
  /// Default constructible.
  task_prologue();

  /// Send a raw field reference to a raw accessor.
  template<typename T, Privileges P, class Topo, typename Topo::index_space S>
  void visit(data::accessor<data::raw, T, P> &,
    const data::field_reference<T, data::raw, Topo, S> &);
  /// Send an index future to a single future.
  template<typename R>
  void visit(future<R> &, const future<R, launch_type_t::index> &);
};
#endif

/*!
  Analyzes task arguments and updates data objects before launching a task.
*/
template<task_processor_type_t ProcessorType>
struct prolog : task_prologue<ProcessorType> {
  // Note that accessors here may be empty versions made to be serialized and
  // that the arguments have been moved from (which doesn't matter for the
  // relevant types).
  template<class P, class... AA>
  prolog(P & p, AA &... aa) {
    util::annotation::rguard<util::annotation::execute_task_prolog> ann;
    std::apply([&](auto &... pp) { (visit(pp, aa), ...); }, p);
  }

private:
  template<class A>
  auto visitor(A & a) {
    return
      [&](auto & p, auto && f) { visit(p, std::forward<decltype(f)>(f)(a)); };
  }

  using task_prologue<ProcessorType>::visit; // for raw accessors, futures, etc.

  template<class P, class A>
  std::enable_if_t<std::is_base_of_v<data::send_tag, P>> visit(P & p, A && a) {
    p.send(visitor(a));
  }

  /*--------------------------------------------------------------------------*
    Non-FleCSI Data Types
   *--------------------------------------------------------------------------*/

  // The const prevents being a better match than more specialized overloads.
  // This is constrained opposite the above because it is more specialized.
  template<class P, class A>
  static std::enable_if_t<!std::is_base_of_v<data::send_tag, P>> visit(P &,
    const A &) {
    flog::devel_guard guard(task_prologue_tag);
    flog_devel(info) << "Skipping argument with type " << util::type<A>()
                     << std::endl;
  } // visit
};

/// \}
} // namespace flecsi::exec
/// \endcond

#endif
