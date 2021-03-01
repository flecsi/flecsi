// Backend-independent task argument handling.

#ifndef FLECSI_EXEC_PROLOG_HH
#define FLECSI_EXEC_PROLOG_HH

#include "flecsi/data/reference.hh"
#include "flecsi/data/topology_slot.hh"
#include "flecsi/flog.hh"
#include "flecsi/util/demangle.hh"

namespace flecsi {
inline log::devel_tag task_prologue_tag("task_prologue");
}

// task_prologue is implemented per backend:
#if FLECSI_RUNTIME_MODEL == FLECSI_RUNTIME_MODEL_legion
#include "flecsi/exec/leg/task_prologue.hh"
#elif FLECSI_RUNTIME_MODEL == FLECSI_RUNTIME_MODEL_mpi
#include "flecsi/exec/mpi/task_prologue.hh"
#endif

namespace flecsi::exec {

/*!
  Analyzes task arguments and updates data objects before launching a task.

  @ingroup execution
*/

struct prolog : task_prologue {
  // Note that accessors here may be empty versions made to be serialized.
  template<class P, class... AA>
  prolog(P & p, AA &... aa) {
    std::apply([&](auto &... pp) { (visit(pp, aa), ...); }, p);
  }

private:
  template<class A>
  auto visitor(A & a) {
    return
      [&](auto & p, auto && f) { visit(p, std::forward<decltype(f)>(f)(a)); };
  }

  using task_prologue::visit; // for raw accessors, futures, etc.

  template<class P, class A>
  std::enable_if_t<std::is_base_of_v<data::send_tag, P>> visit(P & p, A && a) {
    p.send(visitor(a));
  }

  /*--------------------------------------------------------------------------*
    Non-FleCSI Data Types
   *--------------------------------------------------------------------------*/

  // The const prevents being a better match than more specialized overloads.
  template<class P, class A>
  static std::enable_if_t<!std::is_base_of_v<data::convert_tag, A>> visit(P &,
    const A &) {
    log::devel_guard guard(task_prologue_tag);
    flog_devel(info) << "Skipping argument with type " << util::type<A>()
                     << std::endl;
  } // visit
};

} // namespace flecsi::exec

#endif
