// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

// Backend-independent task argument handling.

#ifndef FLECSI_EXEC_PROLOG_HH
#define FLECSI_EXEC_PROLOG_HH

#include "flecsi/data/copy_plan.hh"
#include "flecsi/data/topology.hh"
#include "flecsi/data/topology_slot.hh"
#include "flecsi/flog.hh"
#include "flecsi/util/demangle.hh"

namespace flecsi::exec {
struct prolog_base {
  explicit prolog_base(scheduler & s) : sched(&s) {}
  ~prolog_base() {
    for(auto & epilog : epilog_wrappers)
      epilog();
  }

protected:
  template<Privileges P, class R>
  void add_copy(const R & r) {
    if(const data::copy_plan * const p =
         r.get_region().template ghost_copy<P>(*sched, r))
      copies[p].push_back({r.fid(), privilege_write(P)});
  }
  template<processor P>
  inline void issue_copy() const {
    for(const auto & [p, ff] : copies)
      p->issue_copy<P>(ff);
  }

  scheduler * sched;
  std::vector<std::function<void()>> epilog_wrappers;

private:
  std::map<const data::copy_plan *, data::copy_request::vec> copies;
};
} // namespace flecsi::exec

// task_prologue is implemented per backend:
#if FLECSI_BACKEND == FLECSI_BACKEND_legion
#include "flecsi/exec/leg/task_prologue.hh"
#elif FLECSI_BACKEND == FLECSI_BACKEND_mpi
#include "flecsi/exec/mpi/params.hh"
#elif FLECSI_BACKEND == FLECSI_BACKEND_hpx
#include "flecsi/exec/hpx/task_prologue.hh"
#endif

/// \cond core
namespace flecsi::exec {
/// \addtogroup execution
/// \{

#ifdef DOXYGEN // implemented per-backend
/// Handling for low-level special task parameters/arguments.
/// The exact member function signatures may vary between backends.
/// \tparam Proc for the task being executed
template<processor Proc>
struct task_prologue : prolog_base {
protected:
  /// Constructible as is \c prolog_base.
  explicit task_prologue(scheduler &);

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
template<processor Proc>
struct prolog : task_prologue<Proc> {
  // Note that accessors are (initially) empty and
  // that the arguments have been moved from (which doesn't matter for the
  // relevant types).
  template<class P, class... AA>
  prolog(P & p, AA &... aa) : task_prologue<Proc>(*scheduler::instance) {
    util::annotation::rguard<util::annotation::execute_task_prolog> ann;
    std::apply([&](auto &... pp) { (visit(pp, aa), ...); }, p);
    this->template issue_copy<Proc>();
  }

private:
  template<class A>
  auto visitor(A & a) {
    return
      [&](auto & p, auto && f) { visit(p, std::forward<decltype(f)>(f)(a)); };
  }

  using task_prologue<Proc>::visit; // for raw accessors, futures, etc.

  static void visit(data::detail::host_only &, decltype(nullptr)) {
    static_assert(Proc != flecsi::exec::processor::toc,
      "accessor type is supported only on host");
  }

  template<class P, class A>
  std::enable_if_t<std::is_base_of_v<data::send_tag, P>> visit(P & p, A && a) {
    p.send(visitor(a));
  }

  template<class P, class T>
  void visit(P & p, data::topology_slot<T> & t) {
    visit(p, t.get());
  }

  template<class P, class A>
  std::enable_if_t<detail::must_convert<A>::value> visit(std::vector<P> & pv,
    const std::vector<A> & av) {
    flog_assert(pv.size() == av.size(), "parameter/argument count mismatch");
    P * p = pv.data();
    for(auto & a : av)
      visit(*p++, a);
  }
  template<class... PP, class... AA>
  void visit(std::tuple<PP...> & pt, const std::tuple<AA...> & at) {
    std::apply(
      [&](auto &... pp) {
        std::apply([&](auto &... aa) { (visit(pp, aa), ...); }, at);
      },
      pt);
  }

  // The const prevents being a better match than more specialized overloads.
  // This is constrained opposite the above because it is more specialized.
  template<class P, class A>
  static std::enable_if_t<!std::is_base_of_v<data::send_tag, P>>
  visit(const P &, const A &) {} // visit
};

/// \}
} // namespace flecsi::exec
/// \endcond

#endif
