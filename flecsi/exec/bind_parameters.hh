// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_BIND_PARAMETERS_HH
#define FLECSI_EXEC_BIND_PARAMETERS_HH

#include "flecsi/config.hh"
#include "flecsi/data/privilege.hh"
#include "flecsi/flog.hh"
#include "flecsi/util/annotation.hh"
#include "flecsi/util/demangle.hh"

#include <functional>
#include <tuple>

// bind_accessors is implemented per backend:
#if FLECSI_BACKEND == FLECSI_BACKEND_legion
#include "flecsi/exec/leg/bind_accessors.hh"
#elif FLECSI_BACKEND == FLECSI_BACKEND_mpi
#error "The MPI backend has no need for bind_accessors"
#elif FLECSI_BACKEND == FLECSI_BACKEND_hpx
#include "flecsi/exec/hpx/bind_accessors.hh"
#endif

/// \cond core
namespace flecsi {
namespace exec {
/// \addtogroup execution
/// \{

#ifdef DOXYGEN // implemented per-backend
/// Handling for low-level special task parameters/arguments.
/// The exact member function signatures may vary between backends.
/// \note No constructors are specified.
template<task_processor_type_t Proc>
struct bind_accessors {
protected:
  /// Send a raw field reference to a raw accessor.
  template<typename T, Privileges P>
  void visit(data::accessor<data::raw, T, P> &);
};
#endif

template<task_processor_type_t ProcessorType>
struct bind_parameters : bind_accessors<ProcessorType> {

  template<class A, class... Args>
  explicit bind_parameters(A & a, Args &&... args)
    : bind_accessors<ProcessorType>(std::forward<Args>(args)...) {
    util::annotation::rguard<util::annotation::execute_bind_parameters> ann;
    std::apply([&](auto &... aa) { (visit(aa), ...); }, a);
  }

protected:
  auto visitor() {
    return
      [&](auto & p, auto &&) { visit(p); }; // Clang 8.0.1 deems 'this' unused
  }

  using bind_accessors<ProcessorType>::visit; // for backend-specific stuff

  template<class P>
  std::enable_if_t<std::is_base_of_v<data::send_tag, P>> visit(P & p) {
    p.send(visitor());
  }

  // Note: due to how visitor() is implemented above the first parameter can not
  // be 'const &' here, otherwise template/overload resolution fails (silently).
  template<typename T>
  static void visit(data::detail::scalar_value<T> & s) {
    s.template copy<ProcessorType>();
  }

  /*--------------------------------------------------------------------------*
    Non-FleCSI Data Types
   *--------------------------------------------------------------------------*/

  template<typename D>
  static typename std::enable_if_t<!std::is_base_of_v<data::bind_tag, D>> visit(
    D &) {
    {
    }
  }
};
/// \}
} // namespace exec
} // namespace flecsi
/// \endcond

#endif // FLECSI_EXEC_BIND_PARAMETERS_HH
