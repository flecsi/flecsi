// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_BIND_PARAMETERS_HH
#define FLECSI_EXEC_BIND_PARAMETERS_HH

#include "flecsi/config.hh"
#include "flecsi/data/privilege.hh"
#include "flecsi/exec/launch.hh"
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
template<processor Proc>
struct bind_accessors {
protected:
  /// Send a raw field reference to a raw accessor.
  template<typename T, Privileges P>
  void visit(data::accessor<data::raw, T, P> &);
};
#endif

template<processor Proc>
struct bind_parameters : detail::bind_base<bind_parameters<Proc>>,
                         bind_accessors<Proc> {

  template<class A, class... Args>
  explicit bind_parameters(A & a, Args &&... args)
    : bind_accessors<Proc>(std::forward<Args>(args)...) {
    util::annotation::rguard<util::annotation::execute_bind_parameters> ann;
    std::apply([&](auto &... aa) { (visit(aa), ...); }, a);
  }

private:
  friend typename bind_parameters::bind_base;
  using bind_accessors<Proc>::visit; // for backend-specific stuff
  using bind_parameters::bind_base::visit;

  template<class P>
  std::enable_if_t<std::is_base_of_v<data::send_tag, P>> visit(P & p) {
    p.send(this->visitor());
  }

  template<typename T>
  static void visit(const data::detail::scalar_value<T> & s) {
    s.template copy<Proc>();
  }
};
/// \}
} // namespace exec
} // namespace flecsi
/// \endcond

#endif // FLECSI_EXEC_BIND_PARAMETERS_HH
