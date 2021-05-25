// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_LOCAL_BIND_PROLOGUE_HH
#define FLECSI_EXEC_LOCAL_BIND_PROLOGUE_HH

#include <flecsi-config.h>

#include <hpx/modules/collectives.hpp>

#include "flecsi/data/privilege.hh"
#include "flecsi/flog.hh"
#include "flecsi/util/annotation.hh"
#include "flecsi/util/demangle.hh"

#include <functional>
#include <tuple>

namespace flecsi {
inline flog::devel_tag bind_parameters_tag("bind_parameters");
inline flog::devel_tag bind_accessors_tag("bind_accessors");
} // namespace flecsi

// task_prologue is implemented per backend:
#if FLECSI_BACKEND == FLECSI_BACKEND_legion
#include "flecsi/exec/leg/bind_accessors.hh"
#elif FLECSI_BACKEND == FLECSI_BACKEND_mpi
#error "The MPI backend has no need for bind_accessors"
#elif FLECSI_BACKEND == FLECSI_BACKEND_hpx
#include "flecsi/exec/hpx/bind_accessors.hh"
#endif

namespace flecsi {
namespace exec {

#ifdef DOXYGEN // implemented per-backend
/// Handling for low-level special task parameters/arguments.
/// The exact member function signatures may vary between backends.
template<task_processor_type_t Proc>
struct bind_accessors {
protected:
  /// \note No constructors are specified.

  /// Send a raw field reference to a raw accessor.
  template<typename T, Privileges P>
  void visit(data::accessor<data::raw, T, P> &);
};
#endif

template<task_processor_type_t ProcessorType>
struct bind_parameters : bind_accessors<ProcessorType> {

  template<class A, class... Args>
  bind_parameters(A & a, Args &&... args)
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
    if constexpr(ProcessorType == exec::task_processor_type_t::toc) {
#if defined(__NVCC__) || defined(__CUDACC__)
      cudaMemcpy(s.host, s.device, sizeof(T), cudaMemcpyDeviceToHost);
      return;
#elif defined(__HIPCC__)
      HIP_ASSERT(hipMemcpy(s.host, s.device, sizeof(T), hipMemcpyDeviceToHost));
      return;
#else
      flog_assert(false, "Cuda should be enabled when using toc task");
#endif
    }
    *s.host = *s.device;
  }

  /*--------------------------------------------------------------------------*
    Non-FleCSI Data Types
   *--------------------------------------------------------------------------*/

  template<typename D>
  static typename std::enable_if_t<!std::is_base_of_v<data::bind_tag, D>> visit(
    D &) {
    {
      flog::devel_guard guard(bind_parameters);
      flog_devel(info) << "No setup for parameter of type " << util::type<D>()
                       << std::endl;
    }
  }
};
} // namespace exec
} // namespace flecsi

#endif // FLECSI_EXEC_LOCAL_BIND_PROLOGUE_HH
