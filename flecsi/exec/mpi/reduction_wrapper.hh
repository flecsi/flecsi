// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_MPI_REDUCTION_WRAPPER_HH
#define FLECSI_EXEC_MPI_REDUCTION_WRAPPER_HH

#include "flecsi/exec/fold.hh"
#include "flecsi/run/backend.hh"
#include "flecsi/util/demangle.hh"
#include "flecsi/util/mpi.hh"
#include <flecsi/flog.hh>

namespace flecsi {

inline flog::devel_tag reduction_wrapper_tag("reduction_wrapper");

namespace exec {
namespace fold {
/// \addtogroup mpi-execution
/// \{

template<class R, class T, class = void>
struct wrap {
private:
  static void apply(void * in, void * inout, int * len, MPI_Datatype *) {
    const auto rd = static_cast<const T *>(in);
    const auto rw = static_cast<T *>(inout);

    for(int i = 0, n = *len; i < n; ++i) {
      rw[i] = R::combine(rw[i], rd[i]);
    } // for
  }

  static void init() {
    {
      flog::devel_guard guard(reduction_wrapper_tag);
      flog_devel(info) << "registering reduction operation " << util::type<R>()
                       << " for " << util::type<T>() << std::endl;
    } // scope

    // Create the operator and register it with the runtime
    util::mpi::test(MPI_Op_create(apply, true, &op));
  }

public:
  // NB: The real initialization is in the callback.
  static inline MPI_Op op = (run::context::register_init(init), MPI_Op());
};

template<class>
MPI_Op redop() = delete;
template<>
inline MPI_Op
redop<min>() {
  return MPI_MIN;
}
template<>
inline MPI_Op
redop<max>() {
  return MPI_MAX;
}
template<>
inline MPI_Op
redop<sum>() {
  return MPI_SUM;
}
template<>
inline MPI_Op
redop<product>() {
  return MPI_PROD;
}

template<class, class>
struct ordered {};
template<class T>
struct ordered<min, std::complex<T>>; // undefined
template<class T>
struct ordered<max, std::complex<T>>; // undefined
template<class R>
struct ordered<R, bool>; // undefined

template<class R, class T>
struct wrap<R,
  T,
  decltype(void((redop<R>(), util::mpi::static_type<T>(), ordered<R, T>())))> {
  static inline const MPI_Op & op = redop<R>();
};

/// \}
} // namespace fold
} // namespace exec
} // namespace flecsi

#endif
