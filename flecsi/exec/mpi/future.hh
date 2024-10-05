// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_MPI_FUTURE_HH
#define FLECSI_EXEC_MPI_FUTURE_HH

#include "flecsi/exec/launch.hh"
#include "flecsi/util/function_traits.hh"
#include "flecsi/util/mpi.hh"

namespace flecsi {

template<typename R>
struct future<R> {
  // Provide value semantics:
  future() = default; // for partially constructed task parameters
  future(const future &) = default;
  future & operator=(const future &) & = default;

  void wait() {
    get();
  }

  [[nodiscard]] R & get(bool = false) {
    return (*fut)();
  }

  auto * operator->() {
    return fut.get();
  }

  // To avoid competing with the copy constructor:
  template<class... PP>
  static future make(PP &&... pp) {
    return future(
      std::make_shared<util::mpi::future<R>>(std::forward<PP>(pp)...));
  }

private:
  using pointer = std::shared_ptr<util::mpi::future<R>>;

  future(pointer p) : fut(std::move(p)) {}

  // Note: flecsi::future needs to be copyable and passed by value to user tasks
  // and .wait()/.get() called. See future.cc unit test for such a use case.
  pointer fut;
};

template<>
struct future<void> {
  void wait() {
    get();
  }
  void get(bool = false) {
    util::mpi::test(MPI_Barrier(MPI_COMM_WORLD));
  }
};

template<typename R>
struct future<R, exec::launch_type_t::index> {
  explicit future(R r) : result(std::move(r)) {}

  void wait(bool = false) {
    util::mpi::test(MPI_Barrier(MPI_COMM_WORLD));
  }

  [[nodiscard]] R get(Color index = 0, bool = false) {
    const auto b = [&](R & r) {
      util::mpi::test(
        MPI_Bcast(&r, 1, util::mpi::type<R>(), index, MPI_COMM_WORLD));
    };
    if(run::context::instance().process() == index) {
      b(result);
      return result;
    }
    R ret;
    b(ret);
    return ret;
  }

  Color size() const {
    return run::context::instance().processes();
  }

  R result;
};

template<>
struct future<void, exec::launch_type_t::index> {
  void wait(bool = false) {
    util::mpi::test(MPI_Barrier(MPI_COMM_WORLD));
  }
  void get(Color = 0, bool = false) {
    wait();
  }
  Color size() const {
    return run::context::instance().processes();
  }
};
} // namespace flecsi

#endif
