// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_MPI_FUTURE_HH
#define FLECSI_EXEC_MPI_FUTURE_HH

#include "flecsi/exec/launch.hh"
#include "flecsi/util/function_traits.hh"
#include "flecsi/util/mpi.hh"

#include <future>

namespace flecsi {

template<typename R>
struct future<R> {
  void wait() {
    fut.wait();
  }

  R get(bool = false) {
    return fut.get();
  }

  ~future() {
    // 'fut' might not have a valid shared state due to the "default"
    // construction used for converting future<R, index> to future<R, single>
    // in the generic code.
    if(fut.valid())
      fut.wait();
  }

  // Note: flecsi::future needs to be copyable and passed by value to user tasks
  // and .wait()/.get() called. See future.cc unit test for such a use case.
  std::shared_future<R> fut;
};

template<>
struct future<void> {
  void wait() {}
  void get(bool = false) {}
};

template<typename F>
auto
async(F && f) {
  using R = typename util::function_traits<F>::return_type;
  return future<R>{std::async(std::forward<F>(f)).share()};
}

template<typename T>
auto
make_ready_future(T t) {
  std::promise<T> promise;
  promise.set_value(t);
  return promise.get_future();
}

template<typename R>
struct future<R, exec::launch_type_t::index> {
  using result_type = std::conditional_t<std::is_same_v<R, bool>, char, R>;

  explicit future(R result) : result(result) {
    results.resize(size());

    // Initiate MPI_Iallgather
    util::mpi::test(MPI_Iallgather(&result,
      1,
      flecsi::util::mpi::type<result_type>(),
      results.data(),
      1,
      flecsi::util::mpi::type<result_type>(),
      MPI_COMM_WORLD,
      &request));
  }

  void wait(bool = false) {
    util::mpi::test(MPI_Wait(&request, MPI_STATUS_IGNORE));
  }

  R get(Color index = 0, bool = false) {
    wait();
    return results.at(index);
  }

  Color size() const {
    return run::context::instance().processes();
  }

  result_type result;

  // Handling the case that the future<> is destroyed without wait()/get()
  // (thus MPI_Wait()) being called.
  ~future() {
    wait();
  }

private:
  MPI_Request request{};
  std::vector<result_type> results;
};

template<>
struct future<void, exec::launch_type_t::index> {
  void wait(bool = false) {}
  void get(Color = 0, bool = false) {}
  Color size() const {
    return run::context::instance().processes();
  }
};
} // namespace flecsi

#endif
