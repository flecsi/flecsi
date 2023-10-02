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
  future() : future(make()) {}
  future(const future &) = default;
  future & operator=(const future &) & = default;

  void wait() {
    get();
  }

  R & get(bool = false) {
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
  void wait() {}
  void get(bool = false) {}
};

template<typename R>
struct future<R, exec::launch_type_t::index> {
  using result_type = std::conditional_t<std::is_same_v<R, bool>, char, R>;

  explicit future(R r) : result(std::move(r)) {
    results.resize(size());

    // Initiate MPI_Iallgather
    util::mpi::test(MPI_Iallgather(&result,
      1,
      flecsi::util::mpi::type<result_type>(),
      results.data(),
      1,
      flecsi::util::mpi::type<result_type>(),
      MPI_COMM_WORLD,
      request()));
  }
  future(future &&) = delete;

  void wait(bool = false) {
    this->request = {}; // this-> avoids Clang bug #62818
  }

  std::conditional_t<std::is_same_v<R, bool>, R, R &> get(Color index = 0,
    bool = false) {
    wait();
    return results.at(index);
  }

  Color size() const {
    return run::context::instance().processes();
  }

  result_type result;

private:
  std::vector<result_type> results;
  util::mpi::auto_requests request;
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
