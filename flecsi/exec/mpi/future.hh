/*
    @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
   /@@/////  /@@          @@////@@ @@////// /@@
   /@@       /@@  @@@@@  @@    // /@@       /@@
   /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
   /@@////   /@@/@@@@@@@/@@       ////////@@/@@
   /@@       /@@/@@//// //@@    @@       /@@/@@
   /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
   //       ///  //////   //////  ////////  //

   Copyright (c) 2016, Los Alamos National Security, LLC
   All rights reserved.
                                                                              */
#pragma once

/*! @file */

#include "flecsi/exec/launch.hh"
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

  // FIXME: do we need to call fut.wait() in the destructor?

  // FIXME: can we make fut private?
//private:
  // flecsi::future needs to be copyable
  std::shared_future<R> fut;
};

template<>
struct future<void> {
  void wait() {}
  void get(bool = false) {}
};

template<typename R>
struct future<R, exec::launch_type_t::index> {
  // FIXME: what does copying mean?
  explicit future(R result) : result(result) {
    results.resize(size());

    // Initiate MPI_Iallgather
    util::mpi::test(MPI_Iallgather(&result,
      1,
      flecsi::util::mpi::type<R>(),
      results.data(),
      1,
      flecsi::util::mpi::type<R>(),
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

  R result;

  // Handling the case that the future<> is destroyed without wait()/get()
  // (thus MPI_Wait()) being called.
  ~future() {
    wait();
  }

private:
  MPI_Request request{};
  std::vector<R> results;
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
