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

namespace flecsi {

template<typename R>
struct future<R> {
  void wait() {}
  R get(bool = false) {
    return result_;
  }

  R result_;
};

template<>
struct future<void> {
  void wait() {}
  void get(bool = false) {}
};

template<typename R>
struct future<R, exec::launch_type_t::index> {
  void wait(bool = false) {}
  R get(std::size_t index = 0, bool = false);
  std::size_t size() const {
    return run::context::instance().processes();
  }
};

} // namespace flecsi
