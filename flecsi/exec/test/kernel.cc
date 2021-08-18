/*
    @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
   /@@/////  /@@          @@////@@ @@////// /@@
   /@@       /@@  @@@@@  @@    // /@@       /@@
   /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
   /@@////   /@@/@@@@@@@/@@       ////////@@/@@
   /@@       /@@/@@//// //@@    @@       /@@/@@
   /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
   //       ///  //////   //////  ////////  //

   Copyright (c) 2016, Triad National Security, LLC
   All rights reserved.
                                                                              */

#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;

void
modify(std::vector<size_t> & v) {
  flecsi::util::span<size_t> s = v;
  forall(i, s, "modify") {
    i = 3;
  };
}

void
check(std::vector<size_t> & v) {
  forall(i, v, "check") {
    assert(i == 3);
  };
}

void
reduce_vec(std::vector<size_t> & v) {
  size_t res = reduceall(i, up, v, exec::fold::sum, size_t, "reduce") {
    up += i;
  };

  assert(3 * v.size() == res);
};

int
kernel_driver() {
  UNIT {

    std::vector<size_t> vec;
    vec.resize(run::context::instance().processes());
    for(size_t i = 0; i < 10; i++)
      vec.push_back(i);

    execute<modify, mpi>(vec);
    execute<check, mpi>(vec);
    execute<reduce_vec, mpi>(vec);
  };
} // kernel_driver

flecsi::unit::driver<kernel_driver> driver;
