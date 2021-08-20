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
using namespace flecsi::data;

using intN = field<std::array<size_t, 10>, single>;
const intN::definition<topo::index> array_field;

int
modify(intN::accessor<wo> a) {
  UNIT {
    forall(i, util::span(*a), "modify") {
      i = 3;
    };
  };
}

int
check(intN::accessor<wo> a) {
  UNIT {
    forall(i, util::span(*a), "check") {
      assert(i == 3);
    };
  };
}

int
reduce_vec(intN::accessor<wo> a) {
  UNIT {
    size_t res =
      reduceall(i, up, util::span(*a), exec::fold::sum, size_t, "reduce") {
      up += i;
    };

    assert(3 * a.get().size() == res);
  };
}

int
kernel_driver() {
  UNIT {

    const auto ar = array_field(process_topology);
    test<modify, default_accelerator>(ar);
    test<check, default_accelerator>(ar);
    test<reduce_vec, default_accelerator>(ar);
  };
} // kernel_driver

flecsi::unit::driver<kernel_driver> driver;
