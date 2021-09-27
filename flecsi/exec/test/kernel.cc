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

void
modify(intN::accessor<wo> a) {
  forall(i, util::span(*a), "modify") {
    i = 3;
  };
}

int
check(intN::accessor<ro> a) {
  UNIT {
    for(auto i : util::span(*a)) {
      EXPECT_EQ(i, 3);
    }
  };
}

void
modify_policy(intN::accessor<wo> a) {
  forall(j, flecsi::exec::range_policy(util::span(*a)), "modify_policy") {
    j = 3;
  };
}

int
check_policy(intN::accessor<ro> a) {
  UNIT {
    for(auto j : util::span(*a)) {
      EXPECT_EQ(j, 3);
    }
  };
}

void
modify_bound(intN::accessor<wo> a) {
  forall(i, (flecsi::exec::range_bound{util::span(*a), 0, 10}), "modify_policy") {
    i = 3;
  };
}

int
check_bound(intN::accessor<ro> a) {
  UNIT {
    for(auto i : util::span(*a)) {
      EXPECT_EQ(i, 3);
    }
  };
}



int
reduce_vec(intN::accessor<ro> a) {
  UNIT {
    size_t res =
      reduceall(i, up, util::span(*a), exec::fold::sum, size_t, "reduce") {
      up += i;
    };
    EXPECT_EQ(res, 3 * a.get().size());
  };
}

int
kernel_driver() {
  UNIT {
    const auto ar = array_field(process_topology);
    execute<modify, default_accelerator>(ar);
    EXPECT_EQ(test<check>(ar), 0);
    execute<modify_policy, default_accelerator>(ar);
    EXPECT_EQ(test<check_policy>(ar), 0);
    execute<modify_bound, default_accelerator>(ar);
    EXPECT_EQ(test<check_bound>(ar), 0);
    EXPECT_EQ((test<reduce_vec, default_accelerator>(ar)), 0);
  };
} // kernel_driver

flecsi::unit::driver<kernel_driver> driver;
