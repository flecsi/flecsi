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

#define __FLECSI_PRIVATE__

#include <iostream>
#include <string>

#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/flog.hh"
#include "flecsi/io.hh"
#include "flecsi/topo/canonical/interface.hh"
#include "flecsi/topo/core.hh"
#include "flecsi/util/constant.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;

struct canon : topo::specialization<topo::canonical, canon> {
  enum index_space { vertices, cells };
  using index_spaces = has<cells, vertices>;
  using connectivities = util::types<>;

  static coloring color(std::string const &) {
    flog(info) << "invoking coloring" << std::endl;
    return {4, {32, 45}, {}};
  }
};

canon::slot canonical;
canon::cslot coloring;

template<typename T>
using dense_field = field<T, data::dense>;

const dense_field<double>::definition<canon, canon::cells> cell_field_1,
  cell_field_2;
const dense_field<double>::definition<canon, canon::vertices> vertex_field_1;

void
init(canon::accessor<wo> t,
  dense_field<double>::accessor<wo> f1,
  dense_field<double>::accessor<wo> f2,
  dense_field<double>::accessor<wo> vf1) {
  for(const auto c : t.entities<canon::cells>()) {
    f1[c] = 100. * color() + double(c);
    f2[c] = 100. * color() + 20. + double(c);
  } // for
  for(const auto v : t.entities<canon::vertices>()) {
    vf1[v] = 100. * color() + 40. + double(v);
  } // for
} // init

void
clear(canon::accessor<wo> t,
  dense_field<double>::accessor<wo> f1,
  dense_field<double>::accessor<wo> f2,
  dense_field<double>::accessor<wo> vf1) {
  for(const auto c : t.entities<canon::cells>()) {
    f1[c] = 0.;
    f2[c] = 0.;
  } // for
  for(const auto v : t.entities<canon::vertices>()) {
    vf1[v] = 0.;
  } // for
} // clear

int
check(canon::accessor<ro> t,
  dense_field<double>::accessor<ro> f1,
  dense_field<double>::accessor<ro> f2,
  dense_field<double>::accessor<ro> vf1) {
  UNIT {
    for(auto c : t.entities<canon::cells>()) {
      auto f1exp = 100. * color() + double(c);
      auto f2exp = 100. * color() + 20. + double(c);
      ASSERT_EQ(f1[c], f1exp);
      ASSERT_EQ(f2[c], f2exp);
    } // for
    for(const auto v : t.entities<canon::vertices>()) {
      auto vf1exp = 100. * color() + 40. + double(v);
      ASSERT_EQ(vf1[v], vf1exp);
    } // for
  };
} // check

int
restart_driver() {
  UNIT {
    coloring.allocate("test.txt");
    canonical.allocate(coloring.get());

    auto cf1 = cell_field_1(canonical);
    auto cf2 = cell_field_2(canonical);
    auto vf1 = vertex_field_1(canonical);
    execute<init>(canonical, cf1, cf2, vf1);

    int num_files = 4;
    io::io_interface iif{num_files};
    // TODO:  make this registration automatic, not manual
    iif.add_region<canon, canon::cells>(canonical);
    iif.add_region<canon, canon::vertices>(canonical);
    iif.checkpoint_all_fields("hdf5_restart.dat");

    execute<clear>(canonical, cf1, cf2, vf1);
    iif.recover_all_fields("hdf5_restart.dat");

    EXPECT_EQ(test<check>(canonical, cf1, cf2, vf1), 0);
  };

  return 0;
}

flecsi::unit::driver<restart_driver> driver;
