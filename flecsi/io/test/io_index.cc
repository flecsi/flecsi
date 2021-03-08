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

#include <cassert>
#include <string>

#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/io.hh"
#include "flecsi/util/demangle.hh"
#include "flecsi/util/unit.hh"

#include <mpi.h>

using namespace flecsi;
using namespace flecsi::data;

typedef field<double, single> double_field_t;
const double_field_t::definition<topo::index> test_value_1, test_value_2,
  test_value_3;

void
assign(double_field_t::accessor<rw> ia) {
  ia = color();
} // assign

void
reset_zero(double_field_t::accessor<rw> ia) {
  ia = -1;
} // assign

int
check(double_field_t::accessor<ro> ia) {
  UNIT { ASSERT_EQ(ia, color()); };
} // print

int
index_driver() {
  UNIT {
    const auto fh1 = test_value_1(process_topology);
    const auto fh2 = test_value_2(process_topology);
    const auto fh3 = test_value_3(process_topology);
    execute<assign>(fh1);
    execute<assign>(fh2);
    execute<assign>(fh3);

    auto & flecsi_context = run::context::instance();
    // TODO:  support N-to-M
    int num_files = flecsi_context.processes();
    const std::string outfile{"io_index.dat"};

    io::checkpoint_data(outfile, num_files);

    execute<reset_zero>(fh1);
    execute<reset_zero>(fh2);
    execute<reset_zero>(fh3);

    int num_ranks = flecsi_context.processes();
    int my_rank = flecsi_context.process();
    assert(num_ranks % num_files == 0);
    int num_ranks_per_file = num_ranks / num_files;
    if(my_rank % num_ranks_per_file == 0) {
      io::hdf5 checkpoint_file =
        io::hdf5::open(outfile + std::to_string(my_rank / num_ranks_per_file));

      checkpoint_file.write_string("control", "ds2", "test string 2");

      std::string str3;
      checkpoint_file.read_string("control", "ds2", str3);
      // printf("str 3 %s\n", str3.c_str());
      checkpoint_file.close();
    }

    io::recover_data(outfile, processes());

    EXPECT_EQ(test<check>(fh1), 0);
    EXPECT_EQ(test<check>(fh2), 0);
    EXPECT_EQ(test<check>(fh3), 0);
  };
} // index_driver

flecsi::unit::driver<index_driver> driver;
