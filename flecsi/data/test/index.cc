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
#include <flecsi/data/data.h>
#include <flecsi/execution/execution.h>
#include <flecsi/utils/demangle.h>
#include <flecsi/utils/ftest.h>

using namespace flecsi;

flecsi_add_index_field("test", "value", double, 2);
inline auto fh = flecsi_index_field_instance("test", "value", double, 0);

template<size_t PRIVILEGES>
using accessor =
  flecsi::data::index_accessor_u<double, privilege_pack_u<PRIVILEGES>::value>;

namespace index_test {

void
assign(accessor<rw> ia) {
  ia = flecsi_color();
} // assign

flecsi_register_task(assign, index_test, loc, index);

int
check(accessor<ro> ia) {

  FTEST();

  ASSERT_EQ(ia, flecsi_color());

  return 0;
} // print

flecsi_register_task(check, index_test, loc, index);

} // namespace index_test

int
index_topology(int argc, char ** argv) {

  flecsi_execute_task(assign, index_test, index, fh);
  flecsi_execute_task(check, index_test, index, fh);

  return 0;
} // index

ftest_register_test(index_topology);