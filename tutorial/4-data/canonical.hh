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
#pragma once

#include <flecsi/flog.hh>
#include <flecsi/topo/unstructured/interface.hh>

using namespace flecsi;

struct canon : topo::specialization<topo::unstructured, canon> {
  enum index_space { vertices, cells };
  using index_spaces = has<cells, vertices>;
  // using connectivities = util::types<entity<cells, has<vertices>>>;
  using connectivities = util::types<>;
  enum entity_list {};
  using entity_lists = list<>;

  static coloring color(std::string const &) {
    flog(info) << "invoking coloring" << std::endl;
    return {{MPI_COMM_WORLD,
      0, /* color */
      1, /* colors */
      {4, 2}, /* index allocs */
      {{/* cells */
         {0, 1, 2, 3},
         {0, 1, 2, 3},
         {0, 1, 2, 3},
         {},
         {}},
        {/* vertices */
          {0, 1},
          {0, 1},
          {0, 1},
          {},
          {}}},
      {{}},
      {{}}}};
  }
};
