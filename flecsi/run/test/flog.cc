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
#include "flecsi/util/unit.hh"

using namespace flecsi;

int
log_driver() {
  UNIT {
    {
      std::vector<std::size_t> v;
      for(std::size_t i{0}; i < 10; ++i) {
        v.emplace_back(i);
      }
      flog(info) << log::to_string(v) << std::endl;
    }

    {
      std::vector<std::vector<std::size_t>> v;
      for(std::size_t i{0}; i < 10; ++i) {
        v.push_back({0, 1, 2});
      }
      flog(info) << log::to_string(v) << std::endl;
    }

    {
      std::map<std::size_t, std::size_t> m;
      for(std::size_t i{0}; i < 10; ++i) {
        m[i] = i;
      }
      flog(info) << log::to_string(m) << std::endl;
    }

    {
      std::map<std::size_t, std::vector<std::size_t>> m;
      for(std::size_t i{0}; i < 10; ++i) {
        m[i] = {0, 1, 2};
      }
      flog(info) << log::to_string(m) << std::endl;
    }
  };
} // flog

flecsi::unit::driver<log_driver> driver;
