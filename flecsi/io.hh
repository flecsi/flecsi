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

/*!  @file */

#include "io/backend.hh"

namespace flecsi::io {

// currently these methods don't do anything unless topologies and
// index spaces have been registered manually first
// TODO:  add automatic registration
void
checkpoint_all_fields(const std::string & file_name, int num_files) {
  io_interface(num_files).checkpoint_all_fields(file_name);
}
void
recover_all_fields(const std::string & file_name, int num_files) {
  io_interface(num_files).recover_all_fields(file_name);
}

} // namespace flecsi::io
