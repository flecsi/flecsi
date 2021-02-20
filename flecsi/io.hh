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

#ifdef DOXYGEN // implemented per-backend
struct hdf5_t {
  static hdf5_t create(const std::string &);
  static hdf5_t open(const std::string &);
  hdf5_t(const char *, int num_files);
  /// Must not be called repeatedly.
  /// \return whether the file was successfully closed
  bool close();
  bool write_string(const std::string & group,
    const std::string & dataset,
    const std::string &);
  bool read_string(const std::string & group,
    const std::string & dataset,
    std::string &);
  bool create_dataset(const std::string &, hsize_t size);
};
#endif

void checkpoint_data(const std::string &, int files, bool attach = true);
void recover_data(const std::string &, int files, bool attach = true);

} // namespace flecsi::io
