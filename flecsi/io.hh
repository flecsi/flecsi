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

#ifdef DOXYGEN // these are implemented per-backend
typedef unspecified hdf5_region_t, launch_space_t;

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

template<bool = true>
void checkpoint_data(const std::string &,
  launch_space_t,
  const std::vector<hdf5_region_t> &,
  bool attach);
void recover_data(const std::string &,
  launch_space_t,
  const std::vector<hdf5_region_t> &,
  bool attach);

#ifdef DOXYGEN
struct io_interface_t {
  explicit io_interface_t(int num_files);
  void checkpoint_process_topology(const std::string &);
  void recover_process_topology(const std::string &);
};
#endif

} // namespace flecsi::io
