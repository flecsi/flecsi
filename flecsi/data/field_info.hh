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

#include "flecsi/util/common.hh"
#include "flecsi/util/types.hh"

#include <cstddef>
#include <string>
#include <vector>

namespace flecsi {
namespace data {

/*!
  The field_info_t type provides a structure for capturing runtime field
  information.
 */

struct field_info_t {
  field_id_t fid;
  std::size_t type_size;
  std::string name;
}; // struct field_info_t

using fields = std::vector<const field_info_t *>;

} // namespace data

//----------------------------------------------------------------------------//
// This value is used by the Legion runtime backend to automatically
// assign task and field ids. The current maximum value that is allowed
// in legion_config.h is 1<<20.
//
// We are reserving 4096 places for internal use.
//----------------------------------------------------------------------------//

#if !defined(FLECSI_GENERATED_ID_MAX)
// 1044480 = (1<<20) - 4096
#define FLECSI_GENERATED_ID_MAX 1044480
#endif

/*!
  Unique counter for field ids.
 */
inline util::counter<field_id_t(FLECSI_GENERATED_ID_MAX)> fid_counter(0);

using TopologyType = std::size_t;

} // namespace flecsi
