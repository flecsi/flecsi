// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_FIELD_INFO_HH
#define FLECSI_DATA_FIELD_INFO_HH

#include "flecsi-config.h"
#include "flecsi/util/common.hh"
#include "flecsi/util/types.hh"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

/// \cond core
namespace flecsi {
namespace data {
/// \addtogroup data
/// \{

/*!
  The field_info_t type provides a structure for capturing runtime field
  information.
 */

struct field_info_t {
  field_id_t fid;
  std::size_t type_size;
  std::string name;
}; // struct field_info_t

using fields = std::vector<std::shared_ptr<field_info_t>>;

/// \}
} // namespace data

/// \addtogroup data
/// \{

//----------------------------------------------------------------------------//
// This value is used by the Legion backend to automatically
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

using TopologyType = std::size_t; // for field registration

/// \}
} // namespace flecsi
/// \endcond

#endif
