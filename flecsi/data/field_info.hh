// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_FIELD_INFO_HH
#define FLECSI_DATA_FIELD_INFO_HH

#include "flecsi/config.hh"
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

#if !defined(FLECSI_GENERATED_ID_MAX)
// Reserve a few before those reserved by Legion:
#define FLECSI_GENERATED_ID_MAX ((1 << 20) - (1 << 12))
#endif

/*!
  Unique counter for field ids.  \ns.
 */
inline util::counter<field_id_t(FLECSI_GENERATED_ID_MAX)> fid_counter(0);

/// \}
} // namespace flecsi
/// \endcond

#endif
