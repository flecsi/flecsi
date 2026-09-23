// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_SET_TYPES_HH
#define FLECSI_TOPO_SET_TYPES_HH

#include "flecsi/config.hh"

#include <cstddef>

#if FLECSI_BACKEND == FLECSI_BACKEND_legion
#include <legion.h>
#endif

namespace flecsi {
#if !defined(DOXYGEN) && FLECSI_BACKEND == FLECSI_BACKEND_legion
using field_id_t = Legion::FieldID;
using Legion::Color;
#else
using field_id_t = unsigned;
/// The integer type used to count topology colors, task instances, or
/// processes.
using Color = unsigned; // MPI uses just int
#endif
} // namespace flecsi

#endif
