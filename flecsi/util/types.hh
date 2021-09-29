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

/*! @file */

#include <flecsi-config.h>

#include <cstddef>

#if FLECSI_RUNTIME_MODEL == FLECSI_RUNTIME_MODEL_legion
#include <legion.h>
#endif

namespace flecsi {
#if FLECSI_RUNTIME_MODEL == FLECSI_RUNTIME_MODEL_legion
using field_id_t = Legion::FieldID;
using Legion::Color;
#else
using field_id_t = std::size_t;
using Color = unsigned; // MPI uses just int
#endif
} // namespace flecsi
