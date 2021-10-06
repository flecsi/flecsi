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

//----------------------------------------------------------------------------//
// Pickup Kokkos defines if enabled.
//----------------------------------------------------------------------------//

#if defined(FLECSI_ENABLE_KOKKOS)
#include <Kokkos_Core.hpp>

#define FLECSI_TARGET KOKKOS_FUNCTION
#define FLECSI_INLINE_TARGET KOKKOS_INLINE_FUNCTION

#endif // FLECSI_ENABLE_KOKKOS

//----------------------------------------------------------------------------//
// Defaults.
//----------------------------------------------------------------------------//

#if !defined(FLECSI_TARGET)
#define FLECSI_TARGET
#endif

#if !defined(FLECSI_INLINE_TARGET)
#define FLECSI_INLINE_TARGET inline
#endif

#if defined(__HIPCC__)
#include "hip/hip_runtime.h"
#define HIP_ASSERT(status) assert((status) == hipSuccess)
#endif
