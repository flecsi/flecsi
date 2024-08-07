// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_TARGET_HH
#define FLECSI_UTIL_TARGET_HH

#include "flecsi/config.hh"

/// \addtogroup utils
/// \{

//----------------------------------------------------------------------------//
// Pickup Kokkos defines
//----------------------------------------------------------------------------//

#include <Kokkos_Core.hpp>

/// Make a function available on a GPU.
/// Use before a return type or a lambda's parameter list.
/// \warning Many implementations impose [severe
///   restrictions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#language-restrictions)
///   on such functions.  Most prominent is that they can call only other
///   functions so annotated, preventing the use of much or all of many
///   libraries (including the standard library).  FleCSI documents certain
///   classes and functions as being "supported for GPU execution".
#define FLECSI_TARGET KOKKOS_FUNCTION
/// Make a function inline and available on a GPU.
/// \see FLECSI_TARGET
#define FLECSI_INLINE_TARGET KOKKOS_INLINE_FUNCTION

#ifdef DOXYGEN
/// Specify if the code is being built on GPU.
#define FLECSI_DEVICE_CODE
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define FLECSI_DEVICE_CODE
#endif

/// \}

#endif
