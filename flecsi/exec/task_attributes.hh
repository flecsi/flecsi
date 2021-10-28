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

#include "flecsi/util/bitutils.hh"

namespace flecsi {

using TaskAttributes = unsigned;

/*!
  The task_attributes_mask_t type allows conversion from an eunumeration
  to a bit mask.

  @note This enumeration is not scoped so that users can create
        masks as in:
        @code
        task_attributes_mask_t m(inner | idempotent);
        @endcode
 */

enum task_attributes_mask_t : TaskAttributes {
  leaf = 0x01,
  inner = 0x02,
  idempotent = 0x04,
  loc = 0x08,
  toc = 0x10,
  mpi = 0x20
}; // task_attributes_mask_t

constexpr auto default_accelerator =
#if defined(__NVCC__) || defined(__CUDACC__)
  toc
#else
  loc
#endif
  ;

namespace exec {

/*!
  Enumeration of task types.
 */

enum class task_type_t : size_t { leaf, inner, idempotent }; // task_type_t

/*!
  Enumeration of processor types.
 */

enum class task_processor_type_t : size_t {
  loc,
  toc,
  mpi
}; // task_processor_type_t

// Bits for representing task attributes
constexpr size_t task_attributes_bits = 8;
constexpr size_t task_type_bits = 3;

constexpr auto
as_mask(task_type_t t) {
  return static_cast<task_attributes_mask_t>(
    1 << static_cast<TaskAttributes>(t));
}
constexpr auto
as_mask(task_processor_type_t t) {
  return static_cast<task_attributes_mask_t>(
    1 << task_type_bits + static_cast<TaskAttributes>(t));
}

inline task_type_t
mask_to_task_type(TaskAttributes mask) {
  return static_cast<task_type_t>(util::bit_width(mask) - 1);
} // mask_to_task_type

constexpr task_processor_type_t
mask_to_processor_type(TaskAttributes mask) {
  return static_cast<task_processor_type_t>(
    util::bit_width(mask) - task_type_bits - 1);
} // mask_to_processor_type

} // namespace exec
} // namespace flecsi
