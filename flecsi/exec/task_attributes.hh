// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_TASK_ATTRIBUTES_HH
#define FLECSI_EXEC_TASK_ATTRIBUTES_HH

#include "flecsi/config.hh"
#include "flecsi/util/bitutils.hh"

namespace flecsi {
/// \addtogroup execution
/// \{

using TaskAttributes = unsigned;

/*!
  Task attribute flags.
 */

enum task_attributes_mask_t : TaskAttributes {
  leaf = 0x01,
  inner = 0x02,
  idempotent = 0x04, ///< Task may be replicated to reduce communication.
  loc = 0x08, ///< Run on a Latency-Optimized Core (a CPU).
  /// Run on a Throughput-Optimized Core (a GPU).
  ///
  /// \warning MPI backend: Running one process per node likely
  ///          leads to poor performance.
  toc = 0x10,
  /// Run as an OpenMP task
  ///
  /// \note Legion backend: Can improve OpenMP task execution, since Legion
  ///       knows to assign an entire node to such a task
  ///
  /// \warning MPI backend: Running one process per core likely
  ///          leads to poor performance.
  omp = 0x20,
  /// Run simultaneously on all processes with the obvious color mapping;
  /// allow MPI communication among point tasks, at the cost of significant
  /// startup overhead.
  mpi = 0x40
}; // task_attributes_mask_t

/// \c toc if support for it is available, otherwise \c loc.
inline constexpr auto default_accelerator =
#if defined(__NVCC__) || defined(__CUDACC__) || defined(__HIPCC__)
  toc
#elif defined(REALM_USE_OPENMP)
  omp
#else
  loc
#endif
  ;

/// \}

/// \cond core
namespace exec {
/// \addtogroup execution
/// \{

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
  omp,
  mpi
}; // task_processor_type_t

// Bits for representing task attributes
inline constexpr size_t task_attributes_bits = 8, task_type_bits = 3;

constexpr auto
as_mask(task_type_t t) {
  return static_cast<task_attributes_mask_t>(
    1 << static_cast<TaskAttributes>(t));
}
constexpr auto
as_mask(task_processor_type_t t) {
  return static_cast<task_attributes_mask_t>(
    1 << (task_type_bits + static_cast<TaskAttributes>(t)));
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

/// \}
} // namespace exec
  /// \endcond
} // namespace flecsi

#endif
