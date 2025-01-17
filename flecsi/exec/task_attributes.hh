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
  leaf = 0x4,
  inner = 0x8,
  /// Ignored.
  /// \deprecated No effect.
  idempotent [[deprecated("has no effect")]] = 0x10,
  loc = 0, ///< Run on a Latency-Optimized Core (a CPU).
  /// Run on a Throughput-Optimized Core (a GPU).
  /// The task function itself still runs on the host, but a GPU is reserved
  /// for its use and field data is made available there.
  ///
  /// \warning MPI backend: Running one process per node likely
  ///          leads to poor performance.
  toc,
  /// Run as an OpenMP task
  ///
  /// \note Legion backend: Can improve OpenMP task execution, since Legion
  ///       knows to assign an entire node to such a task
  ///
  /// \warning MPI backend: Running one process per core likely
  ///          leads to poor performance.
  omp,
  /// Run simultaneously on all processes with the obvious color mapping;
  /// allow MPI communication among point tasks, at the cost of significant
  /// startup overhead.
  mpi
}; // task_attributes_mask_t

/// The task attribute to use for tasks that use the
/// \ref kernel "on-node parallelism interface".  Defined as \c toc or \c omp
/// if support for one of those is available, otherwise \c loc.
/// \warning Using \c toc causes field data to be placed on the device, so
///   that it is accessible \e only via the parallelism interface.
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
  Enumeration of processor types.
 */

enum class processor : size_t { loc, toc, omp, mpi };

// Bits for representing task attributes
inline constexpr TaskAttributes processor_mask = 0x3;

constexpr auto
as_mask(processor t) {
  return static_cast<task_attributes_mask_t>(t);
}

constexpr processor
mask_to_processor_type(TaskAttributes mask) {
  return static_cast<processor>(mask & processor_mask);
} // mask_to_processor_type

/// \}
} // namespace exec
  /// \endcond
} // namespace flecsi

#endif
