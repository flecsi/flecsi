// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_TASK_ATTRIBUTES_HH
#define FLECSI_EXEC_TASK_ATTRIBUTES_HH

#include "flecsi/config.hh"

#include <cstddef> // size_t

namespace flecsi {
/// \addtogroup execution
/// \{

using TaskAttributes = unsigned;

/*!
  Task attribute flags.  \ns.
  \deprecated Used only with \c flecsi::execute and \c flecsi::reduce.
 */

enum task_attributes_mask_t : TaskAttributes {
  leaf = 0x4,
  inner = 0x8,
  /// Ignored.
  /// \deprecated No effect.
  idempotent [[deprecated("has no effect")]] = 0x10,
  synchronous_impl = 0x20, // implied by mpi
  /// Run on a Latency-Optimized Core (a CPU).
  /// \deprecated Use \c exec::cpu.
  loc = 0,
  /// Run on a Throughput-Optimized Core (a GPU).
  /// The task function itself still runs on the host, but a GPU is reserved
  /// for its use and field data is made available there.
  /// \deprecated Use \c exec::gpu.
  toc,
  /// Run as an OpenMP task
  ///
  /// \note Legion backend: Can improve OpenMP task execution, since Legion
  ///       knows to assign an entire node to such a task
  /// \deprecated Use \c exec::omp.
  omp,
  /// Run simultaneously on all processes with field data stored on the host
  /// with the obvious color mapping;
  /// allow MPI communication among point tasks, at the cost of significant
  /// startup overhead.
  /// \deprecated Use the needed subset of
  ///   - \c scheduler::wait: finish previous tasks first
  ///   - \c task_class::synchronous: wait on the task to run and let
  ///     reference parameters bind to arguments (but pointer parameters can
  ///     also be used to avoid copies of objects with sufficient lifetime)
  ///   - \c exec::group: run each point task on the corresponding process and
  ///     support mutation through parameters
  ///   - <code>\ref communicator</code>: further, allow MPI usage
  mpi
}; // task_attributes_mask_t

/// The task attribute to use for tasks that use the
/// \ref kernel "on-node parallelism interface".
/// \warning Using \c toc causes field data to be placed on the device, so
///   that it is accessible \e only via the parallelism interface.
///
/// \ns.
/// \deprecated Use \c accelerator.
[[deprecated("use accelerator")]] inline constexpr auto default_accelerator =
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
