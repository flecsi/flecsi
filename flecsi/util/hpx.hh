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

#if !defined(FLECSI_ENABLE_HPX)
#error FLECSI_ENABLE_HPX not defined! This file depends on HPX!
#endif

#include <hpx/modules/collectives.hpp>

#include <cstddef> // byte
#include <type_traits>
#include <vector>

namespace flecsi {
namespace util {
namespace hpx {

/*!
  One-to-All (variable) communication pattern.

  This function uses the FleCSI serialization interface with a packing
  callable object to communicate data from the root rank (0) to all
  other ranks.

  @tparam F The packing functor type with signature \em (rank, size).

  @param f    A callable object.
  @param comm An MPI communicator.

  @return For all ranks besides the root rank (0), the communicated data.
          For the root rank (0), the callable object applied for the root
          rank (0) and size.
 */

template<typename F>
inline auto
one_to_allv(F const & f, ::hpx::collectives::communicator comm) {
  using return_type = std::decay_t<decltype(f(1, 1))>;

  auto [rank, size] = comm.get_info();

  using namespace ::hpx::collectives;
  if(!rank) {
    std::vector<return_type> send;
    send.reserve(size);
    for(std::size_t r = 0; r < size; ++r)
      send.push_back(f(r, size));
    return scatter_to(comm, std::move(send), this_site_arg(rank));
  }
  else {
    return scatter_from<return_type>(comm, this_site_arg(rank));
  }
} // one_to_allv

/// Send data from rank 0 to all others, controlling memory usage.
/// \a mem counts only data being transmitted; one more value may exist.
/// \param f function object
/// \param mem bytes of memory to use for communication
template<class F>
auto
one_to_alli(F && f, std::size_t mem, ::hpx::collectives::communicator comm) {
  return one_to_allv(std::forward<F>(f), comm);
}

/*!
  All-to-All (variable) communication pattern.

  This function uses the FleCSI serialization interface with a packing
  callable object to communicate data from all ranks to all other ranks.

  @tparam F The packing type with signature \em (rank, size).

  @param f    A callable object.
  @param comm An MPI communicator.

  @return A std::vector<return_type>, where \rm return_type is the type
          returned by the callable object.
 */

template<typename F>
inline auto
all_to_allv(F const & f, ::hpx::collectives::communicator comm) {
  using return_type = std::decay_t<decltype(f(1, 1))>;

  auto [rank, size] = comm.get_info();

  std::vector<return_type> result;
  result.reserve(size);
  for(std::size_t r = 0; r != size; ++r)
    result.push_back(f(r, size));

  return ::hpx::collectives::all_to_all(comm, std::move(result));
} // all_to_allv

/*!
  All gather communication pattern implemented using MPI_Allgather. This
  function is convenient for passing more complicated types. Otherwise,
  it may make more sense to use MPI_Allgather directly.

  This function uses the FleCSI serialization interface to copy data from all
  ranks to all other ranks.

  @tparam T serializable data type

  @param t object to send
  @param comm An MPI communicator.

  @return the values from each rank
 */

template<typename T>
std::vector<T>
all_gather(T const & t, ::hpx::collectives::communicator comm) {

  return ::hpx::collectives::all_gather(comm, t);
} // all_gather

} // namespace hpx
} // namespace util
} // namespace flecsi
