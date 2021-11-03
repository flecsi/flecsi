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

#include <flecsi-config.h>

#include "flecsi/util/serialize.hh"

#if !defined(FLECSI_ENABLE_MPI)
#error FLECSI_ENABLE_MPI not defined! This file depends on MPI!
#endif

#include <complex>
#include <cstddef> // byte
#include <cstdint>
#include <numeric>
#include <type_traits>

#include <mpi.h>

namespace flecsi {
namespace util {
namespace mpi {
/// \defgroup mpi-utils MPI
/// These require MPI tasks, not the MPI backend.
/// \ingroup utils
/// \{

inline void
test(int err) {
  if(err != MPI_SUCCESS) {
    char msg[MPI_MAX_ERROR_STRING + 1];
    int len;
    if(MPI_Error_string(err, msg, &len) != MPI_SUCCESS)
      len = 0;
    msg[len] = 0;
    flog_fatal("MPI error " << err << ": " << msg);
  }
}

namespace detail {
struct guard {
  void commit(MPI_Datatype & d) {
    if(!setup) {
      int keyval;
      test(MPI_Comm_create_keyval(
        MPI_COMM_NULL_COPY_FN, destroy, &keyval, nullptr));
      test(MPI_Comm_set_attr(MPI_COMM_SELF, keyval, this));
      test(MPI_Comm_free_keyval(&keyval));
      setup = true;
    }
    test(MPI_Type_commit(&d));
    v.push_back(d);
  }

private:
  static int destroy(MPI_Comm, int, void * attr, void *) {
    int e = MPI_SUCCESS;
    for(auto & v = static_cast<guard *>(attr)->v;
        !v.empty() && e == MPI_SUCCESS;) {
      e = MPI_Type_free(&v.back());
      v.pop_back();
    }
    return e;
  }

  bool setup = false;
  std::vector<MPI_Datatype> v;
} inline datatypes;

struct vector { // for *v functions
  explicit vector(int n) {
    off.reserve(n);
    sz.reserve(n);
  }

  std::vector<std::byte> data;
  std::vector<int> off, sz;

  void skip() {
    sz.push_back(0);
    off.push_back(data.size());
  }
  template<class T>
  void put(const T & t) {
    const auto n = off.emplace_back(data.size());
    data.resize(n + sz.emplace_back(serial::size(t)));
    auto * p = data.data() + n;
    serial::put(p, t);
  }
};
} // namespace detail

// NB: OpenMPI's predefined handles are not constant expressions.
template<class TYPE>
auto
maybe_static() {
  using namespace std;
  static_assert(is_same_v<TYPE, remove_cv_t<remove_reference_t<TYPE>>>);
  // List is from MPI 3.2 draft.
  // TIP: Specializations would collide for, say, int32_t==int.
  if constexpr(is_arithmetic_v<TYPE>) { // save on template instantiations
    if constexpr(is_same_v<TYPE, char>)
      return MPI_CHAR;
    else if constexpr(is_same_v<TYPE, short>)
      return MPI_SHORT;
    else if constexpr(is_same_v<TYPE, int>)
      return MPI_INT;
    else if constexpr(is_same_v<TYPE, long>)
      return MPI_LONG;
    else if constexpr(is_same_v<TYPE, long long>)
      return MPI_LONG_LONG;
    else if constexpr(is_same_v<TYPE, signed char>)
      return MPI_SIGNED_CHAR;
    else if constexpr(is_same_v<TYPE, unsigned char>)
      return MPI_UNSIGNED_CHAR;
    else if constexpr(is_same_v<TYPE, unsigned short>)
      return MPI_UNSIGNED_SHORT;
    else if constexpr(is_same_v<TYPE, unsigned>)
      return MPI_UNSIGNED;
    else if constexpr(is_same_v<TYPE, unsigned long>)
      return MPI_UNSIGNED_LONG;
    else if constexpr(is_same_v<TYPE, unsigned long long>)
      return MPI_UNSIGNED_LONG_LONG;
    else if constexpr(is_same_v<TYPE, float>)
      return MPI_FLOAT;
    else if constexpr(is_same_v<TYPE, double>)
      return MPI_DOUBLE;
    else if constexpr(is_same_v<TYPE, long double>)
      return MPI_LONG_DOUBLE;
    else if constexpr(is_same_v<TYPE, wchar_t>)
      return MPI_WCHAR;
#ifdef INT8_MIN
    else if constexpr(is_same_v<TYPE, int8_t>)
      return MPI_INT8_T;
    else if constexpr(is_same_v<TYPE, uint8_t>)
      return MPI_UINT8_T;
#endif
#ifdef INT16_MIN
    else if constexpr(is_same_v<TYPE, int16_t>)
      return MPI_INT16_T;
    else if constexpr(is_same_v<TYPE, uint16_t>)
      return MPI_UINT16_T;
#endif
#ifdef INT32_MIN
    else if constexpr(is_same_v<TYPE, int32_t>)
      return MPI_INT32_T;
    else if constexpr(is_same_v<TYPE, uint32_t>)
      return MPI_UINT32_T;
#endif
#ifdef INT64_MIN
    else if constexpr(is_same_v<TYPE, int64_t>)
      return MPI_INT64_T;
    else if constexpr(is_same_v<TYPE, uint64_t>)
      return MPI_UINT64_T;
#endif
    else if constexpr(is_same_v<TYPE, MPI_Aint>)
      return MPI_AINT;
    else if constexpr(is_same_v<TYPE, MPI_Offset>)
      return MPI_OFFSET;
    else if constexpr(is_same_v<TYPE, MPI_Count>)
      return MPI_COUNT;
    else if constexpr(is_same_v<TYPE, bool>)
      return MPI_CXX_BOOL;
  }
  else if constexpr(is_same_v<TYPE, complex<float>>)
    return MPI_CXX_FLOAT_COMPLEX;
  else if constexpr(is_same_v<TYPE, complex<double>>)
    return MPI_CXX_DOUBLE_COMPLEX;
  else if constexpr(is_same_v<TYPE, complex<long double>>)
    return MPI_CXX_LONG_DOUBLE_COMPLEX;
  else if constexpr(is_same_v<TYPE, byte>)
    return MPI_BYTE;
  // else: void
}

template<class T>
MPI_Datatype
type() {
  if constexpr(!std::is_void_v<decltype(maybe_static<T>())>)
    return maybe_static<T>();
  else {
    static_assert(bit_assignable_v<T>);
    static const MPI_Datatype ret = [] {
      MPI_Datatype data_type;
      test(MPI_Type_contiguous(sizeof(T), MPI_BYTE, &data_type));
      detail::datatypes.commit(data_type);
      return data_type;
    }();
    return ret;
  }
}
// Use this to restrict to predefined types before MPI_Init:
template<class T,
  class = std::enable_if_t<!std::is_void_v<decltype(maybe_static<T>())>>>
MPI_Datatype
static_type() {
  return maybe_static<T>();
}

/*!
  Convenience function to get basic MPI communicator information.
 */

inline auto
info(MPI_Comm comm = MPI_COMM_WORLD) {
  int rank, size;
  test(MPI_Comm_size(comm, &size));
  test(MPI_Comm_rank(comm, &rank));
  return std::make_pair(rank, size);
} // info

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
one_to_allv(F const & f, MPI_Comm comm = MPI_COMM_WORLD) {
  using return_type = std::decay_t<decltype(f(1, 1))>;

  auto [rank, size] = info(comm);
  if constexpr(bit_copyable_v<return_type>) {
    std::vector<return_type> send;
    if(!rank)
      send.reserve(size);
    send.resize(1);
    if(!rank)
      for(int r = 1; r < size; ++r)
        send.push_back(f(r, size));
    test(MPI_Scatter(MPI_IN_PLACE,
      0,
      MPI_DATATYPE_NULL,
      send.data(),
      1,
      type<return_type>(),
      comm));
    if(rank)
      return send.front();
  }
  else {
    detail::vector v(size);
    v.skip(); // v.sz used even off-root
    if(rank == 0) {
      for(int r = 1; r < size; ++r)
        v.put(f(r, size));
    }

    test(MPI_Scatter(v.sz.data(),
      1,
      MPI_INT,
      rank ? v.sz.data() : MPI_IN_PLACE,
      1,
      MPI_INT,
      0,
      comm));
    if(rank)
      v.data.resize(v.sz.front());
    test(MPI_Scatterv(v.data.data(),
      v.sz.data(),
      v.off.data(),
      MPI_BYTE,
      v.data.data(),
      v.sz.front(),
      MPI_BYTE,
      0,
      comm));

    if(rank) {
      auto const * p = v.data.data();
      return serial::get<return_type>(p);
    } // if
  }

  return f(0, size);
} // one_to_allv

/*!
  All-to-All (variable) communication pattern.

  This function uses the FleCSI serialization interface with a packing
  callable object to communicate data from all ranks to all other ranks.

  @tparam F The packing type with signature \em (rank, size).

  @param f    A callable object.
  @param comm An MPI communicator.

  @return A \c std::vector<return_type>, where \em return_type is the type
          returned by the callable object.
 */

template<typename F>
inline auto
all_to_allv(F const & f, MPI_Comm comm = MPI_COMM_WORLD) {
  using return_type = std::decay_t<decltype(f(1, 1))>;

  auto [rank, size] = info(comm);
  std::vector<return_type> result;
  result.reserve(size);
  if constexpr(bit_copyable_v<return_type>) {
    for(int r = 0; r < size; ++r)
      result.push_back(f(r, size));
    test(MPI_Alltoall(MPI_IN_PLACE,
      0,
      MPI_DATATYPE_NULL,
      result.data(),
      1,
      type<return_type>(),
      comm));
  }
  else {
    detail::vector recv(size);
    {
      detail::vector send(size);

      for(int r = 0; r < size; ++r) {
        if(r == rank)
          send.skip();
        else
          send.put(f(r, size));
      } // for

      recv.sz.resize(size);
      test(MPI_Alltoall(
        send.sz.data(), 1, MPI_INT, recv.sz.data(), 1, MPI_INT, comm));

      {
        int o = 0;
        for(const auto n : recv.sz) {
          recv.off.push_back(o);
          o += n;
        }
        recv.data.resize(o);
      }

      test(MPI_Alltoallv(send.data.data(),
        send.sz.data(),
        send.off.data(),
        MPI_BYTE,
        recv.data.data(),
        recv.sz.data(),
        recv.off.data(),
        MPI_BYTE,
        comm));
    }

    const std::byte * const p = recv.data.data();
    for(int r = 0; r < size; ++r) {
      if(r == rank)
        result.push_back(f(r, size));
      else
        result.push_back(serial::get1<return_type>(p + recv.off[r]));
    } // for
  }

  return result;
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
all_gather(const T & t, MPI_Comm comm = MPI_COMM_WORLD) {
  auto [rank, size] = info(comm);
  std::vector<T> result;
  if constexpr(bit_copyable_v<T>) {
    const auto typ = type<T>();
    result.resize(size);
    test(MPI_Allgather(&t, 1, typ, result.data(), 1, typ, comm));
  }
  else {
    detail::vector v(size); // just a struct here
    v.sz.resize(size);
    v.sz[rank] = serial::size(t);

    test(MPI_Allgather(
      MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, v.sz.data(), 1, MPI_INT, comm));

    v.off.resize(size);
    std::exclusive_scan(v.sz.begin(), v.sz.end(), v.off.begin(), 0);

    v.data.resize(v.off.back() + v.sz.back());
    {
      std::byte * p = v.data.data() + v.off[rank];
      serial::put(p, t);
    }

    test(MPI_Allgatherv(MPI_IN_PLACE,
      0,
      MPI_DATATYPE_NULL,
      v.data.data(),
      v.sz.data(),
      v.off.data(),
      MPI_BYTE,
      comm));

    result.reserve(size);

    auto p = std::as_const(v).data.data();
    for(int r = 0; r < size; ++r) {
      result.push_back(serial::get<T>(p));
    } // for
  }

  return result;
} // all_gather

/// \}
} // namespace mpi
} // namespace util
} // namespace flecsi
