// Copyright (c) 2016 Los Alamos National Laboratory, LLC
// All rights reserved

#ifndef FLECSI_UTIL_GEOMETRY_POINT_HH
#define FLECSI_UTIL_GEOMETRY_POINT_HH

#include "flecsi/util/common.hh"
#include "flecsi/util/dimensioned_array.hh"

#include <array>
#include <cmath>

namespace flecsi {
namespace util {
/// \ingroup utils
/// \defgroup point Point
/// Spatial representation of a point based on dimensioned_array
/// \{

/// The point type defines an interface for storing and manipulating
/// coordinate data. The point type is implemented using
/// \c dimensioned_array. Supports +, -, and *.
/// \tparam TYPE      The type to use to represent coordinate values.
/// \tparam DIMENSION The dimension of the point.
template<typename TYPE, Dimension DIMENSION>
using point = dimensioned_array<TYPE, DIMENSION, 1>;

template<typename TYPE, Dimension DIMENSION>
constexpr point<TYPE, DIMENSION>
operator*(TYPE const val, point<TYPE, DIMENSION> const & p) {
  point<TYPE, DIMENSION> tmp(p);
  for(Dimension d = 0; d < DIMENSION; ++d) {
    tmp[d] *= val;
  } // for

  return tmp;
} // operator *

/// Return the distance between the given points \p a and \p b.
/// \tparam TYPE      The type to use to represent coordinate values.
/// \tparam DIMENSION The dimension of the point.
template<typename TYPE, Dimension DIMENSION>
TYPE
distance(point<TYPE, DIMENSION> const & a, point<TYPE, DIMENSION> const & b) {
  TYPE sum(0);
  for(Dimension d = 0; d < DIMENSION; ++d) {
    sum += (square)(a[d] - b[d]);
  } // for

  return std::sqrt(sum);
} // distance

/// Return the midpoint between two points \p a and \p b
/// \tparam TYPE      The type to use to represent coordinate values.
/// \tparam DIMENSION The dimension of the point.
template<typename TYPE, Dimension DIMENSION>
constexpr point<TYPE, DIMENSION>
midpoint(point<TYPE, DIMENSION> const & a, point<TYPE, DIMENSION> const & b) {
  return point<TYPE, DIMENSION>((a + b) / 2.0);
} // midpoint

/// Return the centroid of the given set of points \p points.
/// \tparam TYPE      The type to use to represent coordinate values.
/// \tparam DIMENSION The dimension of the point.
template<template<typename...> class CONTAINER,
  typename TYPE,
  Dimension DIMENSION>
constexpr auto
centroid(CONTAINER<point<TYPE, DIMENSION>> const & points) {
  point<TYPE, DIMENSION> tmp(0.0);

  for(auto p : points) {
    tmp += p;
  } // for

  tmp /= points.size();

  return tmp;
} // centroid

/// Return the centroid of the given set of points \p points.
/// \tparam TYPE      The type to use to represent coordinate values.
/// \tparam DIMENSION The dimension of the point.
template<typename TYPE, Dimension DIMENSION>
constexpr auto
centroid(std::initializer_list<point<TYPE, DIMENSION>> points) {
  point<TYPE, DIMENSION> tmp(0.0);

  for(auto p : points) {
    tmp += p;
  } // for

  tmp /= points.size();

  return tmp;
} // centroid

/// \}

} // namespace util
} // namespace flecsi

#endif
