// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_DIMENSIONED_ARRAY_HH
#define FLECSI_UTIL_DIMENSIONED_ARRAY_HH

#include <array>
#include <cassert>
#include <cmath>
#include <ostream>

namespace flecsi {
namespace util {
/// \ingroup utils

template<typename... CONDITIONS>
struct and_ : std::true_type {};

template<typename CONDITION, typename... CONDITIONS>
struct and_<CONDITION, CONDITIONS...>
  : std::conditional<CONDITION::value, and_<CONDITIONS...>, std::false_type>::
      type {}; // struct and_

template<typename TARGET, typename... TARGETS>
using are_type = and_<std::is_same<TARGETS, TARGET>...>;

//----------------------------------------------------------------------------//
//! Enumeration for axes.
//----------------------------------------------------------------------------//

enum class axis : Dimension { x = 0, y = 1, z = 2 };

//----------------------------------------------------------------------------//
//! The dimensioned_array type provides a general base for defining
//! contiguous array types that have a specific dimension.
//!
//! @tparam TYPE      The type of the array, e.g., P.O.D. type.
//! @tparam DIMENSION The dimension of the array, i.e., the number of elements
//!                   to be stored in the array.
//! @tparam NAMESPACE The namespace of the array.  This is a dummy parameter
//!                   that is useful for creating distinct types that alias
//!                   dimensioned_array.
//----------------------------------------------------------------------------//

template<typename TYPE, Dimension DIMENSION, std::size_t NAMESPACE>
class dimensioned_array
{
public:
  //--------------------------------------------------------------------------//
  //! Initializer list constructor.
  //--------------------------------------------------------------------------//

  constexpr dimensioned_array(std::initializer_list<TYPE> list) : data_() {
    assert(list.size() == DIMENSION && "dimension size mismatch");
    auto p = list.begin();
    for(auto & x : data_)
      x = *p++; // std::copy isn't constexpr until C++20
  } // dimensioned_array

  //--------------------------------------------------------------------------//
  //! Variadic constructor.
  //--------------------------------------------------------------------------//

  template<typename... ARGS,
    typename = typename std::enable_if<sizeof...(ARGS) == DIMENSION &&
                                       are_type<TYPE, ARGS...>::value>>
  constexpr dimensioned_array(ARGS... args)
    : data_{args...} {} // dimensioned_array

  //--------------------------------------------------------------------------//
  //! Constructor (fill with given value).
  //--------------------------------------------------------------------------//

  constexpr dimensioned_array(TYPE const & val) : data_() {
    for(auto & x : data_)
      x = val; // array::fill isn't constexpr until C++20
  } // dimensioned_array

  //--------------------------------------------------------------------------//
  //! Return the size of the array.
  //--------------------------------------------------------------------------//

  static constexpr Dimension size() {
    return DIMENSION;
  } // size

  operator std::array<TYPE, DIMENSION>() {
    return data_;
  }

  //--------------------------------------------------------------------------//
  //! Support for enumerated type access, e.g., da[x], for accessing the
  //! x axis.
  //--------------------------------------------------------------------------//

  template<typename ENUM_TYPE>
  constexpr TYPE & operator[](ENUM_TYPE e) {
    return data_[static_cast<Dimension>(e)];
  } // operator []

  //--------------------------------------------------------------------------//
  //! Support for enumerated type access, e.g., da[x], for accessing the
  //! x axis.
  //--------------------------------------------------------------------------//

  template<typename ENUM_TYPE>
  constexpr TYPE const & operator[](ENUM_TYPE e) const {
    return data_[static_cast<Dimension>(e)];
  } // operator []

  //--------------------------------------------------------------------------//
  //! Assignment operator.
  //--------------------------------------------------------------------------//

  constexpr dimensioned_array & operator=(const TYPE & val) {
    for(Dimension i = 0; i < DIMENSION; i++) {
      data_[i] = val;
    } // for
    return *this;
  } // operator =

  //--------------------------------------------------------------------------//
  // Macro to avoid code replication.
  //--------------------------------------------------------------------------//

#define define_operator(op)                                                    \
  constexpr dimensioned_array & operator op(dimensioned_array const & rhs) {   \
    if(this != &rhs) {                                                         \
      for(Dimension i = 0; i < DIMENSION; i++) {                               \
        data_[i] op rhs[i];                                                    \
      } /* for */                                                              \
    } /* if */                                                                 \
                                                                               \
    return *this;                                                              \
  }

  //--------------------------------------------------------------------------//
  // Macro to avoid code replication.
  //--------------------------------------------------------------------------//

#define define_operator_type(op)                                               \
  constexpr dimensioned_array & operator op(TYPE val) {                        \
    for(Dimension i = 0; i < DIMENSION; i++) {                                 \
      data_[i] op val;                                                         \
    } /* for */                                                                \
                                                                               \
    return *this;                                                              \
  }

  // clang-format off

  //--------------------------------------------------------------------------//
  //! Addition/Assignment operator.
  //--------------------------------------------------------------------------//

  define_operator(+=)

  //--------------------------------------------------------------------------//
  //! Addition/Assignment operator.
  //--------------------------------------------------------------------------//

  define_operator_type(+=)

  //--------------------------------------------------------------------------//
  //! Subtraction/Assignment operator.
  //--------------------------------------------------------------------------//

  define_operator(-=)

  //--------------------------------------------------------------------------//
  //! Subtraction/Assignment operator.
  //--------------------------------------------------------------------------//

  define_operator_type(-=)

  //--------------------------------------------------------------------------//
  //! Multiplication/Assignment operator.
  //--------------------------------------------------------------------------//

  define_operator(*=)

  //--------------------------------------------------------------------------//
  //! Multiplication/Assignment operator.
  //--------------------------------------------------------------------------//

  define_operator_type(*=)

  //--------------------------------------------------------------------------//
  //! Division/Assignment operator.
  //--------------------------------------------------------------------------//

  define_operator(/=)

  //--------------------------------------------------------------------------//
  //! Division/Assignment operator.
  //--------------------------------------------------------------------------//

  define_operator_type(/=)

  //--------------------------------------------------------------------------//
  //! Equality operator
  //--------------------------------------------------------------------------//

  constexpr bool operator==(const dimensioned_array & da) {
    return this->data_ == da.data_;
  }

  // clang-format on

  //! \brief Division operator involving a constant.
  //! \param[in] val The constant on the right hand side of the operator.
  //! \return A reference to the current object.
  constexpr dimensioned_array operator/(TYPE val) {
    dimensioned_array tmp(*this);
    tmp /= val;

    return tmp;
  } // operator /

  constexpr auto data() {
    return &data_;
  }

private:
  std::array<TYPE, DIMENSION> data_;

}; // class dimensioned_array

//----------------------------------------------------------------------------//
//! Addition operator.
//!
//! @tparam TYPE      The type of the array, e.g., P.O.D. type.
//! @tparam DIMENSION The dimension of the array, i.e., the number of elements
//!                   to be stored in the array.
//! @tparam NAMESPACE The namespace of the array.  This is a dummy parameter
//!                   that is useful for creating distinct types that alias
//!                   dimensioned_array.
//----------------------------------------------------------------------------//

template<typename TYPE, Dimension DIMENSION, std::size_t NAMESPACE>
constexpr dimensioned_array<TYPE, DIMENSION, NAMESPACE>
operator+(const dimensioned_array<TYPE, DIMENSION, NAMESPACE> & lhs,
  const dimensioned_array<TYPE, DIMENSION, NAMESPACE> & rhs) {
  dimensioned_array<TYPE, DIMENSION, NAMESPACE> tmp(lhs);
  tmp += rhs;
  return tmp;
} // operator +

//----------------------------------------------------------------------------//
//! Addition operator.
//!
//! @tparam TYPE      The type of the array, e.g., P.O.D. type.
//! @tparam DIMENSION The dimension of the array, i.e., the number of elements
//!                   to be stored in the array.
//! @tparam NAMESPACE The namespace of the array.  This is a dummy parameter
//!                   that is useful for creating distinct types that alias
//!                   dimensioned_array.
//----------------------------------------------------------------------------//

template<typename TYPE, Dimension DIMENSION, std::size_t NAMESPACE>
constexpr dimensioned_array<TYPE, DIMENSION, NAMESPACE>
operator-(const dimensioned_array<TYPE, DIMENSION, NAMESPACE> & lhs,
  const dimensioned_array<TYPE, DIMENSION, NAMESPACE> & rhs) {
  dimensioned_array<TYPE, DIMENSION, NAMESPACE> tmp(lhs);
  tmp -= rhs;
  return tmp;
} // operator -

//----------------------------------------------------------------------------//
//! Addition operator.
//!
//! @tparam TYPE      The type of the array, e.g., P.O.D. type.
//! @tparam DIMENSION The dimension of the array, i.e., the number of elements
//!                   to be stored in the array.
//! @tparam NAMESPACE The namespace of the array.  This is a dummy parameter
//!                   that is useful for creating distinct types that alias
//!                   dimensioned_array.
//!
//! @param stream The output stream.
//! @param a      The dimensioned array.
//----------------------------------------------------------------------------//

template<typename TYPE, Dimension DIMENSION, std::size_t NAMESPACE>
std::ostream &
operator<<(std::ostream & stream,
  dimensioned_array<TYPE, DIMENSION, NAMESPACE> const & a) {
  stream << "[";

  for(Dimension i = 0; i < DIMENSION; i++) {
    stream << " " << a[i];
  } // for

  stream << " ]";

  return stream;
} // operator <<

} // namespace util
} // namespace flecsi

#endif
