// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_GEOMETRY_FILLING_CURVE_KEY_HH
#define FLECSI_UTIL_GEOMETRY_FILLING_CURVE_KEY_HH

#include "flecsi/util/geometry/point.hh"
#include <climits>

namespace flecsi::util {
/// \ingroup utils
/// \defgroup fillingcurves Filling Curves
/// Space-filling curve, key generators
/// \{

/// Space filling curve keys generator, CRTP base
/// \tparam DIM The spatial dimension for the filling curve (1, 2 or 3
/// supported)
/// \tparam T The integer type used to represent the keys. The type
/// is used as a bit-field.
/// \tparam DERIVED derived class (see below for requirements)
template<Dimension DIM, typename T, class DERIVED>
class filling_curve_key
{
  // Dimension of the curve, 1D, 2D or 3D
  static constexpr Dimension dimension = DIM;

public:
  /// Integer type used to represent the key
  using int_t = T;

protected:
  static constexpr std::size_t bits_ =
    sizeof(int_t) * CHAR_BIT; // Maximum number of bits for representation
  static constexpr std::size_t max_depth_ =
    (bits_ - 1) / dimension; // Maximum depth reachable regarding the size of
                             // the memory word used
  static constexpr std::size_t leading_zeros =
    bits_ - (max_depth_ * dimension + 1);

  /// Value of the filling curve key.
  /// int_t is used as a bit-field to represent the filling curve key.
  int_t value_;

  ~filling_curve_key() = default;

public:
  /// Default constructor: create an invalid key containing only zeros.
  constexpr filling_curve_key() : value_(0) {}

  /// Construct a key from an integer of type int_t
  constexpr explicit filling_curve_key(int_t value) : value_(value) {}

  /// Max depth possible for this type of key
  static constexpr std::size_t max_depth() {
    return max_depth_;
  }
  /// Smallest value possible at max_depth
  static constexpr DERIVED min() {
    return DERIVED(int_t(1) << max_depth_ * dimension);
  }
  /// Biggest value possible at max_depth
  static constexpr DERIVED max() {
    int_t id = ~static_cast<int_t>(0);
    id >>= bits_ - (max_depth_ * dimension + 1);
    return DERIVED(id);
  }
  /// Get the root key (depth 0)
  static constexpr DERIVED root() {
    return DERIVED(int_t(1));
  }
  /// Find the depth of the current key
  std::size_t depth() const {
    return max_depth_ - (__builtin_clz(value_) - leading_zeros) / dimension;
  }
  /// Push bits onto the end of this key
  auto push(int_t bits) const {
    auto ret = *this;
    assert(bits < int_t(1) << dimension);
    ret.value_ <<= dimension;
    ret.value_ |= bits;
    return DERIVED(ret.value_);
  }
  /// Pop last bits and return its value
  int_t pop() {
    assert(depth() > 0);
    int_t popped = last_value();
    value_ >>= dimension;
    return popped;
  }
  /// Return the last bits of the key
  int_t last_value() const {
    return value_ & ((1 << dimension) - 1);
  }
  /// Pop the depth d bits from the end of this key
  void pop(std::size_t d) {
    value_ >>= d * dimension;
  }
  /// Return the parent of this key (depth - 1)
  constexpr filling_curve_key parent() const {
    return DERIVED(value_ >> dimension);
  }
  /// Display a key, starting from the root (1), grouping the bits in digits.
  /// The digits are composed of 1, 2, 3 bits matching the dimension used for
  /// the key.
  friend std::ostream & operator<<(std::ostream & ostr,
    const filling_curve_key & fc) {
    if constexpr(dimension == 3) {
      auto iosflags = ostr.setf(std::ios::oct, std::ios::basefield);
      ostr << fc.value_;
      ostr.flags(iosflags);
    }
    else {
      std::string output;
      filling_curve_key id = fc;
      while(id != root())
        output.insert(0, std::to_string(id.pop()));
      output.insert(output.begin(), '1');
      ostr << output.c_str();
    } // if else
    return ostr;
  }
  /// Get the value associated to this key
  int_t value() const {
    return value_;
  }

  /// \name Relational and equality operators
  /// \{
  /// Equality operator
  /// \return true if the other key has the same value
  constexpr bool operator==(const filling_curve_key & bid) const {
    return value_ == bid.value_;
  }
  /// Less than or equal to operator
  /// \return true if this key is less or equal than the other
  constexpr bool operator<=(const filling_curve_key & bid) const {
    return value_ <= bid.value_;
  }
  /// Greater than or equal to operator
  /// \return true if the other key is less or equal than this
  constexpr bool operator>=(const filling_curve_key & bid) const {
    return value_ >= bid.value_;
  }
  /// Greater than operator
  /// \return true if the other key is less than this
  constexpr bool operator>(const filling_curve_key & bid) const {
    return value_ > bid.value_;
  }
  /// Less than operator
  /// \return true if this key is less than the other
  constexpr bool operator<(const filling_curve_key & bid) const {
    return value_ < bid.value_;
  }
  /// Inequality operator
  /// \return true if the two keys have different values
  constexpr bool operator!=(const filling_curve_key & bid) const {
    return value_ != bid.value_;
  }
  /// \}
}; // class filling_curve_key

#ifdef DOXYGEN

/// Example implementation of filling curve derived from the
/// filling_curve_key CRTP. This class is not really implemented.
template<Dimension DIM, typename T>
struct my_key : filling_curve_key<DIM, T, my_key<DIM, T>> {
  /// Construct a key based on a key value.
  explicit my_key(T it);
};

#endif

/// Point on a Hilbert-Peano space filling curve.
template<Dimension DIM, typename T>
class hilbert_key : public filling_curve_key<DIM, T, hilbert_key<DIM, T>>
{
public:
  using typename hilbert_key::filling_curve_key::int_t;
  /// Point type to represent coordinates.
  using point_t = flecsi::util::point<double, DIM>;

private:
  static constexpr Dimension dimension = DIM;
  using coord_t = std::array<int_t, dimension>;
  using hilbert_key::filling_curve_key::bits_;
  using hilbert_key::filling_curve_key::filling_curve_key;
  using hilbert_key::filling_curve_key::max_depth_;
  using hilbert_key::filling_curve_key::value_;

public:
  /// Create a Hilbert key at a specific depth.
  /// For Hilbert key, they key is generated to the max_depth_ and then
  /// truncated
  /// \param range The bounding box of the overall domain
  /// \param p The point that the key will represent in the \p range
  /// \param depth The depth at which to generate the key.
  hilbert_key(const std::array<point_t, 2> & range,
    const point_t & p,
    const std::size_t depth = max_depth_) {
    *this = hilbert_key::min();
    assert(depth <= max_depth_);
    std::array<int_t, dimension> coords;
    const int_t max_val = (int_t(1) << (bits_ - 1) / dimension) - 1;

    // Convert the position to integer
    for(Dimension i = 0; i < dimension; ++i) {
      double min = range[0][i];
      double scale = range[1][i] - min;
      coords[i] = std::min(max_val,
        static_cast<int_t>((p[i] - min) / scale * (int_t(1) << (max_depth_))));
    }
    // Handle 1D case
    if(dimension == 1) {
      assert(value_ & 1UL << max_depth_);
      value_ |= coords[0] >> dimension;
      value_ >>= (max_depth_ - depth);
      return;
    }
    int_t mask = static_cast<int_t>(1) << (max_depth_);
    for(int_t s = mask >> 1; s > 0; s >>= 1) {
      std::array<int_t, dimension> bits;
      for(Dimension j = 0; j < dimension; ++j) {
        bits[j] = (s & coords[j]) > 0;
      }
      if(dimension == 2) {
        value_ += s * s * ((3 * bits[0]) ^ bits[1]);
        rotation2d(s, coords, bits);
      }
      if(dimension == 3) {
        value_ += s * s * s * ((7 * bits[0]) ^ (3 * bits[1]) ^ bits[2]);
        rotation3d(s, coords, bits);
      }
    }
    // Then truncate the key to the depth
    value_ >>= (max_depth_ - depth) * dimension;
  }

  /// Convert this key to coordinates in range.
  /// \param range The bounding box of the overall domain
  point_t coordinates(const std::array<point_t, 2> & range) {
    point_t p;
    int_t key = value_;
    std::array<int_t, dimension> coords;
    coords.fill(int_t(0));

    int_t n = int_t(1) << (max_depth_); // Number of cells to an edge.
    for(int_t mask = int_t(1); mask < n; mask <<= 1) {
      std::array<int_t, dimension> bits = {};
      if(dimension == 3) {
        bits[0] = (key & 4) > 0;
        bits[1] = ((key & 2) ^ bits[0]) > 0;
        bits[2] = ((key & 1) ^ bits[0] ^ bits[1]) > 0;
        rotation3d(mask, coords, bits);
        coords[0] += bits[0] * mask;
        coords[1] += bits[1] * mask;
        coords[2] += bits[2] * mask;
      }
      if(dimension == 2) {
        bits[0] = (key & 2) > 0;
        bits[1] = ((key & 1) ^ bits[0]) > 0;
        rotation2d(mask, coords, bits);
        coords[0] += bits[0] * mask;
        coords[1] += bits[1] * mask;
      }
      key >>= dimension;
    }
    assert(key == int_t(1));
    for(Dimension j = 0; j < dimension; ++j) {
      double min = range[0][j];
      double scale = range[1][j] - min;
      p[j] =
        min + scale * static_cast<double>(coords[j]) / (int_t(1) << max_depth_);
    } // for
    return p;
  }

private:
  void rotation2d(const int_t & n,
    std::array<int_t, dimension> & coords,
    const std::array<int_t, dimension> & bits) {
    if(bits[1] == 0) {
      if(bits[0] == 1) {
        coords[0] = n - 1 - coords[0];
        coords[1] = n - 1 - coords[1];
      }
      // Swap X-Y
      int t = coords[0];
      coords[0] = coords[1];
      coords[1] = t;
    }
  }

  void rotate_90_x(const int_t & n, std::array<int_t, dimension> & coords) {
    coord_t tmp = coords;
    coords[0] = tmp[0];
    coords[1] = n - 1 - tmp[2];
    coords[2] = tmp[1];
  }
  void rotate_90_y(const int_t & n, std::array<int_t, dimension> & coords) {
    coord_t tmp = coords;
    coords[0] = tmp[2];
    coords[1] = tmp[1];
    coords[2] = n - 1 - tmp[0];
  }
  void rotate_90_z(const int_t & n, std::array<int_t, dimension> & coords) {
    coord_t tmp = coords;
    coords[0] = n - 1 - tmp[1];
    coords[1] = tmp[0];
    coords[2] = tmp[2];
  }
  void rotate_180_x(const int_t & n, std::array<int_t, dimension> & coords) {
    coord_t tmp = coords;
    coords[0] = tmp[0];
    coords[1] = n - 1 - tmp[1];
    coords[2] = n - 1 - tmp[2];
  }
  void rotate_270_x(const int_t & n, std::array<int_t, dimension> & coords) {
    coord_t tmp = coords;
    coords[0] = tmp[0];
    coords[1] = tmp[2];
    coords[2] = n - 1 - tmp[1];
  }
  void rotate_270_y(const int_t & n, std::array<int_t, dimension> & coords) {
    coord_t tmp = coords;
    coords[0] = n - 1 - tmp[2];
    coords[1] = tmp[1];
    coords[2] = tmp[0];
  }
  void rotate_270_z(const int_t & n, std::array<int_t, dimension> & coords) {
    coord_t tmp = coords;
    coords[0] = tmp[1];
    coords[1] = n - 1 - tmp[0];
    coords[2] = tmp[2];
  }

  void rotation3d(const int_t & n,
    std::array<int_t, dimension> & coords,
    const std::array<int_t, dimension> & bits) {
    if(!bits[0] && !bits[1] && !bits[2]) {
      // Left front bottom
      rotate_270_z(n, coords);
      rotate_270_x(n, coords);
    }
    else if(!bits[0] && bits[2]) {
      // Left top
      rotate_90_z(n, coords);
      rotate_90_y(n, coords);
    }
    else if(bits[1] && !bits[2]) {
      // Back bottom
      rotate_180_x(n, coords);
    }
    else if(bits[0] && bits[2]) {
      // Right top
      rotate_270_z(n, coords);
      rotate_270_y(n, coords);
    }
    else if(bits[0] && !bits[2] && !bits[1]) {
      // Right front bottom
      rotate_90_y(n, coords);
      rotate_90_z(n, coords);
    }
  }

  void unrotation3d(const int_t & n,
    std::array<int_t, dimension> & coords,
    const std::array<int_t, dimension> & bits) {
    if(!bits[0] && !bits[1] && !bits[2]) {
      // Left front bottom
      rotate_90_x(n, coords);
      rotate_90_z(n, coords);
    }
    else if(!bits[0] && bits[2]) {
      // Left top
      rotate_270_y(n, coords);
      rotate_270_z(n, coords);
    }
    else if(bits[1] && !bits[2]) {
      // Back bottom
      rotate_180_x(n, coords);
    }
    else if(bits[0] && bits[2]) {
      // Right top
      rotate_90_y(n, coords);
      rotate_90_z(n, coords);
    }
    else if(bits[0] && !bits[2] && !bits[1]) {
      // Right front bottom
      rotate_270_z(n, coords);
      rotate_270_y(n, coords);
    }
  }
}; // class hilbert

/// Point on a Morton space filling curve.
template<Dimension DIM, typename T>
class morton_key : public filling_curve_key<DIM, T, morton_key<DIM, T>>
{

public:
  using typename morton_key::filling_curve_key::int_t;
  /// Point type to represent coordinates.
  using point_t = flecsi::util::point<double, DIM>;

private:
  static constexpr Dimension dimension = DIM;
  using coord_t = std::array<int_t, dimension>;
  using morton_key::filling_curve_key::bits_;
  using morton_key::filling_curve_key::filling_curve_key;
  using morton_key::filling_curve_key::max_depth_;
  using morton_key::filling_curve_key::value_;

public:
  /// Create a Morton key at a specific depth.
  /// \param range The bounding box of the overall domain
  /// \param p The point that the key will represent in the \p range
  /// \param depth The depth at which to generate the key.
  morton_key(const std::array<point_t, 2> & range,
    const point_t & p,
    const std::size_t depth = max_depth_) {
    *this = morton_key::min();
    assert(depth <= max_depth_);
    std::array<int_t, dimension> coords;
    const int_t max_val = (int_t(1) << (bits_ - 1) / dimension) - 1;
    for(Dimension i = 0; i < dimension; ++i) {
      double min = range[0][i];
      double scale = range[1][i] - min;
      coords[i] = std::min(max_val,
        static_cast<int_t>(
          (p[i] - min) / scale *
          static_cast<double>((int_t(1) << (bits_ - 1) / dimension))));
    } // for
    std::size_t k = 0;
    for(std::size_t i = max_depth_ - depth; i < max_depth_; ++i) {
      for(Dimension j = 0; j < dimension; ++j) {
        int_t bit = (coords[j] & int_t(1) << i) >> i;
        value_ |= bit << (k * dimension + j);
      } // for
      ++k;
    } // for
  } // morton_key

  /// Convert this key to coordinates in range.
  /// range The bounding box of the overall domain
  point_t coordinates(const std::array<point_t, 2> & range) {
    point_t p;
    std::array<int_t, dimension> coords;
    coords.fill(int_t(0));
    int_t id = value_;
    std::size_t d = 0;
    while(id >> dimension != int_t(0)) {
      for(Dimension j = 0; j < dimension; ++j) {
        coords[j] |= (((int_t(1) << j) & id) >> j) << d;
      } // for
      id >>= dimension;
      ++d;
    } // while
    constexpr int_t m = (int_t(1) << max_depth_) - 1;
    for(Dimension j = 0; j < dimension; ++j) {
      double min = range[0][j];
      double scale = range[1][j] - min;
      coords[j] <<= max_depth_ - d;
      p[j] = min + scale * static_cast<double>(coords[j]) / m;
    } // for
    return p;
  } //  coordinates

  /// Compute the bounding box of a branch from its key in the overall domain
  /// The space is recursively decomposed regarding the dimension
  /// \param range The bounding box of the overall domain
  std::array<point_t, 2> range(const std::array<point_t, 2> & range) {
    // The result range
    std::array<point_t, 2> result;
    result[0] = range[0];
    result[1] = range[1];
    // Copy the key
    int_t root = morton_key::root().value_;
    // Extract x,y and z
    std::array<int_t, dimension> coords;
    coords.fill(int_t(0));
    int_t id = value_;
    std::size_t d = 0;
    while(id != root) {
      for(Dimension j = 0; j < dimension; ++j) {
        coords[j] |= (((int_t(1) << j) & id) >> j) << d;
      }
      id >>= dimension;
      ++d;
    }
    for(Dimension i = 0; i < dimension; ++i) {
      // apply the reduction
      for(std::size_t j = d; j > 0; --j) {
        double nu = (result[0][i] + result[1][i]) / 2.;
        if(coords[i] & (int_t(1) << j - 1)) {
          result[0][i] = nu;
        }
        else {
          result[1][i] = nu;
        } // if
      } //  for
    } // for
    return result;
  } // range
}; // class morton

/// \}

} // namespace flecsi::util

// We need to specify the numeric_limits in order to use the filling curves with
// the distributed sort utility
namespace std {
template<flecsi::Dimension DIM, typename T>
struct numeric_limits<flecsi::util::morton_key<DIM, T>> {
  static constexpr flecsi::util::morton_key<DIM, T> min() {
    return flecsi::util::morton_key<DIM, T>::min();
  }
  static constexpr flecsi::util::morton_key<DIM, T> max() {
    return flecsi::util::morton_key<DIM, T>::max();
  }
};
template<flecsi::Dimension DIM, typename T>
struct numeric_limits<flecsi::util::hilbert_key<DIM, T>> {
  static constexpr flecsi::util::hilbert_key<DIM, T> min() {
    return flecsi::util::hilbert_key<DIM, T>::min();
  }
  static constexpr flecsi::util::hilbert_key<DIM, T> max() {
    return flecsi::util::hilbert_key<DIM, T>::max();
  }
};
} // namespace std

#endif
