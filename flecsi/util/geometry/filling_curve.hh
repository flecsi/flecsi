// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_GEOMETRY_FILLING_CURVE_HH
#define FLECSI_UTIL_GEOMETRY_FILLING_CURVE_HH

#include "flecsi/util/geometry/point.hh"

namespace flecsi {

// Space filling curve
template<Dimension DIM, typename T, class DERIVED>
class filling_curve
{
  // Dimension of the curve, 1D, 2D or 3D
  static constexpr Dimension dimension = DIM;
  // Integer type used to represent the key
  using int_t = T;
  // Geometric point to represent coordinates
  // \todo Template double type
  using point_t = util::point<double, dimension>;

protected:
  static constexpr std::size_t bits_ =
    sizeof(int_t) * 8; // Maximum number of bits for representation
  static constexpr std::size_t max_depth_ =
    (bits_ - 1) / dimension; // Maximum depth reachable regarding the size of
                             // the memory word used

  int_t value_;

public:
  filling_curve() : value_(0) {}

  filling_curve(int_t value) : value_(value) {}

  // Max depth possible for this key
  static std::size_t max_depth() {
    return max_depth_;
  }
  // Smallest value possible at max_depth
  static constexpr DERIVED min() {
    return DERIVED(int_t(1) << max_depth_ * dimension);
  }
  // Biggest value possible at max_depth
  static constexpr DERIVED max() {
    int_t id = ~static_cast<int_t>(0);
    for(std::size_t i = max_depth_ * dimension + 1; i < bits_; ++i) {
      id ^= int_t(1) << i;
    } // for
    return DERIVED(id);
  }
  // Get the root key (depth 0)
  static constexpr DERIVED root() {
    return DERIVED(int_t(1));
  }
  // Get the null id = 0 (no root)
  static constexpr DERIVED null() {
    return DERIVED(0);
  }
  // Check if value_ is null
  constexpr bool is_null() const {
    return value_ == int_t(0);
  }
  // Find the depth of the current key
  std::size_t depth() const {
    int_t id = value_;
    std::size_t d = 0;
    while(id >>= dimension)
      ++d;
    return d;
  }
  // Push bits onto the end of this key
  void push(int_t bits) {
    assert(bits < int_t(1) << dimension);
    value_ <<= dimension;
    value_ |= bits;
  }
  // Pop the last bits of the key
  void pop() {
    assert(depth() > 0);
    value_ >>= dimension;
  }
  // Search for the conflicting depth between key_a and key_b
  static int conflict_depth(filling_curve key_a, filling_curve key_b) {
    int conflict = max_depth_;
    while(key_a != key_b) {
      key_a.pop();
      key_b.pop();
      --conflict;
    } // while
    return conflict;
  }
  // Pop last bits and return its value
  int pop_value() {
    assert(depth() > 0);
    int poped = 0;
    poped = static_cast<int>(value_ & ((1 << (dimension)) - 1));
    assert(poped < (1 << dimension));
    value_ >>= dimension;
    return poped;
  }
  // Return the last bits of the key
  int last_value() {
    int poped = 0;
    poped = static_cast<int>(value_ & ((1 << (dimension)) - 1));
    return poped;
  }
  // Pop the depth d bits from the end of this key
  void pop(std::size_t d) {
    value_ >>= d * dimension;
  }
  // Return the parent of this key (depth - 1)
  constexpr filling_curve parent() const {
    return DERIVED(value_ >> dimension);
  }
  // Truncate this key until it is of depth \p to_depth
  void truncate(std::size_t to_depth) {
    std::size_t d = depth();
    if(d < to_depth) {
      return;
    }
    value_ >>= (d - to_depth) * dimension;
  }
  // Output a key using oct in 3d and poping values for 2 and 1D
  void output_(std::ostream & ostr) const {
    if(dimension == 3) {
      ostr << std::oct << value_ << std::dec;
    }
    else {
      std::string output;
      filling_curve id = *this;
      int poped;
      while(id != root()) {
        poped = id.pop_value();
        output.insert(0, std::to_string(poped));
      } // while
      output.insert(output.begin(), '1');
      ostr << output.c_str();
    } // if else
  }
  // Get the value associated to this key
  int_t value() const {
    return value_;
  }
  // Convert this key to coordinates in range.
  void coordinates(const std::array<point_t, 2> &, point_t &) {}
  // Compute the range of a branch from its key
  // The space is recursively decomposed regarding the dimension
  std::array<point_t, 2> range(const std::array<point_t, 2> &) {
    return std::array<point_t, 2>{};
  }
  constexpr bool operator==(const filling_curve & bid) const {
    return value_ == bid.value_;
  }
  constexpr bool operator<=(const filling_curve & bid) const {
    return value_ <= bid.value_;
  }
  constexpr bool operator>=(const filling_curve & bid) const {
    return value_ >= bid.value_;
  }
  constexpr bool operator>(const filling_curve & bid) const {
    return value_ > bid.value_;
  }
  constexpr bool operator<(const filling_curve & bid) const {
    return value_ < bid.value_;
  }
  constexpr bool operator!=(const filling_curve & bid) const {
    return value_ != bid.value_;
  }
  explicit operator int_t() const {
    return value_;
  }
}; // class filling_curve

// output for filling_curve using output_ function defined in the class
template<Dimension D, typename T, class DER>
std::ostream &
operator<<(std::ostream & ostr, const filling_curve<D, T, DER> & k) {
  k.output_(ostr);
  return ostr;
}

// Hilbert-Peano space filling curve
template<Dimension DIM, typename T>
class hilbert_curve : public filling_curve<DIM, T, hilbert_curve<DIM, T>>
{
  using int_t = T;
  static constexpr Dimension dimension = DIM;
  using coord_t = std::array<int_t, dimension>;
  using point_t = flecsi::util::point<double, dimension>;

  using filling_curve<DIM, T, hilbert_curve>::value_;
  using filling_curve<DIM, T, hilbert_curve>::max_depth_;
  using filling_curve<DIM, T, hilbert_curve>::bits_;

public:
  hilbert_curve() : filling_curve<DIM, T, hilbert_curve>() {}
  hilbert_curve(const int_t & id) : filling_curve<DIM, T, hilbert_curve>(id) {}
  hilbert_curve(const std::array<point_t, 2> & range, const point_t & p)
    : hilbert_curve(range,
        p,
        filling_curve<DIM, T, hilbert_curve>::max_depth_) {}
  ~hilbert_curve() = default;

  // Hilbert key is always generated to the max_depth_ and then truncated
  // otherwise the key will not be the same
  hilbert_curve(const std::array<point_t, 2> & range,
    const point_t & p,
    const std::size_t depth) {
    *this = filling_curve<DIM, T, hilbert_curve>::min();
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

  void coordinates(const std::array<point_t, 2> & range, point_t & p) {
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
  }

  std::array<point_t, 2> range(const std::array<point_t, 2> &) {
    return std::array<point_t, 2>{};
  } // range

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

// Morton space filling curve (Z ordering)
template<Dimension DIM, typename T>
class morton_curve : public filling_curve<DIM, T, morton_curve<DIM, T>>
{

  using int_t = T;
  static constexpr Dimension dimension = DIM;
  using coord_t = std::array<int_t, dimension>;
  using point_t = flecsi::util::point<double, dimension>;

  using filling_curve<DIM, T, morton_curve>::value_;
  using filling_curve<DIM, T, morton_curve>::max_depth_;
  using filling_curve<DIM, T, morton_curve>::bits_;

public:
  morton_curve() : filling_curve<DIM, T, morton_curve>() {}
  morton_curve(const int_t & id) : filling_curve<DIM, T, morton_curve>(id) {}
  morton_curve(const std::array<point_t, 2> & range, const point_t & p)
    : morton_curve(range, p, filling_curve<DIM, T, morton_curve>::max_depth_) {}
  ~morton_curve() = default;

  // Morton key can be generated directly up to the right depth
  morton_curve(const std::array<point_t, 2> & range,
    const point_t & p,
    const std::size_t depth) {
    *this = filling_curve<DIM, T, morton_curve>::min();
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
  } // morton_curve

  void coordinates(const std::array<point_t, 2> & range, point_t & p) {
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
  } //  coordinates

  std::array<point_t, 2> range(const std::array<point_t, 2> & range) {
    // The result range
    std::array<point_t, 2> result;
    result[0] = range[0];
    result[1] = range[1];
    // Copy the key
    int_t root = filling_curve<DIM, T, morton_curve>::root().value_;
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

} // namespace flecsi

#endif
