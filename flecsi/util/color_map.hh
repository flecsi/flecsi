// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_COLOR_MAP_HH
#define FLECSI_UTIL_COLOR_MAP_HH

#include <flecsi-config.h>

#include "flecsi/data/field_info.hh"
#include "flecsi/flog.hh"
#include "flecsi/util/array_ref.hh"
#include "flecsi/util/serialize.hh"

#include <vector>

namespace flecsi {
namespace util {
/// \addtogroup utils
/// \{

/// A partition of some prefix of the whole numbers into substrings.
template<class D>
struct map_base : with_index_iterator<const D> {
  /// Return a substring.
  constexpr iota_view<std::size_t> operator[](Color c) const {
    return {d()(c), d()(c + 1)};
  }

  /// Return the length of the overall prefix.
  constexpr std::size_t total() const {
    return d()(d().size());
  }

  /// Express a value in terms of its containing substring.
  /// \return the index of the substring and the position of \a i within it
  constexpr std::pair<Color, std::size_t> invert(std::size_t i) const {
    const Color c = d().bin(i);
    return {c, i - d()(c)};
  }

#ifdef DOXYGEN
  /// Return the number of substrings.
  constexpr Color size() const;
  /// Return the start of a substring.
  constexpr std::size_t operator()(Color) const;
  /// Find the substring that contains a value.
  constexpr Color bin(std::size_t) const;
#endif

private:
  constexpr const D & d() const {
    return static_cast<const D &>(*this);
  }
};

// This can be used both for distributing colors over processes and for
// distributing indices over colors (along one structured axis).

/// A partition with substrings of equal size.
/// Initial substrings are 1 longer as needed.
struct equal_map : map_base<equal_map> {
  /// Specify the partition.
  constexpr equal_map(std::size_t size, Color bins)
    : q(size / bins), r(size % bins), n(bins) {}

  constexpr Color size() const {
    return n;
  }

  constexpr std::size_t operator()(Color c) const {
    return c * q + std::min(c, r);
  }

  constexpr Color bin(std::size_t i) const {
    const std::size_t brk = (*this)(r);
    return i < brk ? i / (q + 1) : r + (i - brk) / q;
  }

private:
  std::size_t q;
  Color r, n;
};

/// A partition with substrings of arbitrary size.
struct offsets : map_base<offsets> {
  using storage = std::vector<std::size_t>;

  /// Construct an empty partition.
  offsets() = default;
  /// Construct a partition from specified endpoints.
  /// The first substring starts at an implicit 0.
  offsets(storage e) : end(std::move(e)) {}
  /// Convert an equal mapping.
  offsets(const equal_map & em) {
    end.reserve(em.size());
    for(Color c = 1; c <= em.size(); ++c)
      end.push_back(em(c));
  }

  Color size() const {
    return end.size();
  }

  Color bin(std::size_t i) const {
    return std::upper_bound(end.begin(), end.end(), i) - end.begin();
  }

  std::size_t operator()(Color c) const {
    return c ? end[c - 1] : 0;
  }

  void clear() {
    end.clear();
  }
  void reserve(Color b) {
    end.reserve(b);
  }
  void push_back(std::size_t n) {
    end.push_back(total() + n);
  }

  const storage & ends() const {
    return end;
  }

private:
  storage end;
};

template<>
struct serial::traits<offsets> {
  using type = offsets;
  template<class P>
  static void put(P & p, const type & o) {
    serial::put(p, o.ends());
  }
  static type get(const std::byte *& p) {
    return serial::get<type::storage>(p);
  }
};

/// \}
} // namespace util
} // namespace flecsi

#endif
