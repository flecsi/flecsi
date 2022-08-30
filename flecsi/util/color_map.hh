// Copyright (c) 2016, Triad National Security, LLC
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

template<class D>
struct map_base {
  constexpr iota_view<std::size_t> operator[](Color c) const {
    return {d()(c), d()(c + 1)};
  }

  constexpr std::size_t total() const {
    return d()(d().size());
  }

  constexpr std::pair<Color, std::size_t> invert(std::size_t i) const {
    const Color c = d().bin(i);
    return {c, i - d()(c)};
  }

private:
  constexpr const D & d() const {
    return static_cast<const D &>(*this);
  }
};

// This can be used both for distributing colors over processes and for
// distributing indices over colors (along one structured axis).
struct equal_map : map_base<equal_map> {
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

struct offsets : map_base<offsets> {
  using storage = std::vector<std::size_t>;

  offsets() = default;
  offsets(storage e) : end(std::move(e)) {}
  offsets(std::size_t size, Color bins) {
    end.reserve(bins);
    const equal_map em(size, bins);
    for(Color c = 1; c <= bins; ++c)
      end.push_back(em(c));
  }
  offsets(const equal_map & em) : offsets(em.total(), em.size()) {}

  Color size() const {
    return end.size();
  }

  Color bin(std::size_t i) const {
    return std::upper_bound(end.begin(), end.end(), i) - end.begin();
  }

  std::size_t operator()(Color c) const {
    return c ? end[c - 1] : 0;
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
