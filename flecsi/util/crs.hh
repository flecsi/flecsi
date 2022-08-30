// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_CRS_HH
#define FLECSI_UTIL_CRS_HH

#include "flecsi/flog.hh"
#include "flecsi/util/array_ref.hh"
#include "flecsi/util/color_map.hh"
#include "flecsi/util/serialize.hh"

#include <algorithm>
#include <iterator>
#include <ostream>
#include <vector>

namespace flecsi {
namespace util {
/// \addtogroup utils
/// \{

template<typename T, typename U>
std::vector<T>
as(std::vector<U> const & v) {
  return {v.begin(), v.end()};
} // as

/// Efficient storage for a sequence of sequences of integers.
struct crs : util::with_index_iterator<const crs> {
  using span = util::span<const std::size_t>;

  /// The beginning of each row in \c indices, including a trailing value that
  /// is the end of the last row.
  std::vector<std::size_t> offsets;
  /// The concatenated rows.
  std::vector<std::size_t> indices;

  template<class InputIt>
  void add_row(InputIt first, InputIt last) {
    if(offsets.empty())
      offsets.emplace_back(0);
    offsets.emplace_back(offsets.back() + std::distance(first, last));
    indices.insert(indices.end(), first, last);
  }

  template<class U>
  void add_row(std::initializer_list<U> init) {
    add_row(init.begin(), init.end());
  }

  void add_row(std::vector<std::size_t> const & v) {
    add_row(v.begin(), v.end());
  }

  std::size_t size() const {
    flog_assert(!offsets.empty(), "attempted to call entries on empty object");
    return offsets.size() - 1;
  }

  void clear() {
    offsets.clear();
    indices.clear();
  }

  span operator[](std::size_t i) const {
    flog_assert(offsets.size() - 1 > i, "invalid span index");
    const std::size_t begin = offsets[i];
    const std::size_t end = offsets[i + 1];
    return span(&indices[begin], &indices[end]);
  }
}; // struct crs

struct dcrs : crs {
  util::offsets distribution;

  std::size_t colors() {
    return distribution.size();
  }

  void clear() {
    crs::clear();
    distribution = {};
  }
}; // struct dcrs

inline std::ostream &
operator<<(std::ostream & stream, crs const & graph) {
  stream << "offsets: ";
  for(auto o : graph.offsets) {
    stream << o << " ";
  }
  stream << std::endl;

  stream << "indices: ";
  for(auto o : graph.indices) {
    stream << o << " ";
  }
  return stream << std::endl;
} // operator<<

inline std::ostream &
operator<<(std::ostream & stream, dcrs const & graph) {
  stream << "distribution: ";
  for(auto o : graph.distribution.ends()) {
    stream << o << " ";
  }
  return stream << std::endl << static_cast<const crs &>(graph);
} // operator<<

/// \}
} // namespace util

template<>
struct util::serial::traits<util::crs> {
  using type = util::crs;
  template<class P>
  static void put(P & p, const type & c) {
    serial::put(p, c.offsets, c.indices);
  }
  static type get(const std::byte *& p) {
    const cast r{p};
    return type{{}, r, r};
  }
};

template<>
struct util::serial::traits<util::dcrs> {
  using type = util::dcrs;
  template<class P>
  static void put(P & p, const type & d) {
    serial::put(p, static_cast<util::crs const &>(d), d.distribution);
  }
  static type get(const std::byte *& p) {
    const cast r{p};
    return type{r, r};
  }
};

} // namespace flecsi

#endif
