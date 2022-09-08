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

  template<typename Range>
  void add_row(Range const & it) {
    add_row(it.begin(), it.end());
  }

  /// Return the number of rows.
  std::size_t size() const {
    flog_assert(offsets.empty() || offsets.size() > 1,
      "attempted to call size on invalid crs object");
    return offsets.empty() ? 0 : offsets.size() - 1;
  }

  void clear() {
    offsets.clear();
    indices.clear();
  }

  /// Return a row.
  /// \return substring of \c indices
  span operator[](std::size_t i) const {
    flog_assert(offsets.size() - 1 > i, "invalid span index");
    const std::size_t begin = offsets[i];
    const std::size_t end = offsets[i + 1];
    return span(&indices[begin], &indices[end]);
  }
}; // struct crs

inline std::string
expand(crs const & graph) {
  std::stringstream stream;
  std::size_t r{0};
  for(auto & o : graph.offsets) {
    if(o != graph.offsets.back()) {
      stream << r++ << ": <";
      auto n = *(&o + 1);
      for(std::size_t i{o}; i < n; ++i) {
        stream << graph.indices[i];
        if(i < n - 1) {
          stream << ",";
        }
      }
      stream << ">" << std::endl;
    }
  }
  return stream.str();
}

inline std::ostream &
operator<<(std::ostream & stream, crs const & graph) {
  stream << "crs offsets: ";
  for(auto o : graph.offsets) {
    stream << o << " ";
  }
  stream << "\n\ncrs indices: ";
  for(auto o : graph.indices) {
    stream << o << " ";
  }
  stream << "\n\ncrs expansion:\n" << expand(graph) << std::endl;
  return stream << std::endl;
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

} // namespace flecsi

#endif
