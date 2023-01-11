// Copyright (C) 2016, Triad National Security, LLC
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
  using span = util::span<const util::gid>;

  /// The rows in \c values.
  util::offsets offsets;
  /// The concatenated rows.
  std::vector<util::gid> values;

  /// Create an empty sequence of sequences.
  crs() = default;

  crs(util::offsets os, std::vector<util::gid> vs)
    : offsets(std::move(os)), values(std::move(vs)) {}

  template<class InputIt>
  void add_row(InputIt first, InputIt last) {
    offsets.push_back(std::distance(first, last));
    values.insert(values.end(), first, last);
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
    return offsets.size();
  }

  void clear() {
    offsets.clear();
    values.clear();
  }

  /// Return a row.
  /// \return substring of \c values
  span operator[](std::size_t i) const {
    const auto r = offsets[i];
    return span(values.data() + *r.begin(), r.size());
  }
}; // struct crs

inline std::string
expand(crs const & graph) {
  std::stringstream stream;
  std::size_t r{0};
  for(const crs::span row : graph) {
    stream << r++ << ": <";
    bool first = true;
    for(const std::size_t i : row) {
      if(first)
        first = false;
      else
        stream << ",";
      stream << i;
    }
    stream << ">" << std::endl;
  }
  return stream.str();
}

inline std::ostream &
operator<<(std::ostream & stream, crs const & graph) {
  stream << "crs offsets: ";
  for(auto o : graph.offsets.ends())
    stream << o << " ";
  stream << "\n\ncrs indices: ";
  for(auto o : graph.values) {
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
    serial::put(p, c.offsets, c.values);
  }
  static type get(const std::byte *& p) {
    const cast r{p};
    return type{r, r};
  }
};

} // namespace flecsi

#endif
