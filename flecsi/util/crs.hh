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

/// Efficient (compressed-row) storage for a sequence of sequences of
/// integers.  There are no constraints on the size or contents of either
/// the sequences or sequences of sequences: sequences can be different
/// sizes, can have overlapping values, can include values in any order,
/// and can hold duplicate values.
///
/// This type is a random-access range of \c span objects.  Because the \c
/// offsets and \c values fields must be kept in sync they are best treated
/// as read-only.  Use the \c add_row methods to modify those fields in a
/// consistent manner.
struct crs : util::with_index_iterator<const crs> {
  /// The rows in \c values.
  util::offsets offsets;
  /// The concatenated rows.
  std::vector<util::gid> values;

  /// Create an empty sequence of sequences.
  crs() = default;

  crs(util::offsets os, std::vector<util::gid> vs)
    : offsets(std::move(os)), values(std::move(vs)) {}

  /// Append onto the \c crs (via data copy) a row of values pointed to by
  /// a beginning and an ending iterator.
  template<class InputIt>
  void add_row(InputIt first, InputIt last) {
    offsets.push_back(std::distance(first, last));
    values.insert(values.end(), first, last);
  }

  /// Append onto the \c crs (via data copy) a row of values.
  template<class U>
  void add_row(std::initializer_list<U> init) {
    add_row(init.begin(), init.end());
  }

  /// Append onto the \c crs (via data copy) a row of values acquired by
  /// traversing a range.
  template<typename Range>
  void add_row(Range const & it) {
    add_row(it.begin(), it.end());
  }

  /// Return the number of rows.
  std::size_t size() const {
    return offsets.size();
  }

  /// Discard all data (\c offsets and \c values).
  void clear() {
    offsets.clear();
    values.clear();
  }

  /// Return a row.
  /// \return substring of \c values
  /// \{
  util::span<const util::gid> operator[](std::size_t i) const {
    return const_cast<crs &>(*this)[i];
  }
  util::span<util::gid> operator[](std::size_t i) {
    const auto r = offsets[i];
    return {values.data() + *r.begin(), r.size()};
  }
  /// \}
}; // struct crs

inline std::string
expand(crs const & graph) {
  std::stringstream stream;
  std::size_t r{0};
  for(const auto row : graph) {
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
