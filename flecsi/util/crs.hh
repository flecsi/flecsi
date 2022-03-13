/*
    @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
   /@@/////  /@@          @@////@@ @@////// /@@
   /@@       /@@  @@@@@  @@    // /@@       /@@
   /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
   /@@////   /@@/@@@@@@@/@@       ////////@@/@@
   /@@       /@@/@@//// //@@    @@       /@@/@@
   /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
   //       ///  //////   //////  ////////  //

   Copyright (c) 2016, Triad National Security, LLC
   All rights reserved.
                                                                              */
#pragma once

#include "flecsi/flog.hh"
#include "flecsi/util/array_ref.hh"
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

struct crs : util::with_index_iterator<const crs> {
  using span = util::span<const std::size_t>;

  std::vector<std::size_t> offsets;
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
  std::vector<std::size_t> distribution;

  std::size_t colors() {
    return distribution.size() - 1;
  }

  void clear() {
    crs::clear();
    distribution.clear();
  }
}; // struct dcrs

template<typename FI, typename T>
auto
distribution_offset(FI const & distribution, T index) {
  auto it = std::upper_bound(distribution.begin(), distribution.end(), index);
  flog_assert(it != distribution.end(), "index out of range");
  return std::distance(distribution.begin(), it) - 1;
} // distribution_offset

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
  for(auto o : graph.distribution) {
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
