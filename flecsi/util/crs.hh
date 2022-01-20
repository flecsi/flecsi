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

struct crs {
  using index_type = std::size_t;
  std::vector<index_type> offsets;
  std::vector<index_type> indices;

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

  void add_row(std::vector<index_type> const & v) {
    add_row(v.begin(), v.end());
  }

  index_type rows() {
    return offsets.size() - 1;
  }

  void clear() {
    offsets.clear();
    indices.clear();
  }
}; // struct crs

struct dcrs : crs {
  using index_type = typename crs::index_type;
  std::vector<index_type> distribution;

  index_type entries() const {
    flog_assert(!offsets.empty(), "attempted to call entries on empty object");
    return offsets.size() - 1;
  }

  index_type colors() {
    return distribution.size() - 1;
  }

  void clear() {
    crs::clear();
    distribution.clear();
  }
}; // struct dcrs

template<typename C>
struct crspan : util::with_index_iterator<const crspan<C>> {

  crspan(C * data) : data_(data) {}

  using index_type = typename C::index_type;
  using span = util::span<index_type>;

  span operator[](index_type i) const {
    flog_assert(data_->offsets.size() - 1 > i, "invalid span index");
    const index_type begin = data_->offsets[i];
    const index_type end = data_->offsets[i + 1];
    return span(&data_->indices[begin], &data_->indices[end]);
  }

  std::size_t size() const {
    return data_->offsets.size() - 1;
  }

  operator C &() {
    return *data_;
  }

private:
  C * data_;
}; // struct crspan

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
  stream << std::endl;

  return stream;
} // operator<<

inline std::ostream &
operator<<(std::ostream & stream, dcrs const & graph) {
  stream << "distribution: ";
  for(auto o : graph.distribution) {
    stream << o << " ";
  }
  stream << std::endl;

  stream << static_cast<crs>(graph);

  return stream;
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
    return type{r, r};
  }
};

template<>
struct util::serial::traits<util::dcrs> {
  using type = util::dcrs;
  template<class P>
  static void put(P & p, const type & d) {
    serial::put(p, static_cast<util::crs>(d));
    serial::put(p, d.distribution);
  }
  static type get(const std::byte *& p) {
    const cast r{p};
    return type{r, r};
  }
};

} // namespace flecsi
