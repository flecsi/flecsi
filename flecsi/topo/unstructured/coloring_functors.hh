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
#include "flecsi/util/color_map.hh"
#include "flecsi/util/crs.hh"

#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <vector>

namespace flecsi {
namespace topo {
namespace unstructured_impl {

template<typename Definition>
struct pack_cells {
  pack_cells(Definition const & md, std::vector<std::size_t> const & dist)
    : md_(md), dist_(dist) {}

  auto operator()(int rank, int) const {
    util::crs e2v;

    for(size_t i{dist_[rank]}; i < dist_[rank + 1]; ++i) {
      e2v.add_row(md_.entities(Definition::dimension(), 0, i));
    } // for

    return e2v;
  } // operator(int, int)

private:
  Definition const & md_;
  std::vector<std::size_t> const & dist_;
}; // struct pack_cells

template<typename Definition>
struct pack_vertices {
  pack_vertices(Definition const & md, std::vector<std::size_t> const & dist)
    : md_(md), dist_(dist) {}

  auto operator()(int rank, int) const {
    std::vector<typename Definition::point> vertices;
    vertices.reserve(dist_[rank + 1] - dist_[rank]);

    for(size_t i{dist_[rank]}; i < dist_[rank + 1]; ++i) {
      vertices.emplace_back(md_.vertex(i));
    } // for

    return vertices;
  } // operator(int, int)

private:
  Definition const & md_;
  std::vector<std::size_t> const & dist_;
}; // struct pack_vertices

/*
  Send partial vertex-to-cell connectivity information to the rank that owns
  the vertex in the naive vertex distribution.
 */

struct vertex_referencers {
  vertex_referencers(
    std::map<std::size_t, std::vector<std::size_t>> const & vertex2cell,
    std::vector<std::size_t> const & dist,
    int rank)
    : size_(dist.size() - 1) {
    references_.resize(size_);
    for(auto v : vertex2cell) {
      auto r = util::distribution_offset(dist, v.first);
      if(r != rank) {
        for(auto c : v.second) {
          references_[r][v.first].emplace_back(c);
        } // for
      } // if
    } // for
  } // vertex_refernces

  auto operator()(int rank, int) const {
    flog_assert(rank < size_, "invalid rank");
    return references_[rank];
  } // operator(int, int)

private:
  const int size_;
  std::vector<std::map<std::size_t, std::vector<std::size_t>>> references_;
}; // struct vertex_referencers

/*
  Send full vertex-to-cell connectivity information to all ranks that reference
  one of our vertices.
 */

struct cell_connectivity {
  cell_connectivity(std::vector<std::vector<std::size_t>> const & vertices,
    std::map<std::size_t, std::vector<std::size_t>> const & connectivity,
    std::vector<std::size_t> const & dist,
    int rank)
    : size_(dist.size() - 1), connectivity_(size_) {

    int ro{0};
    for(auto r : vertices) {
      if(ro != rank) {
        for(auto v : r) {
          connectivity_[ro][v] = connectivity.at(v);
        } // for
      } // for
      ++ro;
    } // for
  } // cell_connectivity

  auto operator()(int rank, int) const {
    flog_assert(rank < size_, "invalid rank");
    return connectivity_[rank];
  } // operator(int, int)

private:
  const int size_;
  std::vector<std::map<std::size_t, std::vector<std::size_t>>> connectivity_;
}; // struct cell_connectivity

/*
  Send cells to colors that own them.
 */

struct distribute_cells {
  distribute_cells(util::dcrs const & naive,
    Color colors,
    std::vector<Color> const & index_colors,
    int rank)
    : size_(naive.distribution.size() - 1) {
    util::color_map cm(size_, colors, naive.distribution.back());

    for(std::size_t r{0}; r < std::size_t(size_); ++r) {
      std::vector<std::array<std::size_t, 2>> indices;

      for(std::size_t i{0}; i < naive.size(); ++i) {
        if(cm.process(index_colors[i]) == r) {
          indices.push_back({index_colors[i] /* color of this index */,
            naive.distribution[rank] + i /* index global id */});
        } // if
      } // for

      cells_.emplace_back(indices);
    } // for
  } // distribute_cells

  auto operator()(int rank, int) const {
    flog_assert(rank < size_, "invalid rank");
    return cells_[rank];
  }

private:
  const int size_;
  std::vector<std::vector<std::array<std::size_t, 2>>> cells_;
}; // struct distribute_cells

/*
  Move cells to the colors that own them.
 */

struct migrate_cells {
  // clang-format off
  using return_type =
    std::tuple<
      std::vector</* over cells */
        std::tuple<
          std::array<std::size_t, 2> /* color, mesh id> */,
          std::vector<std::size_t> /* cell definition (vertex mesh ids) */
        >
      >,
      std::map</* over vertices */
        std::size_t, /* mesh id */
        std::vector<std::size_t> /* vertex-to-cell connectivity */
      >,
      std::map</* over cells */
        std::size_t, /* mesh id */
        std::vector<std::size_t> /* cell-to-cell connectivity */
      >
    >;
  // clang-format on

  migrate_cells(util::dcrs const & naive,
    Color colors,
    std::vector<Color> const & index_colors,
    util::crs & e2v,
    std::map<std::size_t, std::vector<std::size_t>> & v2c,
    std::map<std::size_t, std::vector<std::size_t>> & c2c,
    int rank)
    : size_(naive.distribution.size() - 1) {
    util::color_map cm(size_, colors, naive.distribution.back());

    for(std::size_t r{0}; r < std::size_t(size_); ++r) {
      std::vector<
        std::tuple<std::array<std::size_t, 2>, std::vector<std::size_t>>>
        cell_pack;
      std::map<std::size_t, std::vector<std::size_t>> v2c_pack;
      std::map<std::size_t, std::vector<std::size_t>> c2c_pack;

      for(std::size_t i{0}; i < naive.size(); ++i) {
        if(cm.process(index_colors[i]) == r) {
          const std::array<std::size_t, 2> info{
            index_colors[i], naive.distribution[rank] + i};
          cell_pack.push_back(std::make_tuple(info, to_vector(e2v[i])));

          /*
            If we have full connectivity information, we pack it up
            and send it. We do not send information that is potentially,
            or actually incomplete because additional communication
            will be required to resolve it regardless.
           */

          for(auto const & v : e2v[i]) {
            v2c_pack[v] = v2c[v];
          } // for

          c2c_pack[naive.distribution[rank] + i] =
            c2c[naive.distribution[rank] + i];

          /*
            Remove information that we are migrating. We can't remove
            vertex-to-cell information until the loop over ranks is done.
           */

          c2c.erase(i);
        } // if
      } // for

      packs_.emplace_back(std::make_tuple(cell_pack, v2c_pack, c2c_pack));
    } // for

    e2v.clear();
    c2c.clear();
    v2c.clear();
  } // migrate_cells

  return_type operator()(int rank, int) const {
    flog_assert(rank < size_, "invalid rank");
    return packs_[rank];
  }

private:
  const int size_;
  std::vector<return_type> packs_;
}; // struct migrate_cells

struct communicate_entities {
  // clang-format off
  using return_type =
    std::tuple<
      std::vector</* over entities */
        std::tuple<
          std::array<std::size_t, 2> /* color, mesh id> */,
          std::vector<std::size_t> /* entity definition (vertex mesh ids) */
        >
      >,
      std::map</* over vertices */
        std::size_t, /* mesh id */
        std::vector<std::size_t> /* vertex-to-entity connectivity */
      >,
      std::map</* over entities */
        std::size_t, /* mesh id */
        std::vector<std::size_t> /* entity-to-entity connectivity */
      >
    >;
  // clang-format on

  communicate_entities(std::vector<std::vector<std::size_t>> const & entities,
    std::unordered_map<std::size_t, Color> const & colors,
    util::crs const & e2v,
    std::map<std::size_t, std::vector<std::size_t>> const & v2e,
    std::map<std::size_t, std::vector<std::size_t>> const & e2e,
    std::map<std::size_t, std::size_t> const & m2p)
    : size_(entities.size()) {
    for(auto re : entities) {
      std::vector<
        std::tuple<std::array<std::size_t, 2>, std::vector<std::size_t>>>
        entity_pack;
      std::map<std::size_t, std::vector<std::size_t>> v2e_pack;
      std::map<std::size_t, std::vector<std::size_t>> e2e_pack;

      for(auto c : re) {
        const std::array<std::size_t, 2> info{colors.at(c), c};
        entity_pack.push_back(std::make_tuple(info, to_vector(e2v[m2p.at(c)])));

        for(auto const & v : e2v[m2p.at(c)]) {
          v2e_pack[v] = v2e.at(v);
        } // for

        e2e_pack[c] = e2e.at(c);
      } // for

      packs_.emplace_back(std::make_tuple(entity_pack, v2e_pack, e2e_pack));
    } // for
  } // communicate_entities

  return_type operator()(int rank, int) const {
    flog_assert(rank < size_, "invalid rank");
    return packs_[rank];
  }

private:
  const int size_;
  std::vector<return_type> packs_;
}; // struct communicate_entities

#if 0
template<typename Definition>
struct communicate_coordinates {
  using return_type = 

private:
  std::vector<return_type> packs_;
}; // struct communicate_coordinates
#endif

struct vertex_coloring {

  vertex_coloring(std::vector<std::size_t> const & pdist,
    std::unordered_map<std::size_t, Color> const & v2co)
    : size_(pdist.size() - 1), vertices_(size_) {

    for(auto const & v : v2co) {
      std::size_t r{std::numeric_limits<std::size_t>::max()};
      for(std::size_t i{0}; i < std::size_t(size_); ++i) {
        if(v.first < pdist[i + 1]) {
          r = i;
          break;
        } // if
      } // for

      vertices_[r].emplace_back(std::array<std::size_t, 2>{v.first, v.second});
    } // for
  } // vertex_coloring

  auto operator()(int rank, int) const {
    flog_assert(rank < size_, "invalid rank");
    return vertices_[rank];
  }

private:
  const int size_;
  std::vector<std::vector<std::array<std::size_t, 2>>> vertices_;
}; // struct vertex_coloring

template<typename Definition>
struct migrate_vertices {

  using return_type = std::vector</* over vertices */
    std::tuple<std::array<std::size_t, 2> /* color, mesh id */,
      typename Definition::point /* coordinates */
      >>;

  migrate_vertices(std::vector<std::size_t> dist,
    Color colors,
    std::vector<Color> const & index_colors,
    std::vector<typename Definition::point> & vertices,
    int rank)
    : size_(dist.size() - 1) {
    util::color_map cm(size_, colors, dist.back());

    for(std::size_t r{0}; r < std::size_t(size_); ++r) {
      return_type vertex_pack;
      for(std::size_t i{0}; i < dist[rank + 1] - dist[rank]; ++i) {
        if(cm.process(index_colors[i]) == r) {
          const std::array<std::size_t, 2> info{
            index_colors[i], dist[rank] + i};
          vertex_pack.push_back(std::make_tuple(info, vertices[i]));
        } // if
      } // for

      packs_.emplace_back(vertex_pack);
    } // for
  } // migrate_vertices

  return_type operator()(int rank, int) const {
    flog_assert(std::size_t(rank) < size_, "invalid rank");
    return packs_[rank];
  }

private:
  std::size_t size_;
  std::vector<return_type> packs_;
}; // struct migrate_vertices

} // namespace unstructured_impl
} // namespace topo
} // namespace flecsi
