// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_UNSTRUCTURED_COLORING_FUNCTORS_HH
#define FLECSI_TOPO_UNSTRUCTURED_COLORING_FUNCTORS_HH

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

template<typename D>
struct pack_definitions {
  pack_definitions(D const & md, std::size_t id, const util::equal_map & dist)
    : md_(md), id_(id), dist_(dist) {}

  auto operator()(int rank, int) const {
    util::crs e2v; // entity to vertex graph

    for(const std::size_t i : dist_[rank]) {
      e2v.add_row(md_.entities(id_, 0, i));
    } // for

    return e2v;
  } // operator(int, int)

private:
  D const & md_;
  std::size_t const id_;
  const util::equal_map & dist_;
}; // struct pack_definitions

template<typename Definition>
struct pack_vertices {
  pack_vertices(Definition const & md, const util::equal_map & dist)
    : md_(md), dist_(dist) {}

  auto operator()(int rank, int) const {
    std::vector<typename Definition::point> vertices;
    const auto dr = dist_[rank];
    vertices.reserve(dr.size());

    for(const std::size_t i : dr) {
      vertices.emplace_back(md_.vertex(i));
    } // for

    return vertices;
  } // operator(int, int)

private:
  Definition const & md_;
  const util::equal_map & dist_;
}; // struct pack_vertices

/*
  Send partial vertex-to-cell connectivity information to the rank that owns
  the vertex in the naive vertex distribution.
 */

struct vertex_referencers {
  vertex_referencers(
    std::map<std::size_t, std::vector<std::size_t>> const & vertex2cell,
    const util::equal_map & dist,
    int rank)
    : size_(dist.size()) {
    references_.resize(size_);
    for(auto v : vertex2cell) {
      auto r = dist.bin(v.first);
      if(int(r) != rank) {
        for(auto c : v.second) {
          references_[r][v.first].emplace_back(c);
        } // for
      } // if
    } // for
  } // vertex_refernces

  auto & operator()(int rank, int) const {
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

struct entity_connectivity {
  entity_connectivity(std::vector<std::vector<std::size_t>> const & vertices,
    std::map<std::size_t, std::vector<std::size_t>> const & connectivity,
    const util::equal_map & dist,
    int rank)
    : size_(dist.size()), connectivity_(size_) {

    int ro{0};
    for(auto r : vertices) {
      if(ro != rank) {
        for(auto v : r) {
          connectivity_[ro][v] = connectivity.at(v);
        } // for
      } // for
      ++ro;
    } // for
  } // entity_connectivity

  auto & operator()(int rank, int) const {
    flog_assert(rank < size_, "invalid rank");
    return connectivity_[rank];
  } // operator(int, int)

private:
  const int size_;
  std::vector<std::map<std::size_t, std::vector<std::size_t>>> connectivity_;
}; // struct entity_connectivity

/*
  Move entity connectivity information to the colors that own them.
 */

struct move_primaries {
  // clang-format off
  using return_type =
    std::tuple<
      std::vector</* over cells */
        std::tuple<
          std::array<std::size_t, 2> /* color, mesh id> */,
          std::vector<std::size_t> /* entity definition (vertex mesh ids) */
        >
      >,
      std::map</* over vertices */
        std::size_t, /* mesh id */
        std::vector<std::size_t> /* vertex-to-entity connectivity */
      >,
      std::map</* over cells */
        std::size_t, /* mesh id */
        std::vector<std::size_t> /* cell-to-cell connectivity */
      >
    >;
  // clang-format on

  move_primaries(const util::offsets & dist,
    Color colors,
    std::vector<Color> const & index_colors,
    util::crs & e2v,
    std::map<std::size_t, std::vector<std::size_t>> & v2e,
    std::map<std::size_t, std::vector<std::size_t>> & e2e,
    int rank)
    : size_(dist.size()) {
    const util::equal_map em(colors, size_);

    for(std::size_t r{0}; r < std::size_t(size_); ++r) {
      std::vector<
        std::tuple<std::array<std::size_t, 2>, std::vector<std::size_t>>>
        cell_pack;
      std::map<std::size_t, std::vector<std::size_t>> v2e_pack;
      std::map<std::size_t, std::vector<std::size_t>> e2e_pack;

      for(std::size_t i{0}; i < dist[rank].size(); ++i) {
        if(em.bin(index_colors[i]) == r) {
          const auto j = dist(rank) + i;
          const std::array<std::size_t, 2> info{index_colors[i], j};
          cell_pack.push_back(std::make_tuple(info, to_vector(e2v[i])));

          /*
            If we have full connectivity information, we pack it up
            and send it. We do not send information that is potentially,
            or actually incomplete because additional communication
            will be required to resolve it regardless.
           */

          for(auto const & v : e2v[i]) {
            v2e_pack[v] = v2e[v];
          } // for

          e2e_pack[j] = e2e[j];

          /*
            Remove information that we are migrating. We can't remove
            vertex-to-cell information until the loop over ranks is done.
           */

          e2e.erase(i);
        } // if
      } // for

      packs_.emplace_back(std::make_tuple(cell_pack, v2e_pack, e2e_pack));
    } // for

    e2v.clear();
    e2e.clear();
    v2e.clear();
  } // move_primaries

  const return_type & operator()(int rank, int) const {
    flog_assert(rank < size_, "invalid rank");
    return packs_[rank];
  }

private:
  const int size_;
  std::vector<return_type> packs_;
}; // struct move_primaries

struct communicate_entities {
  // clang-format off
  using return_type =
    std::tuple<
      std::vector</* over entities */
        std::tuple<
          std::array<std::size_t, 2> /* color, mesh id> */,
          std::vector<std::size_t> /* entity definition (vertex mesh ids) */,
          std::set<Color> /* dependents */
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
    std::unordered_map<std::size_t, std::set<Color>> const & deps,
    std::unordered_map<std::size_t, Color> const & colors,
    util::crs const & e2v,
    std::map<std::size_t, std::vector<std::size_t>> const & v2e,
    std::map<std::size_t, std::vector<std::size_t>> const & e2e,
    std::map<std::size_t, std::size_t> const & m2p)
    : size_(entities.size()) {
    for(auto re : entities) {
      std::vector<std::tuple<std::array<std::size_t, 2>,
        std::vector<std::size_t>,
        std::set<Color>>>
        entity_pack;
      std::map<std::size_t, std::vector<std::size_t>> v2e_pack;
      std::map<std::size_t, std::vector<std::size_t>> e2e_pack;

      for(auto c : re) {
        const std::array<std::size_t, 2> info{colors.at(c), c};
        entity_pack.push_back(std::make_tuple(info,
          to_vector(e2v[m2p.at(c)]),
          deps.count(c) ? deps.at(c) : std::set<Color>{}));

        for(auto const & v : e2v[m2p.at(c)]) {
          v2e_pack[v] = v2e.at(v);
        } // for

        e2e_pack[c] = e2e.at(c);
      } // for

      packs_.emplace_back(std::make_tuple(entity_pack, v2e_pack, e2e_pack));
    } // for
  } // communicate_entities

  const return_type & operator()(int rank, int) const {
    flog_assert(rank < size_, "invalid rank");
    return packs_[rank];
  }

private:
  const int size_;
  std::vector<return_type> packs_;
}; // struct communicate_entities

struct rank_coloring {

  rank_coloring(const util::equal_map & pdist,
    std::unordered_map<std::size_t, Color> const & v2co)
    : size_(pdist.size()), vertices_(size_) {

    for(auto const & v : v2co) {
      vertices_[pdist.bin(v.first)].emplace_back(
        std::array<std::size_t, 2>{v.first, v.second});
    } // for
  } // rank_coloring

  auto operator()(int rank, int) const {
    flog_assert(rank < size_, "invalid rank");
    return vertices_[rank];
  }

private:
  const int size_;
  std::vector<std::vector<std::array<std::size_t, 2>>> vertices_;
}; // struct rank_coloring

template<typename Definition>
struct move_vertices {

  using return_type = std::vector</* over vertices */
    std::tuple<std::array<std::size_t, 2> /* color, mesh id */,
      typename Definition::point /* coordinates */
      >>;

  move_vertices(const util::equal_map & dist,
    Color colors,
    std::vector<Color> const & index_colors,
    std::vector<typename Definition::point> & vertices,
    int rank)
    : size_(dist.size()) {
    const util::equal_map em(colors, size_);
    const auto v0 = dist(rank);
    const auto n = index_colors.size();

    for(std::size_t r{0}; r < std::size_t(size_); ++r) {
      return_type vertex_pack;
      for(std::size_t i{0}; i < n; ++i) {
        if(em.bin(index_colors[i]) == r) {
          const std::array<std::size_t, 2> info{index_colors[i], v0 + i};
          vertex_pack.push_back(std::make_tuple(info, vertices[i]));
        } // if
      } // for

      packs_.emplace_back(vertex_pack);
    } // for
  } // move_vertices

  return_type operator()(int rank, int) const {
    flog_assert(std::size_t(rank) < size_, "invalid rank");
    return packs_[rank];
  }

private:
  std::size_t size_;
  std::vector<return_type> packs_;
}; // struct move_vertices

} // namespace unstructured_impl
} // namespace topo
} // namespace flecsi

#endif
