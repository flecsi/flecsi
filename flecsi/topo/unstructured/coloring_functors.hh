// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_UNSTRUCTURED_COLORING_FUNCTORS_HH
#define FLECSI_TOPO_UNSTRUCTURED_COLORING_FUNCTORS_HH

#include "flecsi/flog.hh"
#include "flecsi/topo/unstructured/types.hh"
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

using entity_kind = std::size_t;

template<typename D>
struct pack_definitions {
  pack_definitions(D const & md, entity_kind id, const util::equal_map & dist)
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
  entity_kind const id_;
  const util::equal_map & dist_;
}; // struct pack_definitions

template<typename T>
struct pack_field {
  pack_field(const util::equal_map & dist, const std::vector<T> & field)
    : dist_(dist), field_(field) {}

  auto operator()(int rank, int) const {
    std::vector<std::pair<std::size_t, T>> field_pack;
    const auto dr = dist_[rank];
    field_pack.reserve(dr.size());
    for(const std::size_t i : dr) {
      field_pack.emplace_back(i, field_[i]);
    } // for
    return field_pack;
  } // operator(int, int)

private:
  const util::equal_map & dist_;
  const std::vector<T> & field_;
}; // struct pack_field

/*
  Send partial vertex-to-cell connectivity information to the rank that owns
  the vertex in the naive vertex distribution.
 */

struct vertex_referencers {
  vertex_referencers(
    std::map<util::gid, std::vector<util::gid>> const & vertex2cell,
    const util::equal_map & dist,
    int rank)
    : size_(dist.size()) {
    references_.resize(size_);
    for(auto & v : vertex2cell) {
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
  std::vector<std::map<util::gid, std::vector<util::gid>>> references_;
}; // struct vertex_referencers

/*
  Send full vertex-to-cell connectivity information to all ranks that reference
  one of our vertices.
 */

struct entity_connectivity {
  entity_connectivity(std::vector<std::vector<util::gid>> const & vertices,
    std::map<util::gid, std::vector<util::gid>> const & connectivity,
    const util::equal_map & dist,
    int rank)
    : size_(dist.size()), connectivity_(size_) {

    int ro{0};
    for(auto & r : vertices) {
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
  std::vector<std::map<util::gid, std::vector<util::gid>>> connectivity_;
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
          std::pair<Color, util::gid>,
          std::vector<util::gid> /* entity definition (vertex mesh ids) */
        >
      >,
      std::map</* vertex-to-entity connectivity map */
        util::gid,
        std::vector<util::gid>
      >,
      std::map</* cell-to-cell connectivity map */
        util::gid,
        std::vector<util::gid>
      >
    >;
  // clang-format on

  move_primaries(const util::offsets & dist,
    Color colors,
    std::vector<Color> const & index_colors,
    util::crs & e2v,
    std::map<util::gid, std::vector<util::gid>> & v2e,
    std::map<util::gid, std::vector<util::gid>> & e2e,
    int rank)
    : size_(dist.size()) {
    const util::equal_map em(colors, size_);

    for(std::size_t r{0}; r < std::size_t(size_); ++r) {
      std::vector<
        std::tuple<std::pair<Color, util::gid>, std::vector<util::gid>>>
        cell_pack;
      std::map<util::gid, std::vector<util::gid>> v2e_pack;
      std::map<util::gid, std::vector<util::gid>> e2e_pack;

      for(std::size_t i{0}; i < dist[rank].size(); ++i) {
        if(em.bin(index_colors[i]) == r) {
          const auto j = dist(rank) + i;
          const std::pair<Color, util::gid> info{index_colors[i], j};
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
          std::pair<Color, util::gid>,
        std::vector<util::gid> /* entity definition (vertex mesh ids) */,
          std::set<Color> /* dependents */
        >
      >,
      std::map</* over vertices */
        util::gid, /* mesh id */
        std::vector<util::gid> /* vertex-to-entity connectivity */
      >,
      std::map</* over entities */
        util::gid, /* mesh id */
        std::vector<util::gid> /* entity-to-entity connectivity */
      >
    >;
  // clang-format on

  communicate_entities(std::vector<std::vector<util::gid>> const & entities,
    std::unordered_map<util::gid, std::set<Color>> const & deps,
    std::unordered_map<util::gid, Color> const & colors,
    util::crs const & e2v,
    std::map<util::gid, std::vector<util::gid>> const & v2e,
    std::map<util::gid, std::vector<util::gid>> const & e2e,
    std::map<util::gid, util::id> const & m2p)
    : size_(entities.size()) {
    for(auto re : entities) {
      std::vector<std::tuple<std::pair<Color, util::gid>,
        std::vector<util::gid>,
        std::set<Color>>>
        entity_pack;
      std::map<util::gid, std::vector<util::gid>> v2e_pack;
      std::map<util::gid, std::vector<util::gid>> e2e_pack;

      for(auto c : re) {
        const std::pair<Color, util::gid> info{colors.at(c), c};
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
    std::unordered_map<util::gid, Color> const & v2co)
    : size_(pdist.size()), vertices_(size_) {

    for(auto const & v : v2co) {
      vertices_[pdist.bin(v.first)].push_back({v.first, v.second});
    } // for
  } // rank_coloring

  auto operator()(int rank, int) const {
    flog_assert(rank < size_, "invalid rank");
    return vertices_[rank];
  }

private:
  const int size_;
  std::vector<std::vector<std::pair<util::gid, Color>>> vertices_;
}; // struct rank_coloring

template<typename T>
struct move_field {

  move_field(std::size_t size,
    Color colors,
    const std::vector<Color> & index_colors,
    const std::vector<T> & field)
    : em_(colors, size), index_colors_(index_colors), field_(field) {}

  std::vector<T> operator()(int rank, int) const {
    std::vector<T> field_pack;
    int i = 0;
    for(const auto & v : index_colors_) {
      if(em_.bin(v) == static_cast<unsigned int>(rank)) {
        field_pack.push_back(field_[i]);
      } // if
      ++i;
    } // for
    return field_pack;
  } // operator(int, int)

private:
  const util::equal_map em_;
  const std::vector<Color> & index_colors_;
  const std::vector<T> & field_;
}; // struct move_field

} // namespace unstructured_impl
} // namespace topo
} // namespace flecsi

#endif
