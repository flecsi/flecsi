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
auto
pack_definitions(D const & md, entity_kind id, const util::equal_map & dist) {
  return [&, id](Color rank) {
    util::crs e2v; // entity to vertex graph

    for(const std::size_t i : dist[rank]) {
      e2v.add_row(md.entities(id, 0, i));
    } // for

    return e2v;
  };
}

template<typename T>
auto
pack_field(const util::equal_map & dist, const std::vector<T> & field) {
  return [&](Color rank) {
    std::vector<std::pair<std::size_t, T>> field_pack;
    const auto dr = dist[rank];
    field_pack.reserve(dr.size());
    for(const std::size_t i : dr) {
      field_pack.emplace_back(i, field[i]);
    } // for
    return field_pack;
  };
}

/*
  Send partial vertex-to-cell connectivity information to the rank that owns
  the vertex in the naive vertex distribution.
 */

inline auto
vertex_referencers(
  std::map<util::gid, std::vector<util::gid>> const & vertex2cell,
  const util::equal_map & dist,
  int rank) {
  std::vector<std::map<util::gid, std::vector<util::gid>>> ret(dist.size());
  for(auto & v : vertex2cell) {
    auto r = dist.bin(v.first);
    if(int(r) != rank) {
      for(auto c : v.second) {
        ret[r][v.first].emplace_back(c);
      } // for
    } // if
  } // for
  return ret;
}

/*
  Send full vertex-to-cell connectivity information to all ranks that reference
  one of our vertices.
 */

inline auto
entity_connectivity(std::vector<std::vector<util::gid>> const & vertices,
  std::map<util::gid, std::vector<util::gid>> const & connectivity,
  const util::equal_map & dist,
  int rank) {
  std::vector<std::map<util::gid, std::vector<util::gid>>> ret(dist.size());

  int ro{0};
  for(auto & r : vertices) {
    if(ro != rank) {
      for(auto v : r) {
        ret[ro][v] = connectivity.at(v);
      } // for
    } // for
    ++ro;
  } // for
  return ret;
} // entity_connectivity

/*
  Move entity connectivity information to the colors that own them.
 */

inline auto
move_primaries(const util::offsets & dist,
  Color colors,
  std::vector<Color> const & index_colors,
  util::crs && e2v,
  std::map<util::gid, std::vector<util::gid>> && v2e,
  int rank) {
  return [&,
           rank,
           e2v = std::move(e2v),
           v2e = std::move(v2e),
           em = util::equal_map(colors, dist.size())](Color r) {
    std::vector<std::tuple<std::pair<Color, util::gid>, std::vector<util::gid>>>
      cell_pack;
    std::map<util::gid, std::vector<util::gid>> v2e_pack;

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
          v2e_pack.try_emplace(v, v2e.at(v));
        } // for
      } // if
    } // for

    return std::make_pair(std::move(cell_pack), std::move(v2e_pack));
  };
} // move_primaries

inline auto
communicate_entities(std::vector<std::vector<util::gid>> const & entities,
  std::unordered_map<util::gid, std::set<Color>> const & deps,
  std::unordered_map<util::gid, Color> const & colors,
  util::crs const & e2v,
  std::map<util::gid, std::vector<util::gid>> const & v2e,
  std::map<util::gid, util::id> const & m2p) {
  return [&](std::size_t i) {
    std::map<Color,
      std::vector</* over entities */
        std::tuple<util::gid,
          std::vector<util::gid> /* entity definition (vertex mesh ids) */,
          std::set<Color> /* dependents */
          >>>
      entity_pack;
    std::map</* over vertices */
      util::gid, /* mesh id */
      std::vector<util::gid> /* vertex-to-entity connectivity */
      >
      v2e_pack;

    for(auto c : entities[i]) {
      entity_pack[colors.at(c)].push_back(std::make_tuple(c,
        to_vector(e2v[m2p.at(c)]),
        deps.count(c) ? deps.at(c) : std::set<Color>{}));

      for(auto const & v : e2v[m2p.at(c)]) {
        v2e_pack[v] = v2e.at(v);
      } // for
    } // for

    return std::make_pair(std::move(entity_pack), std::move(v2e_pack));
  };
} // communicate_entities

inline auto
rank_coloring(const util::equal_map & pdist,
  std::unordered_map<util::gid, Color> const & v2co) {
  std::vector<std::vector<std::pair<util::gid, Color>>> ret(pdist.size());
  for(auto const & v : v2co) {
    ret[pdist.bin(v.first)].push_back({v.first, v.second});
  } // for
  return ret;
} // rank_coloring

template<typename T>
auto
move_field(std::size_t size,
  Color colors,
  const std::vector<Color> & index_colors,
  const std::vector<T> & field) {
  return [&index_colors, &field, em = util::equal_map(colors, size)](int rank) {
    std::vector<T> field_pack;
    int i = 0;
    for(const auto & v : index_colors) {
      if(int(em.bin(v)) == rank) {
        field_pack.push_back(field[i]);
      } // if
      ++i;
    } // for
    return field_pack;
  };
}

} // namespace unstructured_impl
} // namespace topo
} // namespace flecsi

#endif
