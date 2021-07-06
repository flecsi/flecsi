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

/*! @file */

#include "flecsi/data/field_info.hh"
#include "flecsi/data/topology.hh"
#include "flecsi/execution.hh"
#include "flecsi/topo/index.hh"
#include "flecsi/util/color_map.hh"
#include "flecsi/util/common.hh"
#include "flecsi/util/mpi.hh"
#include "flecsi/util/serialize.hh"

#include <algorithm>
#include <cstddef>
#include <map>
#include <set>
#include <vector>

namespace flecsi {
namespace topo {
namespace unstructured_impl {

struct shared_entity {
  std::size_t id;
  std::vector<std::size_t> dependents;

  /*
    These operators are designed to be used with util::force_unique, which
    first applies std::sort (using operator<), and then applies std::unique
    (using operator==). The implementation of the operators is designed to be
    greedy, i.e., it favors shared entity variants that have more dependents by
    placing them first in the sorted order. When std::unique is applied, the
    variant with the most dependents will be kept.
   */

  bool operator<(const shared_entity & s) const {
    return id == s.id ? dependents.size() > s.dependents.size()
                      : /* greedy dependendents */
             id < s.id;
  }

  bool operator==(const shared_entity & s) const {
    return id == s.id; // ignores dependendents
  }
};

inline std::ostream &
operator<<(std::ostream & stream, shared_entity const & s) {
  stream << "<" << s.id << ": ";
  bool first = true;
  for(auto d : s.dependents) {
    if(first)
      first = false;
    else
      stream << ", ";
    stream << d;
  } // for
  stream << ">";
  return stream;
}

struct ghost_entity {
  std::size_t id;
  Color color;

  bool operator<(const ghost_entity & g) const {
    return id < g.id;
  }

  bool operator==(const ghost_entity & g) const {
    return id == g.id && color == g.color;
  }
};

inline std::ostream &
operator<<(std::ostream & stream, ghost_entity const & g) {
  stream << "<" << g.id << ":" << g.color << ">";
  return stream;
}

struct index_coloring {
  std::vector<std::size_t> all;
  std::vector<std::size_t> owned;
  std::vector<std::size_t> exclusive;
  std::vector<shared_entity> shared;
  std::vector<ghost_entity> ghost;
};

struct crs {
  std::vector<std::size_t> offsets;
  std::vector<std::size_t> indices;

  template<class InputIt>
  void add_row(InputIt first, InputIt last) {
    if(first == last)
      return;
    if(offsets.empty())
      offsets.emplace_back(0);
    offsets.emplace_back(offsets.back() + std::distance(first, last));
    indices.insert(indices.end(), first, last);
  }

  template<class U>
  void add_row(std::initializer_list<U> init) {
    add_row(init.begin(), init.end());
  }
};

/*
  Closure tokens for specifying the behavior of closure function.
 */

template<size_t IndexSpace,
  Dimension D,
  Dimension ThroughDimension,
  size_t Depth = 1>
struct primary_independent {
  static constexpr size_t index_space = IndexSpace;
  static constexpr Dimension dimension = D, thru_dimension = ThroughDimension;
  static constexpr size_t depth = Depth;
}; // struct primary_independent

template<std::size_t IndexSpace, Dimension D, Dimension PrimaryDimension>
struct auxiliary_independent {
  static constexpr size_t index_space = IndexSpace;
  static constexpr Dimension dimension = D,
                             primary_dimension = PrimaryDimension;
}; // struct auxiliary_independent

inline void
transpose(field<util::id, data::ragged>::accessor<ro, na> input,
  field<util::id, data::ragged>::mutator<rw, na> output) {
  for(std::size_t e{0}; e < input.size(); ++e) {
    for(std::size_t v{0}; v < input[e].size(); ++v) {
      output[input[e][v]].push_back(e);
    }
  }
}

struct coloring_definition {
  Color colors;
  std::size_t idx;
  std::size_t dim;
  std::size_t depth;

  std::size_t vidx;

  struct auxiliary {
    std::size_t idx;
    std::size_t dim;
    bool cnx;
  };

  std::vector<auxiliary> aux;
};

} // namespace unstructured_impl

struct unstructured_base {

  using index_coloring = unstructured_impl::index_coloring;
  using ghost_entity = unstructured_impl::ghost_entity;
  using crs = unstructured_impl::crs;

  struct process_color {
    /*
      The current coloring utilities and topology initialization assume
      the use of MPI. This could change in the future, e.g., if legion
      matures to the point of developing its own software stack. However,
      for the time being, this comm is provided to retain consistency
      with the coloring utilities for unstructured.
     */

    MPI_Comm comm;

    /*
      The global id of this color
     */

    Color color;

    /*
      The global number of colors
     */

    Color colors;

    /*
      The global number of entities in each index space
     */

    std::vector<std::size_t> idx_entities;

    /*
      The local coloring information for each index space.

      The coloring information is expressed in the mesh index space,
      i.e., the ids are global.
     */

    std::vector<index_coloring> idx_colorings;

    /*
     The local allocation size for each connectivity. The outer vector
     is over index spaces. The inner vector is over connectivities between
     the outer entity type and another entity type. The ordering follows
     that given in the specialization policy.
     */

    std::vector<std::vector<std::size_t>> cnx_allocs;

    /*
      The local graph for each connectivity.

      The graph information is expressed in the mesh index space,
      i.e., the ids are global. The outer and inner vectors are like
      cnx_allocs.
     */

    std::vector<std::vector<crs>> cnx_colorings;
  }; // struct process_color

  using coloring = std::vector<process_color>;

  static std::size_t idx_size(index_coloring const & ic, std::size_t) {
    return ic.owned.size() + ic.ghost.size();
  }

  /*
    Using the Mesh Index Space (MIS) ordering, compute intervals,
    and the number of intervals for each color. Also compute the
    point offsets. The references num_intervals, intervals,
    and points, respectively, are filled with this information.
   */

  template<PrivilegeCount N>
  static void idx_itvls(index_coloring const & ic,
    std::vector<std::size_t> & num_intervals,
    std::vector<std::pair<std::size_t, std::size_t>> & intervals,
    std::map<Color, std::vector<std::pair<std::size_t, std::size_t>>> &
      src_points,
    field<util::id>::accessor1<privilege_cat<privilege_repeat<wo, N - (N > 1)>,
      privilege_repeat<na, (N > 1)>>> fmap,
    std::map<std::size_t, std::size_t> & rmap,
    MPI_Comm const & comm) {
    std::vector<std::size_t> entities;

    /*
      Define the entity ordering from coloring. This version uses the
      mesh ordering, i.e., the entities are sorted by ascending mesh id.
     */

    for(auto e : ic.owned) {
      entities.push_back(e);
    } // for

    auto [rank, size] = util::mpi::info(comm);

    std::vector<std::vector<std::size_t>> requests(size);
    for(auto e : ic.ghost) {
      entities.push_back(e.id);
      requests[e.color].emplace_back(e.id);
    } // for

    /*
      This call is what actually establishes the entity ordering by
      sorting the mesh entity ids.
     */

    util::force_unique(entities);

    /*
      Initialize the forward and reverse maps.
     */

    flog_assert(entities.size() == (ic.owned.size() + ic.ghost.size()),
      "entities size(" << entities.size() << ") doesn't match sum of owned("
                       << ic.owned.size() << ") and ghost(" << ic.ghost.size()
                       << ")");

    std::size_t off{0};
    rmap.clear();
    for(auto e : entities) {
      fmap[off] = e;
      rmap[e] = off++;
    }

    /*
      After the entity order has been established, we need to create
      a lookup table for local ghost offsets.
     */

    std::map<std::size_t, std::size_t> ghost_offsets;
    for(auto e : ic.ghost) {
      auto it = std::find(entities.begin(), entities.end(), e.id);
      flog_assert(it != entities.end(), "ghost entity doesn't exist");
      ghost_offsets[e.id] = std::distance(entities.begin(), it);
    } // for

    /*
      We also need to create a lookup table so that we can provide
      local shared offset information to other processes that request it.
     */

    std::map<std::size_t, std::size_t> shared_offsets;
    for(auto & e : ic.shared) {
      auto it = std::find(entities.begin(), entities.end(), e.id);
      flog_assert(it != entities.end(), "shared entity doesn't exist");
      shared_offsets[e.id] = std::distance(entities.begin(), it);
    } // for

    /*
      Send/Receive requests for shared offsets with other processes.
     */

    auto requested = util::mpi::all_to_allv(
      [&requests](int r, int) -> auto & { return requests[r]; }, comm);

    /*
      Fulfill the requests that we received from other processes, i.e.,
      provide the locaL offset for the requested mesh ids.
     */

    std::vector<std::vector<std::size_t>> fulfills(size);
    {
      int r = 0;
      for(const auto & rv : requested) {
        for(auto c : rv) {
          fulfills[r].emplace_back(shared_offsets[c]);
        } // for
        ++r;
      } // for
    } // scope

    /*
      Send/Receive the local offset information with other processes.
     */

    auto fulfilled = util::mpi::all_to_allv(
      [f = std::move(fulfills)](int r, int) { return std::move(f[r]); }, comm);
    /*
      Setup source pointers.
     */

    int r = 0;
    for(const auto & rv : fulfilled) {
      if(r == rank) {
        ++r;
        continue;
      } // if

      auto & points = src_points[r];
      points.reserve(rv.size());
      auto & request = requests[r];

      std::size_t i{0};
      for(auto v : rv) {
        points.emplace_back(std::make_pair(ghost_offsets[request[i]], v));
        ++i;
      }
      ++r;
    } // for

    /*
      Compute local intervals.
     */

    auto g = ghost_offsets.begin();
    std::size_t begin = 0, run = 0;
    for(; g != ghost_offsets.end(); ++g) {
      if(!run || g->second != begin + run) {
        if(run) {
          intervals.emplace_back(std::make_pair(begin, begin + run));
          begin = g->second;
        }
        run = 1;
      }
      else {
        ++run;
      }
    } // for

    intervals.emplace_back(std::make_pair(begin, begin + run));
    std::size_t local_itvls = intervals.size();

    /*
      Gather global interval sizes.
     */

    num_intervals = util::mpi::all_gather(local_itvls, comm);
  } // idx_itvls

  static void set_dests(field<data::intervals::Value>::accessor<wo> a,
    std::vector<std::pair<std::size_t, std::size_t>> const & intervals,
    MPI_Comm const &) {
    flog_assert(a.span().size() == intervals.size(), "interval size mismatch");
    std::size_t i{0};
    for(auto it : intervals) {
      a[i++] = data::intervals::make(it, process());
    } // for
  }

  template<PrivilegeCount N>
  static void set_ptrs(
    field<data::points::Value>::accessor1<privilege_repeat<wo, N>> a,
    std::map<Color, std::vector<std::pair<std::size_t, std::size_t>>> const &
      shared_ptrs,
    MPI_Comm const &) {
    for(auto const & si : shared_ptrs) {
      for(auto p : si.second) {
        // si.first: owner
        // p.first: local ghost offset
        // p.second: remote shared offset
        a[p.first] = data::points::make(si.first, p.second);
      } // for
    } // for
  }

  template<std::size_t S>
  static void idx_subspaces(index_coloring const & ic,
    field<util::id, data::ragged>::mutator<rw> owned,
    field<util::id, data::ragged>::mutator<rw> exclusive,
    field<util::id, data::ragged>::mutator<rw> shared,
    field<util::id, data::ragged>::mutator<rw> ghost) {
    const auto cp = [](auto r, const std::vector<util::id> & v) {
      r.assign(v.begin(), v.end());
    };

    cp(owned[S], ic.owned);
    cp(exclusive[S], ic.exclusive);
    cp(shared[S], ic.shared);
    cp(ghost[S], ic.ghost);
  }

  static void cnx_size(std::size_t size, resize::Field::accessor<wo> a) {
    a = size;
  }

}; // struct unstructured_base

} // namespace topo

/*----------------------------------------------------------------------------*
  Serialization Rules
 *----------------------------------------------------------------------------*/

template<>
struct util::serial<topo::unstructured_impl::shared_entity> {
  using type = topo::unstructured_impl::shared_entity;
  template<class P>
  static void put(P & p, const type & s) {
    serial_put(p, s.id, s.dependents);
  }
  static type get(const std::byte *& p) {
    const serial_cast r{p};
    return type{r, r};
  }
};

template<>
struct util::serial<topo::unstructured_impl::ghost_entity> {
  using type = topo::unstructured_impl::ghost_entity;
  template<class P>
  static void put(P & p, const type & s) {
    serial_put(p, s.id, s.color);
  }
  static type get(const std::byte *& p) {
    const serial_cast r{p};
    return type{r, r};
  }
};

template<>
struct util::serial<topo::unstructured_impl::index_coloring> {
  using type = topo::unstructured_impl::index_coloring;
  template<class P>
  static void put(P & p, const type & c) {
    serial_put(p, c.all, c.owned, c.exclusive, c.shared, c.ghost);
  }
  static type get(const std::byte *& p) {
    const serial_cast r{p};
    return type{r, r, r, r, r};
  }
};

} // namespace flecsi
