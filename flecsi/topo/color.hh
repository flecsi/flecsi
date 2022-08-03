// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

// The most basic topology.

#ifndef FLECSI_TOPO_COLOR_HH
#define FLECSI_TOPO_COLOR_HH

#include "flecsi/data/topology.hh"

namespace flecsi::topo {
/// \addtogroup topology
/// \{

// A topology with a fixed number of index points per color.
// Used for defining (and, for Legion, setting) the per-row sizes of other
// topologies (and thus is unsuitable for the ragged layout).
struct color_base {
  using coloring = data::size2;
};

template<class P>
struct color : color_base, data::partitioned<data::rows> {
  color(const coloring & c) : partitioned(data::make_region<P>(c)) {}
};
template<>
struct detail::base<color> {
  using type = color_base;
};

/// \}
} // namespace flecsi::topo

#endif
