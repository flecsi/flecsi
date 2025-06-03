// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

// The most basic topologies, used to represent per-color metadata.
// For convenience, they are allowed to be movable (no topology accessors are
// needed anyway).

#ifndef FLECSI_TOPO_COLOR_HH
#define FLECSI_TOPO_COLOR_HH

#include "flecsi/data/topology.hh"
#include "flecsi/exec/fwd.hh" // topology

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
struct topology<P, color_base> : color_base, data::partitioned<data::rows> {
  topology(scheduler &, const coloring & c)
    : partitioned(data::make_region<P>(c)) {}
};
template<class P>
using color = topology<P, color_base>;
template<>
struct detail::base<color> {
  using type = color_base;
};

// The common special case of one index point per color.
struct column_base {
  using coloring = Color;
};

template<class P>
struct topology<P, column_base> : column_base, color<P> {
  using column_base::coloring;
  topology(scheduler & s, coloring c) : color<P>(s, {c, 1}) {}
};
template<class P>
using column = topology<P, column_base>;
template<>
struct detail::base<column> {
  using type = column_base;
};

// An optional color to use for a point task.
struct claims : specialization<column, claims> {
  using Field = flecsi::field<data::borrow::Claim, data::single>;
  static const Field::definition<claims> field;
};
inline const claims::Field::definition<claims> claims::field;

/// \}
} // namespace flecsi::topo

#endif
