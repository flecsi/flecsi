// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

// The most basic topologies, used to represent per-color metadata.

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

// The common special case of one index point per color.
struct column_base {
  using coloring = Color;
};

template<class P>
struct column : column_base, color<P> {
  using column_base::coloring;
  explicit column(coloring c) : color<P>({c, 1}) {}
};
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
