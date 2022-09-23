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

// A topology that reuses a region from another.
// NB: No check is made that the field ID corresponds to the partition.
struct indirect_base : data::borrow {
  struct coloring {};

  template<class... AA>
  explicit indirect_base(data::region & r, AA &&... aa)
    : borrow(r, std::forward<AA>(aa)...), reg(&r) {}

  data::region & get_region() const {
    return *reg;
  }

private:
  data::region * reg;
};
// Note that P is not the underlying topology Q, but rather indirect<Q>.
template<class P>
struct indirect_category : indirect_base {
  template<class... AA>
  explicit indirect_category(typename P::Base::core & t, AA &&... aa)
    : indirect_base(t.template get_region<P::index_spaces::value>(),
        std::forward<AA>(aa)...) {}

  template<typename P::index_space>
  data::region & get_region() const {
    return indirect_base::get_region();
  }
};
template<>
struct detail::base<indirect_category> {
  using type = indirect_base;
};

template<class Q, typename Q::index_space S = Q::default_space()>
struct indirect : specialization<indirect_category, indirect<Q>> {
  using Base = Q;
  using index_space = typename Q::index_space;
  using index_spaces = util::constants<S>;

  static TopologyType id() = delete; // prevent ineffectual field registration
};

// An optional color to use for a point task.
struct claims : specialization<column, claims> {
  using Field = flecsi::field<data::borrow::Value, data::single>;
  static const Field::definition<claims> field;

  // The color is encoded as a prefix of size 0 or 1.
  static data::borrow::Value row(std::optional<Color> c) {
    return data::borrow::make(!!c, c.value_or(0));
  }
};
inline const claims::Field::definition<claims> claims::field;

/// \}
} // namespace flecsi::topo

#endif
