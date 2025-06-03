// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_SET_INTERFACE_HH
#define FLECSI_TOPO_SET_INTERFACE_HH

#include "flecsi/topo/index.hh"

/// \cond core
namespace flecsi {
namespace topo {
/// \defgroup set Particle Set
/// Supports non-interacting particle methods.
/// Can be used for coloring and binning particles.
/// \ingroup topology
/// \{
struct set_base : base {
  /// This struct gives the coloring interface for the Set topology.
  /// \ingroup set
  struct coloring {
    /// Pointer to the underlying topology
    void * ptr;
    /// Counts per color
    std::vector<std::size_t> counts;
  };
}; // set_base

/// This struct is a Set topology interface.
/// \tparam Policy the specialization following \ref set_specialization
///
/// The Set topology supports a single index space.
template<typename Policy>
struct topology<Policy, set_base> : set_base {

  using index_space = typename Policy::index_space;
  using mesh = typename Policy::mesh_type::topology;

  template<Privileges Priv>
  struct access {

    static_assert(privilege_count(Priv) == 1,
      "there is only one privilege in set topology");
    using accessorm = data::topology_accessor<typename Policy::mesh_type,
      privilege_repeat<get_privilege(0, Priv), 3>>;
    accessorm mesh;

    template<class F>
    void send(F && f) {
      f(
        mesh, [](auto && ts) -> auto & { return *ts->p; });
    }
  };

  topology(scheduler & s, coloring x)
    : p{static_cast<mesh *>(x.ptr)}, part{make_repartitioned<Policy>(
                                       x.counts.size(),
                                       s,
                                       [a = x.counts](
                                         std::size_t c) { return a[c]; })} {}

  Color colors() const {

    return part.colors();
  }

  template<typename Policy::index_space>
  data::region & get_region() {
    return part;
  }

  template<typename Policy::index_space>
  repartition & get_partition() {

    return part;
  }

private:
  mesh * p;
  repartitioned part;
};

/// Topology category.
template<class P>
using set = topology<P, set_base>;
template<>
struct detail::base<set> {
  using type = set_base;
};
#ifdef DOXYGEN
/// Example specialization which is not really implemented. Specializations
/// defining their own `index_space` and `index_spaces` are not supported by set
/// topology.
struct set_specialization : specialization<set, set_specialization> {
  /// Underlying topology type.
  using mesh_type = int;
};
#endif

/// \}
} // namespace topo
} // namespace flecsi
/// \endcond
#endif
