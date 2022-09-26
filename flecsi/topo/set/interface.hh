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
struct set_base {
  /// This struct gives the coloring interface for the Set topology.
  /// \ingroup set
  struct coloring {
    /// Pointer to the underlying topology slot
    void * ptr;
    /// Counts per color
    std::vector<std::size_t> counts;
  };

  static std::size_t allocate(const std::vector<std::size_t> & arr,
    const std::size_t & i) {

    return arr[i];
  }

}; // set_base

/// This struct is a Set topology interface.
/// \tparam Policy the specialization following \ref set_specialization
///
/// The Set topology supports a single index space.
template<typename Policy>
struct set : set_base {

  using index_space = typename Policy::index_space;
  using mesh_slot = typename Policy::mesh_type::slot;

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

  explicit set(coloring x)
    : p{static_cast<mesh_slot *>(x.ptr)}, part{make_repartitioned<Policy>(
                                            x.counts.size(),
                                            make_partial<allocate>(x.counts))} {
  }

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
  mesh_slot * p;
  repartitioned part;
};

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
