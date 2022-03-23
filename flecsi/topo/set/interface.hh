// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_SET_INTERFACE_HH
#define FLECSI_TOPO_SET_INTERFACE_HH

#include "flecsi/topo/core.hh" // base
#include "flecsi/topo/set/types.hh"

namespace flecsi {
namespace topo {

template<typename Policy>
struct set : set_base {};

template<>
struct detail::base<set> {
  using type = set_base;
};

} // namespace topo
} // namespace flecsi

#endif
