// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#include "flecsi/data.hh"

namespace flecsi::data {
pointers::pointers(prefixes & p, topo::claims::core & src)
  : column(src.colors()) {
  execute<expand, flecsi::mpi>(topo::claims::field(src), p.size(), **this);
}

void
pointers::expand(topo::claims::Field::accessor<ro> c,
  std::size_t w,
  local::claims::Field::accessor<wo> i) {
  auto [r, sz] = *c;
  i[0] = {r, sz * w};
}
} // namespace flecsi::data
