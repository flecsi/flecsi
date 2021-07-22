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

#include "flecsi/topo/core.hh" // base

namespace flecsi {
namespace topo {

struct set_base {

  struct coloring {

    void * ptr;

    std::vector<std::size_t> counts;
  };

  static std::size_t allocate(const std::vector<std::size_t> & arr,
    const std::size_t & i) {

    return arr[i];
  }

}; // set_base

template<typename P>
struct set : set_base {

  using T = typename P::t_type;

  template<Privileges Priv>
  struct access {

    template<class F>
    void send(F &&) {}
  };

  explicit set(coloring x)
    : p{static_cast<T *>(x.ptr)}, part{make_repartitioned<P>(x.counts.size(),
                                    make_partial<allocate>(x.counts))} {}

  Color colors() const {

    return part.colors();
  }

  template<typename P::index_space>
  data::region & get_region() {
    return part;
  }

  template<typename P::index_space>
  repartition & get_partition(field_id_t) {

    return part;
  }

private:
  T * p;
  repartitioned part;
};

template<>
struct detail::base<set> {
  using type = set_base;
};

} // namespace topo
} // namespace flecsi
