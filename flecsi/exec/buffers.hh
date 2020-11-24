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

#include <flecsi-config.h>

#if !defined(__FLECSI_PRIVATE__)
#error Do not include this file directly!
#endif

#include "flecsi/data/field.hh"
#include "flecsi/data/privilege.hh"
#include "flecsi/data/reference.hh"
#include "flecsi/exec/launch.hh"
#include "flecsi/run/context.hh"
#include "flecsi/util/annotation.hh"
#include "flecsi/util/demangle.hh"

namespace flecsi {

inline log::devel_tag param_buffers_tag("param_buffers");

namespace exec {

namespace detail {
// Note that what is visited are the objects \e moved into the user's
// parameters (and are thus the same object only in case of a reference).
struct param_buffers {
  template<data::layout L, typename DATA_TYPE, size_t PRIVILEGES>
  void visit(data::accessor<L, DATA_TYPE, PRIVILEGES> &) {} // visit

  template<data::layout L, class T, std::size_t P>
  void visit(data::mutator<L, T, P> & m) {
    m.commit();
  }

  template<class Topo, std::size_t P>
  void visit(data::topology_accessor<Topo, P> &) {}

  /*--------------------------------------------------------------------------*
    Non-FleCSI Data Types
   *--------------------------------------------------------------------------*/

  template<typename DATA_TYPE>
  static
    typename std::enable_if_t<!std::is_base_of_v<data::bind_tag, DATA_TYPE>>
    visit(DATA_TYPE &) {
    {
      log::devel_guard guard(param_buffers_tag);
      flog_devel(info) << "No cleanup for parameter of type "
                       << util::type<DATA_TYPE>() << std::endl;
    }
  } // visit
};
} // namespace detail

template<class... TT>
struct param_buffers : private detail::param_buffers {
  using Tuple = std::tuple<TT...>;

  param_buffers(Tuple & t, const std::string & nm) : acc(t), nm(nm) {
    buffer(std::index_sequence_for<TT...>());
  }
  // Prevent using temporaries, which is often unsafe:
  param_buffers(Tuple &, const std::string &&) = delete;

  ~param_buffers() noexcept(false) {
    util::annotation::rguard<util::annotation::execute_task_unbind> ann(nm);
    std::apply(
      [this](TT &... tt) {
        (void)this; // to appease Clang
        (visit(tt), ...);
      },
      acc);
  }

private:
  template<std::size_t... II>
  void buffer(std::index_sequence<II...>) {
    (set_buffer(std::get<II>(acc), std::get<II>(buf)), ...);
  }

  Tuple & acc;
  const std::string & nm;
  std::tuple<buffer_t<TT>...> buf;
};

} // namespace exec
} // namespace flecsi
