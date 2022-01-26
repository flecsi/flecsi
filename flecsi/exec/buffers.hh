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

#include <flecsi-config.h>

#include "flecsi/data/field.hh"
#include "flecsi/data/privilege.hh"
#include "flecsi/data/topology_accessor.hh"
#include "flecsi/run/context.hh"
#include "flecsi/util/annotation.hh"
#include "flecsi/util/demangle.hh"
#include "flecsi/util/type_traits.hh"

namespace flecsi {
/// \addtogroup execution
/// \{

namespace data {
template<class>
struct multi;
}

inline log::devel_tag param_buffers_tag("param_buffers");

namespace exec {

namespace detail {
// Manage the auxiliary object that survives the user task needed for a T, or
// a std::nullptr_t if none.
template<class T, class = void>
struct buffer {
  using type = decltype(nullptr);
  static void apply(T &, type) {}
};
template<class T>
struct buffer<T, util::voided<typename T::TaskBuffer>> {
  using type = typename T::TaskBuffer;
  static void apply(T & t, type & b) {
    t.buffer(b);
  }
};

template<class T, class B>
void
set_buffer(T & t, B & b) {
  detail::buffer<T>::apply(t, b);
}

// Note that what is visited are the objects \e moved into the user's
// parameters (and are thus the same object only in case of a reference).
struct param_buffers {
private:
  auto visitor() {
    return [&](auto & p, auto &&) { visit(p); };
  }

public:
  template<data::layout L, typename D, Privileges P>
  void visit(data::accessor<L, D, P> &) {} // visit

  template<data::layout L, class T, Privileges P>
  void visit(data::mutator<L, T, P> & m) {
    m.commit();
  }

  template<class A>
  void visit(data::multi<A> & m) {
    m.send(visitor());
  }

  template<class Topo, Privileges P>
  void visit(data::topology_accessor<Topo, P> &) {}

  /*--------------------------------------------------------------------------*
    Non-FleCSI Data Types
   *--------------------------------------------------------------------------*/

  template<typename D>
  static typename std::enable_if_t<!std::is_base_of_v<data::bind_tag, D>> visit(
    D &) {
    {
      log::devel_guard guard(param_buffers_tag);
      flog_devel(info) << "No cleanup for parameter of type " << util::type<D>()
                       << std::endl;
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
    (detail::set_buffer(std::get<II>(acc), std::get<II>(buf)), ...);
  }

  Tuple & acc;
  const std::string & nm;
  std::tuple<typename detail::buffer<TT>::type...> buf;
};

/// \}
} // namespace exec
} // namespace flecsi
