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

#include "flecsi/exec/buffers.hh"
#include "flecsi/exec/leg/bind_accessors.hh"
#include "flecsi/exec/leg/future.hh"
#include "flecsi/exec/task_attributes.hh"
#include "flecsi/run/backend.hh"
#include "flecsi/util/annotation.hh"
#include "flecsi/util/common.hh"
#include "flecsi/util/function_traits.hh"
#include "flecsi/util/serialize.hh"
#include <flecsi/flog.hh>

#if !defined(FLECSI_ENABLE_LEGION)
#error FLECSI_ENABLE_LEGION not defined! This file depends on Legion!
#endif

#include <legion.h>

#include <string>
#include <utility>

namespace flecsi {

inline log::devel_tag task_wrapper_tag("task_wrapper");

namespace data {
template<class, Privileges, Privileges>
struct ragged_accessor;

namespace detail {
template<class A>
struct convert_accessor {
  using type = A;
  using Base = typename type::base_type;
  static const Base & put(const type & a) {
    return a.get_base();
  }
  static type get(Base b) {
    return b;
  }
};
} // namespace detail
} // namespace data

// Send and receive only the field ID:
template<data::layout L, class T, Privileges Priv>
struct util::serial::convert<data::accessor<L, T, Priv>> {
  using type = data::accessor<L, T, Priv>;
  using Rep = field_id_t;
  static Rep put(const type & r) {
    return r.field();
  }
  static type get(const Rep & r) {
    return type(r);
  }
};
template<class T, Privileges Priv>
struct util::serial::convert<data::accessor<data::single, T, Priv>>
  : data::detail::convert_accessor<data::accessor<data::single, T, Priv>> {};
template<class T, Privileges P, Privileges OP>
struct util::serial::convert<data::ragged_accessor<T, P, OP>>
  : data::detail::convert_accessor<data::ragged_accessor<T, P, OP>> {};
template<data::layout L, class T, Privileges Priv>
struct util::serial::traits<data::mutator<L, T, Priv>> {
  using type = data::mutator<L, T, Priv>;
  template<class P>
  static void put(P & p, const type & m) {
    serial::put(p, m.get_base());
  }
  static type get(const std::byte *& b) {
    return serial::get<typename type::base_type>(b);
  }
};
template<class T, Privileges Priv>
struct util::serial::traits<data::mutator<data::ragged, T, Priv>> {
  using type = data::mutator<data::ragged, T, Priv>;
  template<class P>
  static void put(P & p, const type & m) {
    serial::put(p, m.get_base(), m.get_grow());
  }
  static type get(const std::byte *& b) {
    const serial::cast r{b};
    return {r, r};
  }
};
template<class T, Privileges Priv>
struct util::serial::traits<data::topology_accessor<T, Priv>,
  std::enable_if_t<!util::bit_copyable_v<data::topology_accessor<T, Priv>>>>
  : util::serial::value<data::topology_accessor<T, Priv>> {};

template<auto & F, class... AA>
struct util::serial::traits<exec::partial<F, AA...>,
  std::enable_if_t<!util::bit_copyable_v<exec::partial<F, AA...>>>> {
  using type = exec::partial<F, AA...>;
  using Rep = typename type::Base;
  template<class P>
  static void put(P & p, const type & t) {
    serial::put(p, static_cast<const Rep &>(t));
  }
  static type get(const std::byte *& b) {
    return serial::get<Rep>(b);
  }
};

template<class T>
struct util::serial::traits<future<T>> : util::serial::value<future<T>> {};

namespace exec::leg {
using run::leg::task;

namespace detail {
inline util::counter<LEGION_MAX_APPLICATION_TASK_ID> task_counter(
  run::FLECSI_TOP_LEVEL_TASK_ID);
/*!
  Register a task with Legion.

  @tparam RETURN The return type of the task.
  @tparam TASK   The legion task.
  \tparam A task attributes

  @ingroup legion-execution
 */

template<typename RETURN, task<RETURN> * TASK, TaskAttributes A>
void register_task();

template<class>
struct tuple_get;
template<class... TT>
struct tuple_get<std::tuple<TT...>> {
  static auto get(const Legion::Task & t) {
    const auto p = static_cast<const std::byte *>(t.args);
    return util::serial::get_tuple<std::decay_t<TT>...>(p, p + t.arglen);
  }
};
} // namespace detail

/*!
  Arbitrary index for each task.

  @tparam F          Legion task function.
  @tparam A task attributes mask
 */

template<auto & F, TaskAttributes A = loc | leaf>
// 'extern' works around GCC bug #96523
extern const Legion::TaskID
  task_id = (run::context::instance().register_init(detail::register_task<
               typename util::function_traits<decltype(F)>::return_type,
               F,
               A>),
    detail::task_counter());

template<typename RETURN, task<RETURN> * TASK, TaskAttributes A>
void
detail::register_task() {
  constexpr auto processor_type = mask_to_processor_type(A);
  static_assert(processor_type != task_processor_type_t::mpi,
    "Legion tasks cannot use MPI");

  const std::string name = util::symbol<*TASK>();
  {
    log::devel_guard guard(task_wrapper_tag);
    flog_devel(info) << "registering pure Legion task " << name << std::endl;
  }

  Legion::TaskVariantRegistrar registrar(task_id<*TASK, A>, name.c_str());
  Legion::Processor::Kind kind = processor_type == task_processor_type_t::toc
                                   ? Legion::Processor::TOC_PROC
                                   : Legion::Processor::LOC_PROC;
  registrar.add_constraint(Legion::ProcessorConstraint(kind));

  registrar.set_leaf(A & leaf || ~A & inner);
  registrar.set_inner(A & inner);
  registrar.set_idempotent(A & idempotent);

  /*
    This section of conditionals is necessary because there is still
    a distinction between void and non-void task registration with
    Legion.
   */

  if constexpr(std::is_same_v<RETURN, void>) {
    Legion::Runtime::preregister_task_variant<TASK>(registrar, name.c_str());
  }
  else {
    Legion::Runtime::preregister_task_variant<RETURN, TASK>(
      registrar, name.c_str());
  } // if
}

// A trivial wrapper for nullary functions.
template<auto & F>
auto
verb(const Legion::Task *,
  const std::vector<Legion::PhysicalRegion> &,
  Legion::Context,
  Legion::Runtime *) {
  return F();
}

/*!
 The task_wrapper type provides execution
 functions for user and MPI tasks.

 \tparam F the user task
 \tparam P the target processor type

 @ingroup legion-execution
 */

template<auto & F, task_processor_type_t P>
struct task_wrapper {

  using Traits = util::function_traits<decltype(F)>;
  using RETURN = typename Traits::return_type;
  using param_tuple = typename Traits::arguments_type;

  static constexpr task_processor_type_t LegionProcessor = P;

  /*!
    Execution wrapper method for user tasks.
   */

  static RETURN execute(const Legion::Task * task,
    const std::vector<Legion::PhysicalRegion> & regions,
    Legion::Context context,
    Legion::Runtime * runtime) {
    {
      log::devel_guard guard(task_wrapper_tag);
      flog_devel(info) << "In execute_user_task" << std::endl;
    }

    // Unpack task arguments
    auto task_args = detail::tuple_get<param_tuple>::get(*task);

    namespace ann = util::annotation;
    auto tname = util::symbol<F>();
    const param_buffers buf(task_args, tname);
    (ann::rguard<ann::execute_task_bind>(tname),
      bind_accessors(runtime, context, regions, task->futures)(task_args));
    return ann::rguard<ann::execute_task_user>(tname),
           apply(F, std::forward<param_tuple>(task_args));
  } // execute_user_task

}; // struct task_wrapper

template<auto & F>
struct task_wrapper<F, task_processor_type_t::mpi> {
  using Traits = util::function_traits<decltype(F)>;
  using RETURN = typename Traits::return_type;
  using param_tuple = typename Traits::arguments_type;

  static constexpr auto LegionProcessor = task_processor_type_t::loc;

  static RETURN execute(const Legion::Task * task,
    const std::vector<Legion::PhysicalRegion> & regions,
    Legion::Context context,
    Legion::Runtime * runtime) {
    {
      log::devel_guard guard(task_wrapper_tag);
      flog_devel(info) << "In execute_mpi_task" << std::endl;
    }

    flog_assert(!task->arglen, "unexpected task arguments");
    auto & c = run::context::instance();
    const auto p = static_cast<param_tuple *>(c.mpi_params);

    namespace ann = util::annotation;
    auto tname = util::symbol<F>();
    const param_buffers buf(*p, tname);
    (ann::rguard<ann::execute_task_bind>(tname)),
      bind_accessors(runtime, context, regions, task->futures)(*p);

    // Set the MPI function and make the runtime active.
    if constexpr(std::is_void_v<RETURN>) {
      (ann::rguard<ann::execute_task_user>(tname)),
        c.mpi_call([&] { apply(F, std::move(*p)); });
    }
    else {
      std::optional<RETURN> result;
      (ann::rguard<ann::execute_task_user>(tname)),
        c.mpi_call([&] { result.emplace(std::apply(F, std::move(*p))); });
      return std::move(*result);
    }

  } // execute
}; // task_wrapper

} // namespace exec::leg
} // namespace flecsi
