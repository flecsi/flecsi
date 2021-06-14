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

#if !defined(FLECSI_ENABLE_HPX)
#error FLECSI_ENABLE_HPX not defined! This file depends on HPX!
#endif

#include <hpx/modules/collectives.hpp>
#include <hpx/modules/concurrency.hpp>
#include <hpx/modules/serialization.hpp>

#include "flecsi/data/privilege.hh"
#include "flecsi/data/topology.hh"
#include "flecsi/exec/hpx/future.hh"
#include "flecsi/util/demangle.hh"

#include <memory>

namespace flecsi {
namespace topo {
struct global_base;
}

namespace exec {

struct task_prologue {

private:
  // the spinlock-pool is used to protect the initialization of the
  // futures that are associated with the storage for a field
  using spinlock_pool = ::hpx::util::spinlock_pool<::hpx::shared_future<void>>;

public:
  task_prologue()
    : promise(), future(promise.get_future()), has_dependencies(false) {}

protected:
  // Those methods are "protected" because they are *only* called by
  // flecsi::exec::prolog() which inherits from task_prologue.

  // Patch up the "un-initialized" type conversion from future<R, index>
  // to future<R, single> in the generic code.
  template<typename R>
  static void visit(future<R, exec::launch_type_t::single> & single,
    future<R, exec::launch_type_t::index> & index) {
    single = index.get(flecsi::run::context::instance().color());
  }

  template<typename T,
    Privileges P,
    class Topo,
    typename Topo::index_space Space>
  void visit(data::accessor<data::raw, T, P> & accessor,
    const data::field_reference<T, data::raw, Topo, Space> & ref) {
    const field_id_t f = ref.fid();
    auto & t = ref.topology();
    data::region & reg = t.template get_region<Space>();
    constexpr bool glob =
      std::is_same_v<typename Topo::base, topo::global_base>;

    auto & r = [&]() -> auto & {
      if constexpr(glob) {
        return *t;
      }
      else {
        // The partition controls how much memory is allocated.
        return t.template get_partition<Space>(f);
      }
    }
    ();

    // if the current argument as a valid future associated with it, then
    // it has to be used a dependency for executing the current task
    auto & field_future = r.get_future(f);
    if(field_future.valid()) {
      std::lock_guard<::hpx::util::detail::spinlock> l(
        spinlock_pool::spinlock_for(
          ::hpx::traits::detail::get_shared_state(future).get()));
      dependencies.push_back(field_future);
    }

    const auto storage = r.template get_storage<T>(f);
    if constexpr(glob) {
      using data_type = ::hpx::serialization::serialize_buffer<T>;
      auto comm = flecsi::run::context::instance().world_comm();
      if(reg.ghost<privilege_pack<get_privilege(0, P), ro>>(f)) {
        if(comm.is_root()) {
          auto f = ::hpx::collectives::broadcast_to(comm,
            data_type(storage.data(), storage.size(), data_type::reference));
          f.get();
        }
        else {
          auto f = ::hpx::collectives::broadcast_from<data_type>(comm);
          auto && data = f.get();
          assert(data.size() == storage.size());
          std::move(data.begin(), data.begin() + data.size(), storage.data());
        }
      }
    }
    else {
      reg.ghost_copy<P>(ref);
    }

    if(privilege_write(P)) {
      // if the task writes to the current argument then we must associate
      // the future that represents the result of the task with the argument
      std::lock_guard<::hpx::util::detail::spinlock> l(
        spinlock_pool::spinlock_for(
          ::hpx::traits::detail::get_shared_state(future).get()));
      field_future = future;
      has_dependencies = true;
    }

    accessor.bind(storage);
  } // visit generic topology

public:
  /*!
    Delay the execution of the given task until all dependencies have been
    satisfied.
   */
  template<typename Params, typename Task>
  auto
  delay_execution(Params && params, std::string && task_name, Task && task) {
    return ::hpx::dataflow(
      ::hpx::launch::sync,
      [&, params = std::move(params), task_name = std::move(task_name)](
        auto && deps) mutable {
        // TIP: param_buffers is an RAII type. We create an instance and give
        // the object a reference to the parameters and name of the task. We
        // then assign it to the `finalize` variable so it is not destroyed
        // immediately. The object will be destroyed when reduce_internal()
        // returns. The ~param_buffers() will then perform the necessary clean
        // up on the parameters (mostly calling mutator.commit()).
        auto finalize = param_buffers(params, task_name);

        // rethrow exceptions
        for(auto && f : deps)
          f.get();

        // invoke actual task
        return task(std::move(params));
      },
      dependencies);
  }

  /*!
    Make sure the future that is associated with the result of this
    task is made ready once the task finishes executing.
   */
  template<typename T>
  void attach_result(::hpx::future<T> const & f) {
    if(has_dependencies) {
      hpx::traits::detail::get_shared_state(f)->set_on_completed(
        [p = std::move(promise)]() mutable { p.set_value(); });
    }
  }

private:
  /*!
    The futures that represent the dependencies of the current task on its
    arguments
   */
  std::vector<::hpx::shared_future<void>> dependencies;

  /*!
    This future is used as a dependency for all arguments, if needed. It
    is used to convey the availability of this task's result.
   */
  ::hpx::lcos::local::promise<void> promise;
  ::hpx::shared_future<void> future;

  /*!
    This is a flag that stores whether the future that represents the result
    of the current task will be used as a dependency for other tasks.
   */
  bool has_dependencies;
}; // struct task_prologue_t
} // namespace exec
} // namespace flecsi
