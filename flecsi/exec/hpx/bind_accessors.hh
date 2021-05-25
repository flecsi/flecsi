// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_HPX_BIND_ACCESSORS_HH
#define FLECSI_EXEC_HPX_BIND_ACCESSORS_HH

#include <flecsi-config.h>

#if !defined(FLECSI_ENABLE_HPX)
#error FLECSI_ENABLE_HPX not defined! This file depends on HPX!
#endif

#include <hpx/modules/collectives.hpp>
#include <hpx/modules/serialization.hpp>

#include "flecsi/data/privilege.hh"
#include "flecsi/data/topology.hh"
#include "flecsi/exec/hpx/future.hh"
#include "flecsi/exec/hpx/reduction_wrapper.hh"
#include "flecsi/flog.hh"
#include "flecsi/util/annotation.hh"
#include "flecsi/util/demangle.hh"

#include <cstddef>
#include <functional>
#include <string>
#include <variant>
#include <vector>

namespace flecsi {
namespace exec {

using region_or_partition =
  std::variant<std::shared_ptr<data::local::region_impl>,
    std::shared_ptr<data::local::partition_impl>>;

/*!
  The bind_accessors type is called to walk the user task arguments inside of an
  executing HPX task to properly complete the users accessors, i.e., by pointing
  the accessor \em view instances to the appropriate buffers.

  This is the other half of the wire protocol implemented by \c task_prologue.
 */
template<task_processor_type_t ProcessorType>
struct bind_accessors {

  bind_accessors(std::vector<region_or_partition> & regions_partitions)
    : argument(0), regions_partitions(regions_partitions) {}

protected:
  template<typename T>
  auto next_storage(field_id_t f) {
    flog_assert(argument < regions_partitions.size(),
      "there shouldn't be more arguments than partitions/regions");
    return std::visit(
      [f](auto && r_or_p) {
        return r_or_p->template get_storage<T, ProcessorType>(f);
      },
      regions_partitions[argument++]);
  }

  // visit generic topology
  template<typename T, Privileges P>
  void visit(data::accessor<data::raw, T, P> & accessor) {
    // Bind the ExecutionSpace storage to the accessor. This will also trigger a
    // host <-> device copy if needed.
    auto const storage = next_storage<T>(accessor.field());
    accessor.bind(storage);
  }

  // visit for reduction operation
  template<typename R, typename T>
  void visit(data::reduction_accessor<R, T> & accessor) {
    field_id_t const f = accessor.field();

    auto storage = next_storage<T>(f);
    accessor.bind(storage);

    // Reset the storage to identity on all processes except rank 0
    if(run::context::instance().process() != 0)
      std::fill(storage.begin(), storage.end(), R::template identity<T>);

    auto comm_gen = flecsi::run::context::instance().world_comm(
      "reduce_" + std::to_string(f));

    reductions.push_back([storage, comm_gen] {
      using data_type = ::hpx::serialization::serialize_buffer<T>;
      using namespace ::hpx::collectives;
      auto & [comm, generation] = comm_gen;
      auto fut = all_reduce(comm,
        data_type(storage.data(), storage.size()),
        exec::fold::wrap<R>{},
        generation_arg(generation));

      return fut.then([storage](auto && fut) {
        auto && data = fut.get();
        flog_assert(data.size() == storage.size(),
          "received size of data must be the same as the storage size");
        std::move(data.begin(), data.begin() + data.size(), storage.data());
      });
    });
  }

public:
  ~bind_accessors() {
    flog_assert(argument == regions_partitions.size(),
      "all fields should be used by bind_accessors");

    if(!reductions.empty()) {
      std::vector<::hpx::future<void>> requests;
      requests.reserve(reductions.size());
      for(auto & f : reductions) {
        requests.push_back(f());
      }
      ::hpx::wait_all_nothrow(requests);
      for(auto && f : requests) {
        if(f.has_exception()) {
          // there is no way to report the error to the user's code at this
          // point, thus termination is the only option
          flog_fatal("future is in exceptional state during destruction of "
                     "bind_accessors");
        }
      }
    }
  }

private:
  std::vector<std::function<::hpx::future<void>()>> reductions;

  std::size_t argument = 0;

  // regions_partitions is held alive by the task
  std::vector<region_or_partition> & regions_partitions;
};
} // namespace exec
} // namespace flecsi

#endif // FLECSI_EXEC_HPX_BIND_ACCESSORS_HH
