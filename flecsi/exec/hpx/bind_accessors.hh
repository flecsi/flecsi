// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_HPX_BIND_ACCESSORS_HH
#define FLECSI_EXEC_HPX_BIND_ACCESSORS_HH

#include <hpx/modules/collectives.hpp>
#include <hpx/modules/serialization.hpp>

#include "flecsi/config.hh"
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
template<processor Proc>
struct bind_accessors {
  explicit bind_accessors(run::communicator * comm,
    std::vector<region_or_partition> & regions_partitions)
    : comm(comm), regions_partitions(regions_partitions) {}

protected:
  void visit(processor_space_t<Proc> & s) {
    auto & c = run::context::instance();
    s.bind(c.colors(), c.color());
  }

  template<typename T, privilege P>
  auto next_storage(field_id_t f) {
    flog_assert(argument < regions_partitions.size(),
      "there shouldn't be more arguments than partitions/regions");
    return std::visit(
      [f](
        auto && r_or_p) { return r_or_p->template get_storage<T, P, Proc>(f); },
      regions_partitions[argument++]);
  }

  // visit generic topology
  template<typename T, Privileges P>
  void visit(data::accessor<data::raw, T, P> & accessor) {
    // Bind the ExecutionSpace storage to the accessor. This will also trigger a
    // host <-> device copy if needed.
    auto const storage = next_storage<T, privilege_merge(P)>(accessor.field());
    accessor.bind(storage);
  }

  // visit for reduction operation
  template<typename R, typename T>
  void visit(data::reduction_accessor<R, T> & accessor) {
    field_id_t const f = accessor.field();

    auto storage = next_storage<T, rw>(f);
    accessor.bind(storage);

    // Reset the storage to identity on all processes except rank 0
    if(run::context::instance().process() != 0)
      std::fill(storage.begin(), storage.end(), R::template identity<T>);

    reductions.push_back([storage](run::communicator & comm) {
      using data_type = ::hpx::serialization::serialize_buffer<T>;
      using namespace ::hpx::collectives;
      auto fut = all_reduce(comm.comm(),
        data_type(storage.data(), storage.size()),
        exec::fold::wrap<R>{},
        comm.gen());

      return fut.then(::hpx::launch::sync, [storage](auto && fut) {
        // manage task_local variables for this task
        run::task_local_base::guard tlg;

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

    flog_assert(reductions.empty() || comm, "no communicator for reductions");
    std::vector<data::fate> requests;
    requests.reserve(reductions.size());
    for(auto & f : reductions) {
      requests.push_back(data::fate::make(f(*comm)));
    }
  }

private:
  run::communicator * comm;
  std::vector<std::function<::hpx::future<void>(run::communicator &)>>
    reductions;

  std::size_t argument = 0;

  // regions_partitions is held alive by the task
  std::vector<region_or_partition> & regions_partitions;
};
} // namespace exec
} // namespace flecsi

#endif // FLECSI_EXEC_HPX_BIND_ACCESSORS_HH
