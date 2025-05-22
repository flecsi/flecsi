// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_HPX_BIND_ACCESSORS_HH
#define FLECSI_EXEC_HPX_BIND_ACCESSORS_HH

#include <hpx/modules/collectives.hpp>
#include <hpx/modules/serialization.hpp>

#include "flecsi/exec/hpx/reduction_wrapper.hh"
#include "flecsi/exec/prolog.hh"
#include "flecsi/flog.hh"

#include <cstddef>
#include <functional>
#include <string>
#include <variant>
#include <vector>

namespace flecsi {
namespace exec {

/*!
  The bind_accessors type is called to walk the user task arguments inside of an
  executing HPX task to properly complete the users accessors, i.e., by pointing
  the accessor \em view instances to the appropriate buffers.

  This is the other half of the wire protocol implemented by \c task_prologue.
 */
template<processor Proc>
struct bind_accessors : local::bind<bind_accessors<Proc>, Proc> {
  explicit bind_accessors(run::communicator * comm,
    data::local::storages & regions_partitions)
    : bind_accessors::bind(regions_partitions), comm(comm) {}

  template<typename R, typename T>
  void reduce(util::span<T> storage) {
    reductions.push_back([storage](run::communicator & comm) {
      using data_type = ::hpx::serialization::serialize_buffer<T>;
      using namespace ::hpx::collectives;
      auto fut = all_reduce(comm.comm(),
        data_type(storage.data(), storage.size()),
        exec::fold::wrap<R>{},
        comm.gen());

      return fut.then(::hpx::launch::sync, [storage](auto && fut) {
        auto && data = fut.get();
        flog_assert(data.size() == storage.size(),
          "received size of data must be the same as the storage size");
        std::move(data.begin(), data.begin() + data.size(), storage.data());
      });
    });
  }

  ~bind_accessors() {
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
};
} // namespace exec
} // namespace flecsi

#endif // FLECSI_EXEC_HPX_BIND_ACCESSORS_HH
