// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_HPX_COPY_HH
#define FLECSI_DATA_HPX_COPY_HH

#include <flecsi-config.h>

#if !defined(FLECSI_ENABLE_HPX)
#error FLECSI_ENABLE_HPX not defined! This file depends on HPX!
#endif

#include <hpx/modules/collectives.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/modules/synchronization.hpp>

#include "flecsi/data/field_info.hh"
#include "flecsi/data/local/copy.hh"
#include "flecsi/run/hpx/context.hh"
#include "flecsi/util/mpi.hh"
#include "flecsi/util/types.hh"

#include <cstddef>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace flecsi {
namespace data {
namespace detail {
//  All-to-All (variable) communication pattern (HPX version).
template<typename F>
inline auto
all_to_allv(F && f, run::context_t::communicator_data comm_data) {

  using namespace ::hpx::collectives;

  auto & [comm, generation] = comm_data;
  auto [size, rank] = comm.get_info();

  std::vector<std::vector<std::size_t>> result;
  result.reserve(size);

  for(int r = 0; r < size; ++r)
    result.push_back(f(r, size));

  return all_to_all(
    comm, std::move(result), this_site_arg(), generation_arg(generation))
    .get();
} // all_to_allv
} // namespace detail

/// Add the delayed_ghost_copy as a dependency to the given field's dependencies
template<typename Field, typename F>
void
init_delayed_ghost_copy(Field & field, F && delayed_ghost_copy) {
  if(field.future.valid() && !field.future.is_ready()) {
    // make the field's value depend on this ghost copy operation after the
    // previous operation has finished
    field.future = field.future.then(
      [delayed_ghost_copy = std::forward<F>(delayed_ghost_copy)](
        auto && f) mutable {
        f.get(); // propagate exceptions
        delayed_ghost_copy();
      });
  }
  else {
    // make the fields value depend on this ghost copy operation
    field.future = ::hpx::async(std::forward<F>(delayed_ghost_copy));
  }

  // ghost copy operations are implicit write operations to the field
  field.dep = dependency::write;
}

struct copy_engine : local::copy_engine {
  // One copy engine for each entity type i.e. vertex, cell, edge.
  using local::copy_engine::copy_engine;

  // whenever this copy_engine is destroyed we have to wait for all pending
  // tasks to complete before exiting
  ~copy_engine() {
    ::hpx::wait_all_nothrow(dependencies);
    for(auto && f : dependencies) {
      if(f.has_exception()) {
        // there is no way to report the error to the user's code at this
        // point, thus termination is the only option
        flog_fatal("future is in exceptional state during destruction of "
                   "copy_engine:\n" +
                   ::hpx::diagnostic_information(f.get_exception_ptr()));
      }
    }
  }

  // called with each field (and field_id_t) on the entity, for example, one
  // for pressure, temperature, density etc.
  void operator()(field_id_t data_fid) {

    // schedule the actual copy operation
    auto & field = (*source.r)[data_fid];

    auto comm_tag = std::to_string(data_fid);
    auto p2p_gen = flecsi::run::context::instance().p2p_comm(comm_tag);
    auto comm_gen = flecsi::run::context::instance().world_comm(comm_tag);

    init_delayed_ghost_copy(field,
      [this, data_fid, comm_tag = std::move(comm_tag), comm_gen, p2p_gen]() {
        // annotate new HPX thread
        ::hpx::scoped_annotation _(comm_tag);

        // First make sure this copy_engine is completely initialized. For the
        // HPX backend this has to be delayed until the actual copy operation is
        // being executed.
        {
          std::unique_lock l(mtx);
          if(max_local_source_idx == 0) {
            init_copy_engine([&](auto const & remote_shared_entities) {
              return detail::all_to_allv(
                [&](int r, int) -> auto & {
                  static std::vector<std::size_t> const empty;
                  auto const it = remote_shared_entities.find(r);
                  return it == remote_shared_entities.end() ? empty
                                                            : it->second;
                },
                comm_gen);
            });
          }
        }

        // Since we are doing ghost copy via HPX, we always want the host side
        // version.
        auto source_storage =
          source.r->get_storage<std::byte>(data_fid, max_local_source_idx);
        auto destination_storage = destination.get_storage<std::byte>(data_fid);
        auto type_size = source.r->get_field_info(data_fid)->type_size;

        using namespace ::hpx::collectives;
        using data_type = std::vector<std::byte>;

        auto & [comm, generation] = p2p_gen;

        std::vector<::hpx::future<void>> ops;
        ops.reserve(ghost_entities.size() + shared_entities.size());
        for(auto const & entry : ghost_entities) {
          auto src_rank = entry.first;
          ops.push_back(
            get<data_type>(comm, that_site_arg(src_rank), tag_arg(generation))
              .then([&, src_rank](auto && f) {
                auto && data = f.get();
                std::size_t i = 0;
                for(auto ghost_idx : ghost_entities.find(src_rank)->second) {
                  std::memcpy(
                    destination_storage.data() + ghost_idx * type_size,
                    data.data() + i++ * type_size,
                    type_size);
                }
              }));
        }

        for(auto const & [dst_rank, shared_indices] : shared_entities) {
          data_type send_buffer(shared_indices.size() * type_size);
          std::size_t i = 0;
          for(auto shared_idx : shared_indices) {
            std::memcpy(send_buffer.data() + i++ * type_size,
              source_storage.data() + shared_idx * type_size,
              type_size);
          }
          ops.push_back(set(comm,
            that_site_arg(dst_rank),
            std::move(send_buffer),
            tag_arg(generation)));
        }

        ::hpx::wait_all(std::move(ops)); // rethrows exceptions, if needed
      });

    // this copy_engine object must be held alive at least until all ghost_copy
    // operations have finished executing
    if(field.future.valid() && !field.future.is_ready()) {
      dependencies.push_back(field.future);
    }
  }

private:
  ::hpx::spinlock mtx;
  std::vector<::hpx::shared_future<void>> dependencies;
};

} // namespace data
} // namespace flecsi

#endif // FLECSI_DATA_HPX_COPY_HH
