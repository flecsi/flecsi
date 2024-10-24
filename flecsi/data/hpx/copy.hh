// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_HPX_COPY_HH
#define FLECSI_DATA_HPX_COPY_HH

#include <hpx/modules/collectives.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/modules/lock_registration.hpp>
#include <hpx/modules/synchronization.hpp>

#include "flecsi/config.hh"
#include "flecsi/data/field_info.hh"
#include "flecsi/data/local/copy.hh"
#include "flecsi/run/hpx/context.hh"
#include "flecsi/util/mpi.hh"
#include "flecsi/util/types.hh"

#include <algorithm>
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

  for(std::size_t r = 0; r < size; ++r)
    result.push_back(f(r));

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
    // make the fields' values depend on this ghost copy operation after the
    // previous operation has finished
    field.future = field.future.then(::hpx::launch::async,
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

template<typename SrcField, typename DestField, typename F>
void
init_delayed_ghost_copy(SrcField & src_field,
  DestField & dest_field,
  F && delayed_ghost_copy) {

  ::hpx::future<void> future;
  if((src_field.future.valid() && !src_field.future.is_ready()) ||
     (dest_field.future.valid() && !dest_field.future.is_ready())) {
    // make the fields' values depend on this ghost copy operation after the
    // previous operation has finished
    future = ::hpx::dataflow(
      [delayed_ghost_copy = std::forward<F>(delayed_ghost_copy)](
        auto && src_f, auto && dest_f) mutable {
        src_f.get(); // propagate exceptions
        dest_f.get();
        delayed_ghost_copy();
      },
      src_field.future,
      dest_field.future);
  }
  else {
    // make the fields value depend on this ghost copy operation
    future = ::hpx::async(std::forward<F>(delayed_ghost_copy));
  }

  // ghost copy operations are implicit write operations to the field
  src_field.future = dest_field.future = future.share();
  src_field.dep = dest_field.dep = dependency::write;
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

  void use_as_dependency(::hpx::shared_future<void> const & f) {
    auto it = std::find_if(dependencies.begin(),
      dependencies.end(),
      [&](auto const & future) { return flecsi::detail::is_same(f, future); });
    if(it == dependencies.end()) {
      dependencies.push_back(f);
    }
  }

  // called with each field (and field_id_t) on the entity, for example, one
  // for pressure, temperature, density etc.
  void operator()(field_id_t data_fid) {

    // schedule the actual copy operation
    auto & src_field = (*source.r)[data_fid];
    auto & dest_field = destination[data_fid];

    auto comm_tag = std::to_string(data_fid);
    auto p2p_gen = flecsi::run::context::instance().p2p_comm(comm_tag);

    // Request communicator only if it is needed. Otherwise, a new generation
    // number is generated, which may cause for the communicator to hang if the
    // generation is not subsequently 'used'.
    if(!comm_gen.first) {
      comm_gen = flecsi::run::context::instance().world_comm(comm_tag);
    }

    init_delayed_ghost_copy(src_field,
      dest_field,
      [this, data_fid, comm_tag = std::move(comm_tag), p2p_gen]() {
        // manage task_local variables for this task
        run::task_local_base::guard tlg;

        // annotate new HPX thread
        ::hpx::scoped_annotation _(comm_tag);

        // First make sure this copy_engine is completely initialized. For the
        // HPX backend this has to be delayed until the actual copy operation is
        // being executed.
        {
          std::unique_lock l(mtx);
          // ignore lock while suspending
          [[maybe_unused]] ::hpx::util::ignore_while_checking il(&l);

          // initialize every copy_engine exactly once
          if(max_local_source_idx == 0) {
            init_copy_engine([&](auto const & remote_shared_entities) {
              flog_assert(comm_gen.first && comm_gen.second,
                "communicator should have been initialized");
              return detail::all_to_allv(
                [&](int r) -> auto & {
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
        auto destination_storage =
          destination.get_storage<std::byte, rw>(data_fid);
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
              .then(::hpx::launch::sync, [&, &src = entry.second](auto && f) {
                auto && data = f.get();
                for(std::size_t i = 0, n = src.size(); i < n; ++i)
                  std::memcpy(
                    destination_storage.data() + src.data()[i] * type_size,
                    data.data() + i * type_size,
                    type_size);
              }));
        }

        for(auto const & [dst_rank, shared_indices] : shared_entities) {
          data_type send_buffer(shared_indices.size() * type_size);
          for(std::size_t i = 0, n = shared_indices.size(); i < n; ++i)
            std::memcpy(send_buffer.data() + i * type_size,
              source_storage.data() + shared_indices.data()[i] * type_size,
              type_size);
          ops.push_back(set(comm,
            that_site_arg(dst_rank),
            std::move(send_buffer),
            tag_arg(generation)));
        }

        ::hpx::wait_all(std::move(ops)); // rethrows exceptions, if needed
      });

    // this copy_engine object must be kept alive at least until all ghost_copy
    // operations have finished executing
    if(src_field.future.valid() && !src_field.future.is_ready()) {
      use_as_dependency(src_field.future);
    }
    if(dest_field.future.valid() && !dest_field.future.is_ready()) {
      use_as_dependency(dest_field.future);
    }
  }

private:
  ::hpx::spinlock mtx;
  run::context_t::communicator_data comm_gen;
  std::vector<::hpx::shared_future<void>> dependencies;
};

} // namespace data
} // namespace flecsi

#endif // FLECSI_DATA_HPX_COPY_HH
