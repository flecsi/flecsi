// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_HPX_COPY_HH
#define FLECSI_DATA_HPX_COPY_HH

#include <hpx/modules/collectives.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/modules/lock_registration.hpp>
#include <hpx/modules/synchronization.hpp>

#include "flecsi/data/backend.hh"
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
all_to_allv(F && f, run::communicator & comm) {
  using namespace ::hpx::collectives;

  // NB: comm.comm().get_info() would require explicit set_info()
  auto size = run::context::instance().processes();
  std::vector<std::vector<std::size_t>> result;
  result.reserve(size);

  for(std::size_t r = 0; r < size; ++r)
    result.push_back(f(r));

  return all_to_all(comm.comm(), std::move(result), this_site_arg(), comm.gen())
    .get();
} // all_to_allv
} // namespace detail

struct dependencies {
  using type = std::vector<fate::future>;

  template<class... FF>
  dependencies(FF &&... ff) {
    ((*this)(std::forward<FF>(ff)), ...);
  }

  void operator()(fate::future f) {
    if(f.valid() && !f.is_ready())
      v.push_back(std::move(f));
  }

  // hpx::dataflow's parameters can't trigger implicit conversions.
  type detach() {
    return std::move(v);
  }

private:
  type v;
};

template<typename F>
void
init_delayed_ghost_copy(backend_storage & src_field,
  backend_storage & dest_field,
  F && delayed_ghost_copy) {
  // It is typical that src_field and dest_field alias.  do_write always
  // clears all holds before performing the callback, so then s is empty;
  // do_read stores the copy future, but then do_write discards it properly.
  dest_field.do_write([&](std::vector<hold> d) {
    auto future = hold::make();
    src_field.do_read([&](hold & s) {
      static constexpr bool use_comm = !std::is_invocable_v<F>;
      dependencies dep(future.depend(s));
      for(auto & h : d)
        dep(future.depend(h));
      future.assign(::hpx::dataflow(
        [out = run::context::instance().outstanding(),
          delayed_ghost_copy = std::forward<F>(delayed_ghost_copy),
          comm = use_comm ? &future.comm() : nullptr](
          dependencies::type ff) mutable {
          ::hpx::wait_all(std::move(ff)); // propagate exceptions
          const auto local = out();
          if constexpr(use_comm)
            std::forward<F>(delayed_ghost_copy)(*comm);
          else
            std::forward<F>(delayed_ghost_copy)();
        },
        dep.detach()));
      return future;
    });
    return future;
  });
}

struct copy_engine : local::copy_base {
  // One copy engine for each entity type i.e. vertex, cell, edge.
  copy_engine(const prefixes & src, const data::intervals & dest, field_id_t f)
    : p(std::make_shared<local::copy_engine>(src,
        dest,
        f,
        [&](auto const & remote_shared_entities) {
          return detail::all_to_allv(
            [&](int r) -> auto & {
              static std::vector<std::size_t> const empty;
              auto const it = remote_shared_entities.find(r);
              return it == remote_shared_entities.end() ? empty : it->second;
            },
            run::context::instance().world0);
        })) {}

  template<exec::processor>
  void copy(const copy_request::vec & ff) const {
    auto & ctx = run::context::instance();
    for(auto & [data_fid, _] : ff) {
      auto &src_storage = (*p->source)[data_fid].storage(),
           &dst_storage = (*p->destination)[data_fid].storage();
      init_delayed_ghost_copy(src_storage,
        dst_storage,
        // Don't use context asynchronously:
        [p = p,
          &src_storage,
          &dst_storage,
          type_size = p->source->get_field_info(data_fid)->type_size,
          comm = ctx.p2p_comm(),
          p2p = ctx.p2p_tag()]() {
          // manage task_local variables for this task
          run::task_local_base::guard tlg;

          // annotate new HPX thread
          ::hpx::scoped_annotation _("copy");

          // Since we are doing ghost copy via HPX, we always want the host side
          // version.
          std::byte * const dst = dst_storage.data<rw>().data();

          using namespace ::hpx::collectives;
          using data_type = std::vector<std::byte>;

          std::vector<::hpx::future<void>> ops;
          ops.reserve(p->ghost_entities.size() + p->shared_entities.size());
          for(auto const & entry : p->ghost_entities) {
            auto src_rank = entry.first;
            ops.push_back(
              get<data_type>(comm, that_site_arg(src_rank), p2p)
                .then(::hpx::launch::sync, [&, &src = entry.second](auto && f) {
                  auto && data = f.get();
                  for(std::size_t i = 0, n = src.size(); i < n; ++i)
                    std::memcpy(dst + src.data()[i] * type_size,
                      data.data() + i * type_size,
                      type_size);
                }));
          }

          const std::byte * const src = src_storage.data().data();
          for(auto const & [dst_rank, shared_indices] : p->shared_entities) {
            data_type send_buffer(shared_indices.size() * type_size);
            for(std::size_t i = 0, n = shared_indices.size(); i < n; ++i)
              std::memcpy(send_buffer.data() + i * type_size,
                src + shared_indices.data()[i] * type_size,
                type_size);
            ops.push_back(set(comm,
              that_site_arg(dst_rank),
              std::move(send_buffer),
              tag_arg(p2p)));
          }

          ::hpx::wait_all(std::move(ops)); // rethrows exceptions, if needed
        });
    }
  }

private:
  std::shared_ptr<local::copy_engine> p;
};

} // namespace data
} // namespace flecsi

#endif // FLECSI_DATA_HPX_COPY_HH
