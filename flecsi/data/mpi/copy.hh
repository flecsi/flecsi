// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_MPI_COPY_HH
#define FLECSI_DATA_MPI_COPY_HH

#include "flecsi/data/field_info.hh"
#include "flecsi/data/local/copy.hh"

#include <unordered_map>

namespace flecsi {
namespace data {

struct copy_engine : local::copy_engine {
  // One copy engine for each entity type i.e. vertex, cell, edge.
  copy_engine(const prefixes & src,
    const data::intervals & intervals,
    field_id_t meta_fid /* for remote shared entities */)
    : local::copy_engine(src,
        intervals,
        meta_fid,
        [](auto const & remote_shared_entities) {
          return util::mpi::all_to_allv([&](int r) -> auto & {
            static const std::vector<std::size_t> empty;
            const auto i = remote_shared_entities.find(r);
            return i == remote_shared_entities.end() ? empty : i->second;
          });
        }) {}

  template<exec::processor P>
  void copy(const copy_request::vec & ff) const {
    using util::mpi::test;

    std::vector<std::vector<std::byte>> recv_buffers;
    std::size_t max_scatter_buffer_size = 0;

    {
      std::vector<std::vector<std::byte>> send_buffers;
      util::mpi::auto_requests requests(
        (ghost_entities.size() + shared_entities.size()) * ff.size());

      for(auto & [data_fid, write] : ff) {
        auto type_size = source->get_field_info(data_fid)->type_size;

        for(const auto & [src_rank, ghost_indices] : ghost_entities) {
          recv_buffers.emplace_back(ghost_indices.size() * type_size);
          max_scatter_buffer_size =
            std::max(max_scatter_buffer_size, recv_buffers.back().size());
          test(MPI_Irecv(recv_buffers.back().data(),
            int(recv_buffers.back().size()),
            MPI_BYTE,
            int(src_rank),
            0,
            MPI_COMM_WORLD,
            requests()));
        }

        if(write && P != exec::processor::toc)
          (*destination)[data_fid].storage().prefer<P>(true);
        // NB: source and destination typically alias.
        const auto [cpu, gpu] = (*source)[data_fid].storage().data2();

        std::optional<Kokkos::View<std::byte *, Kokkos::DefaultExecutionSpace>>
          gather_buffer_device_view;

        // Shared data in the field storage is copied to the gather buffer
        // in parallel. It is then copied to the send buffer (on host) and
        // sent to the peer via MPI_Send.
        for(const auto & [peer, indices] : shared_entities) {
          const auto n_elements = indices.size();
          auto n_bytes = n_elements * type_size;
          std::byte * const dst = send_buffers.emplace_back(n_bytes).data();

          if(cpu) {
            auto src_indices_view = indices.data();

            for(std::size_t i = 0; i < src_indices_view.extent(0); i++) {
              std::memcpy(dst + i * type_size,
                cpu + src_indices_view[i] * type_size,
                type_size);
            }
          }
          else {
            auto src_indices_view =
              indices.template data<exec::processor::toc>();

            if(!gather_buffer_device_view)
              gather_buffer_device_view.emplace(
                Kokkos::ViewAllocateWithoutInitializing("gather"),
                max_shared_indices_size * type_size);

            // copy shared values to gather buffer on device in parallel,
            // for each element
            const auto base = gpu; // to be captured, until C++20
            Kokkos::parallel_for(
              n_elements, KOKKOS_LAMBDA(const auto & i) {
                // Yes, memcpy is supported on device as long as there is no
                // std:: qualifier.
                memcpy(gather_buffer_device_view->data() + i * type_size,
                  base + src_indices_view[i] * type_size,
                  type_size);
              });

            auto gather_view = Kokkos::subview(*gather_buffer_device_view,
              std::pair<std::size_t, std::size_t>(0, n_bytes));
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
              backend_storage::host_view{dst, n_bytes},
              gather_view);
          }

          test(MPI_Isend(dst,
            int(n_bytes),
            MPI_BYTE,
            int(peer),
            0,
            MPI_COMM_WORLD,
            requests()));
        }
      }
    }

    std::optional<Kokkos::View<std::byte *, Kokkos::DefaultExecutionSpace>>
      scatter_buffer_device_view;

    // Copy recv_buffers to scatter_buffer_device_view and then in parallel
    // into the field's storage (on device).
    auto recv_buffer = recv_buffers.begin();
    for(auto & [data_fid, write] : ff) {
      auto & d = (*destination)[data_fid].storage();
      if(write)
        d.prefer<P>(); // has effect only for toc

      auto type_size = source->get_field_info(data_fid)->type_size;
      const auto [cpu, gpu] = d.data2();

      for(const auto & [_, indices] : ghost_entities) {
        const auto n_elements = indices.size();

        if(cpu) {
          auto dst_indices_view = indices.data();

          const std::byte * src = recv_buffer->data();

          for(std::size_t i = 0; i < dst_indices_view.extent(0); i++) {
            std::memcpy(cpu + dst_indices_view[i] * type_size,
              src + i * type_size,
              type_size);
          }
        }
        if(gpu) {
          if(!scatter_buffer_device_view)
            scatter_buffer_device_view.emplace(
              Kokkos::ViewAllocateWithoutInitializing("scatter"),
              max_scatter_buffer_size);
          auto scatter_view = Kokkos::subview(*scatter_buffer_device_view,
            std::pair<std::size_t, std::size_t>(0, recv_buffer->size()));
          Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
            scatter_view,
            backend_storage::host_view{
              recv_buffer->data(), recv_buffer->size()});

          auto dst_indices_view = indices.template data<exec::processor::toc>();

          // copy ghost values from scatter buffer on device to field
          // storage in parallel, for each element
          const auto base = gpu; // to be captured, until C++20
          Kokkos::parallel_for(
            n_elements, KOKKOS_LAMBDA(const auto & i) {
              memcpy(base + dst_indices_view[i] * type_size,
                scatter_buffer_device_view->data() + i * type_size,
                type_size);
            });
        }
        recv_buffer++;
      }
    }
  }
};

} // namespace data
} // namespace flecsi

#endif // FLECSI_DATA_MPI_COPY_HH
