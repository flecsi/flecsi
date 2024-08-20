// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_MPI_COPY_HH
#define FLECSI_DATA_MPI_COPY_HH

#include "flecsi/data/field_info.hh"
#include "flecsi/data/local/copy.hh"

#include <unordered_map>

namespace flecsi {
namespace data {

// Copy/Paste from cppreference.com to make std::visit looks more
// like pattern matching in ML or Haskell.
template<class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
// explicit deduction guide (not needed as of C++20)
template<class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

struct copy_engine : local::copy_engine {
  // One copy engine for each entity type i.e. vertex, cell, edge.
  copy_engine(const data::points & pts,
    const data::intervals & intervals,
    field_id_t meta_fid /* for remote shared entities */)
    : local::copy_engine(pts, intervals, meta_fid) {
    init_copy_engine([](auto const & remote_shared_entities) {
      return util::mpi::all_to_allv([&](int r) -> auto & {
        static const std::vector<std::size_t> empty;
        const auto i = remote_shared_entities.find(r);
        return i == remote_shared_entities.end() ? empty : i->second;
      });
    });
  }

  // called with each field (and field_id_t) on the entity, for example, one
  // for pressure, temperature, density etc.
  void operator()(field_id_t data_fid) const {
    using util::mpi::test;

    auto type_size = source.r->get_field_info(data_fid)->type_size;

    std::vector<std::vector<std::byte>> recv_buffers;
    std::size_t max_scatter_buffer_size = 0;

    {
      std::vector<std::vector<std::byte>> send_buffers;
      util::mpi::auto_requests requests(
        ghost_entities.size() + shared_entities.size());

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

      // allocate gather buffer on device
      auto gather_buffer_device_view =
        Kokkos::View<std::byte *, Kokkos::DefaultExecutionSpace>{
          Kokkos::ViewAllocateWithoutInitializing("gather"),
          max_shared_indices_size * type_size};

      // shared_indices is created on the host, but is accessed from
      // the device. It will be copied to the device on the first iteration.
      // Shared data in the field storage is copied to the gather buffer
      // in parallel. It is then copied to the send buffer (on host) and
      // sent to the peer via MPI_Send.
      for(const auto & [dst_rank, shared_indices] : shared_entities) {
        auto n_elements = shared_indices.size();
        auto n_bytes = n_elements * type_size;
        send_buffers.emplace_back(n_bytes);

        const auto & src_indices = shared_indices;
        std::visit(
          overloaded{[&](const local::detail::host_const_view & src_view) {
                       std::byte * dst = send_buffers.back().data();
                       const std::byte * src = src_view.data();

                       for(std::size_t i = 0; i < src_indices.size(); i++) {
                         std::memcpy(dst + i * type_size,
                           src + src_indices.data()[i] * type_size,
                           type_size);
                       }
                     },
            [&](const local::detail::device_const_view & src) {
              const auto * shared_indices_device_data =
                src_indices.data<exec::task_processor_type_t::toc>();

              Kokkos::parallel_for(
                n_elements, KOKKOS_LAMBDA(const auto & i) {
                  // Yes, memcpy is supported on device as long as there is no
                  // std:: qualifier.
                  memcpy(gather_buffer_device_view.data() + i * type_size,
                    src.data() + shared_indices_device_data[i] * type_size,
                    type_size);
                });

              auto gather_view = Kokkos::subview(gather_buffer_device_view,
                std::pair<std::size_t, std::size_t>(0, n_bytes));
              Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
                local::detail::host_view{send_buffers.back().data(), n_bytes},
                gather_view);
            }},
          source.r->kokkos_view<partition_privilege_t::ro>(data_fid));

        test(MPI_Isend(send_buffers.back().data(),
          int(send_buffers.back().size()),
          MPI_BYTE,
          int(dst_rank),
          0,
          MPI_COMM_WORLD,
          requests()));
      }
    }

    // Construct the view based off the maximum recv_buffer size
    auto scatter_buffer_device_view =
      Kokkos::View<std::byte *, Kokkos::DefaultExecutionSpace>{
        Kokkos::ViewAllocateWithoutInitializing("scatter"),
        max_scatter_buffer_size};

    // ghost_indices is created on the host, but is accessed from
    // the device. It will be copied to the device on the first iteration.
    // Ghost data is received from peers bia MPI_Recv into the
    // recv_buffers. It is then copied to the scatter_buffer (on device)
    // and eventually copied in parallel into the field's storage (on device).

    auto recv_buffer = recv_buffers.begin();
    for(const auto & [src_rank, ghost_indices] : ghost_entities) {
      auto n_elements = ghost_indices.size();
      const auto & dst_indices = ghost_indices;
      std::visit(
        overloaded{[&](const local::detail::host_view & dst_view) {
                     std::byte * dst = dst_view.data();
                     const std::byte * src = recv_buffer->data();

                     for(std::size_t i = 0; i < dst_indices.size(); i++) {
                       std::memcpy(dst + dst_indices.data()[i] * type_size,
                         src + i * type_size,
                         type_size);
                     }
                   },
          [&](const local::detail::device_view & dst) {
            auto scatter_view = Kokkos::subview(scatter_buffer_device_view,
              std::pair<std::size_t, std::size_t>(0, recv_buffer->size()));
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
              scatter_view,
              local::detail::host_view{
                recv_buffer->data(), recv_buffer->size()});

            const auto * ghost_indices_device_data =
              dst_indices.data<exec::task_processor_type_t::toc>();

            Kokkos::parallel_for(
              n_elements, KOKKOS_LAMBDA(const auto & i) {
                memcpy(dst.data() + ghost_indices_device_data[i] * type_size,
                  scatter_buffer_device_view.data() + i * type_size,
                  type_size);
              });
          }},
        destination.r->kokkos_view<partition_privilege_t::wo>(data_fid));
      recv_buffer++;
    }
  }
};

} // namespace data
} // namespace flecsi

#endif // FLECSI_DATA_MPI_COPY_HH
