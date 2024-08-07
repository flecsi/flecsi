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

    {
      std::vector<std::vector<std::byte>> send_buffers;
      util::mpi::auto_requests requests(
        ghost_entities.size() + shared_entities.size());

      for(const auto & [src_rank, ghost_indices] : ghost_entities) {
        recv_buffers.emplace_back(ghost_indices.size() * type_size);
        test(MPI_Irecv(recv_buffers.back().data(),
          int(recv_buffers.back().size()),
          MPI_BYTE,
          int(src_rank),
          0,
          MPI_COMM_WORLD,
          requests()));
      }

      for(const auto & [dst_rank, shared_indices] : shared_entities) {
        auto n_elements = shared_indices.size();
        auto n_bytes = n_elements * type_size;
        send_buffers.emplace_back(n_bytes);

        // We can not capture shared_indices in the overloaded lambda inside
        // std::visit directly
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
              // copy shared indices from host to device
              const auto * shared_indices_device_data =
                src_indices.data<exec::task_processor_type_t::toc>();

              // allocate gather buffer on device
              auto gather_buffer_device_view =
                Kokkos::View<std::byte *, Kokkos::DefaultExecutionSpace>{
                  "gather", n_bytes};

              // copy shared values to gather buffer on device in parallel, for
              // each element
              Kokkos::parallel_for(
                n_elements, KOKKOS_LAMBDA(const auto & i) {
                  // Yes, memcpy is supported on device as long as there is no
                  // std:: qualifier.
                  memcpy(gather_buffer_device_view.data() + i * type_size,
                    src.data() + shared_indices_device_data[i] * type_size,
                    type_size);
                });

              // copy gathered shared values to send_buffer on host to be sent
              // through MPI.
              Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
                local::detail::host_view{send_buffers.back().data(), n_bytes},
                gather_buffer_device_view);
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

    // copy from intermediate receive buffer to destination storage
    auto recv_buffer = recv_buffers.begin();
    for(const auto & [src_rank, ghost_indices] : ghost_entities) {
      auto n_elements = ghost_indices.size();
      // We can not capture ghost_indices in the overloaded lambda inside
      // std::visit directly
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
            // get the ghost indices from the device
            const auto * ghost_indices_device_data =
              dst_indices.template data<exec::task_processor_type_t::toc>();

            // copy recv buffer from host to scatter buffer on device
            auto scatter_buffer_device_view = [](const auto & hvec,
                                                const std::string & label) {
              using T = typename std::decay_t<decltype(hvec)>::value_type;
              Kokkos::View<T *, Kokkos::DefaultExecutionSpace> dview{
                Kokkos::ViewAllocateWithoutInitializing(label), hvec.size()};
              Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
                dview,
                Kokkos::View<const T *, Kokkos::HostSpace>{
                  hvec.data(), hvec.size()});
              return dview;
            }(*recv_buffer, "scatter");

            // copy ghost values from scatter buffer on device to field storage
            // in parallel, for each element
            Kokkos::parallel_for(
              n_elements, KOKKOS_LAMBDA(const auto & i) {
                memcpy(dst.data() + ghost_indices_device_data[i] * type_size,
                  scatter_buffer_device_view.data() + i * type_size,
                  type_size);
              });
          }},
        destination.kokkos_view<partition_privilege_t::wo>(data_fid));
      recv_buffer++;
    }
  }
};

} // namespace data
} // namespace flecsi

#endif // FLECSI_DATA_MPI_COPY_HH
