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
  copy_engine(const data::points & pts,
    const data::intervals & intervals,
    field_id_t meta_fid /* for remote shared entities */)
    : local::copy_engine(pts, intervals, meta_fid) {
    init_copy_engine([](auto const & remote_shared_entities) {
      return util::mpi::all_to_allv([&](int r, int) -> auto & {
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

    // Since we are doing ghost copy via MPI, we always want the host side
    // version. Unless we are doing some direct CUDA MPI thing in the future.
    auto source_storage =
      source.r->get_storage<std::byte>(data_fid, max_local_source_idx);
    auto destination_storage = destination.get_storage<std::byte>(data_fid);
    auto type_size = source.r->get_field_info(data_fid)->type_size;

    std::vector<MPI_Request> requests;
    requests.reserve(ghost_entities.size() + shared_entities.size());

    std::vector<std::vector<std::byte>> recv_buffers;
    for(const auto & [src_rank, ghost_indices] : ghost_entities) {
      recv_buffers.emplace_back(ghost_indices.size() * type_size);
      requests.resize(requests.size() + 1);
      test(MPI_Irecv(recv_buffers.back().data(),
        int(recv_buffers.back().size()),
        MPI_BYTE,
        int(src_rank),
        0,
        MPI_COMM_WORLD,
        &requests.back()));
    }

    std::vector<std::vector<std::byte>> send_buffers;
    for(const auto & [dst_rank, shared_indices] : shared_entities) {
      requests.resize(requests.size() + 1);
      send_buffers.emplace_back(shared_indices.size() * type_size);
      std::size_t i = 0;
      for(auto shared_idx : shared_indices) {
        std::memcpy(send_buffers.back().data() + i++ * type_size,
          source_storage.data() + shared_idx * type_size,
          type_size);
      }
      test(MPI_Isend(send_buffers.back().data(),
        int(send_buffers.back().size()),
        MPI_BYTE,
        int(dst_rank),
        0,
        MPI_COMM_WORLD,
        &requests.back()));
    }

    std::vector<MPI_Status> status(requests.size());
    test(MPI_Waitall(int(requests.size()), requests.data(), status.data()));

    // copy from intermediate receive buffer to destination storage
    auto recv_buffer = recv_buffers.begin();
    for(const auto & [src_rank, ghost_indices] : ghost_entities) {
      std::size_t i = 0;
      for(auto ghost_idx : ghost_indices) {
        std::memcpy(destination_storage.data() + ghost_idx * type_size,
          recv_buffer->data() + i++ * type_size,
          type_size);
      }
      recv_buffer++;
    }
  }
};

} // namespace data
} // namespace flecsi

#endif // FLECSI_DATA_MPI_COPY_HH
