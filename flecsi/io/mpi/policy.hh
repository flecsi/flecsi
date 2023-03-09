// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_IO_MPI_POLICY_HH
#define FLECSI_IO_MPI_POLICY_HH

/*!  @file */

#include <cstddef>
#include <hdf5.h>
#include <mpi.h>
#include <ostream>
#include <string>

#include "flecsi-config.h"

#if !defined(FLECSI_ENABLE_MPI)
#error FLECSI_ENABLE_MPI not defined! This file depends on MPI!
#endif

#if !defined(H5_HAVE_PARALLEL)
#error H5_HAVE_PARALLEL not defined! This file depends on parallel HDF5!
#endif

#include "flecsi/data/mpi/policy.hh"
#include "flecsi/io/hdf5.hh"
#include "flecsi/run/context.hh"
#include "flecsi/util/mpi.hh"

namespace flecsi {
namespace io {

const auto hsize_mpi_type = util::mpi::static_type<hsize_t>();

struct io_interface {

  explicit io_interface(Color ranks_per_file = 1) {

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    num_files = util::ceil_div(Color(world_size), ranks_per_file);

    MPI_Comm new_comm;
    new_color = rank / ranks_per_file;
    MPI_Comm_split(MPI_COMM_WORLD, new_color, rank, &new_comm);

    mpi_hdf5_comm = new_comm;

    MPI_Comm_size(new_comm, &new_world_size);
    MPI_Comm_rank(new_comm, &new_rank);
  }

  ~io_interface() {
    MPI_Comm_free(&mpi_hdf5_comm);
  }

  template<bool W>
  using buffer_ptr = std::conditional_t<W, const void, void> *;

  template<bool W = true> // whether to write or read the file
  void checkpoint_field_data(const hid_t & hdf5_file_id,
    const std::string & dataset_name,
    buffer_ptr<W> buffer,
    hsize_t nitems,
    hsize_t displ,
    hsize_t item_size) {
    using namespace hdf5;

    const dataset dataset_id(hdf5_file_id, dataset_name.c_str());

    const hsize_t count[2] = {nitems, 1};
    const hsize_t offset[2] = {displ, 0};
    const dataspace mem_dataspace_id(count), file_dataspace_id(dataset_id);

    // Select hyperslab in the file.
    H5Sselect_hyperslab(
      file_dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);

    // Create property list for collective dataset write.
    const plist xfer_plist_id(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer_plist_id, H5FD_MPIO_COLLECTIVE);

    [] {
      if constexpr(W)
        return H5Dwrite;
      else
        return H5Dread;
    }()(dataset_id,
      datatype::bytes(item_size),
      mem_dataspace_id,
      file_dataspace_id,
      xfer_plist_id,
      buffer);
  }

  template<bool W = true> // whether to write or read the file
  void checkpoint_field(hdf5::file & checkpoint_file,
    const std::string & field_name,
    buffer_ptr<W> buffer,
    const hsize_t nitems,
    const hsize_t item_size) {
    if constexpr(W) {
      hsize_t sum_nitems = 0;
      MPI_Allreduce(
        &nitems, &sum_nitems, 1, hsize_mpi_type, MPI_SUM, mpi_hdf5_comm);
      checkpoint_file.create_dataset(field_name.data(), sum_nitems, item_size);
    }

    hsize_t displ;
    MPI_Exscan(&nitems, &displ, 1, hsize_mpi_type, MPI_SUM, mpi_hdf5_comm);
    if(new_rank == 0)
      displ = 0;

    hid_t hdf5_file_id = (hid_t)checkpoint_file.hdf5_file_id;
    checkpoint_field_data<W>(
      hdf5_file_id, field_name.data(), buffer, nitems, displ, item_size);
  } // checkpoint_field

  template<bool W = true> // whether to write or read the file
  inline void checkpoint_data(const std::string & file_name_in) {
    using F = hdf5::file;
    std::string file_name = file_name_in + std::to_string(new_color);
    F checkpoint_file = (W ? F::pcreate : F::popen)(file_name, mpi_hdf5_comm);

    // checkpoint
    auto & context = run::context::instance();
    auto & isd_vector = context.get_index_space_info();

    {
      flog::devel_guard guard(io_tag);
      flog_devel(info) << (W ? "Checkpoint" : "Recover")
                       << " data to HDF5 file " << file_name << " regions size "
                       << isd_vector.size() << std::endl;
    }

    int idx = 0;
    for(auto & isd : isd_vector) {
      std::string region_name = "region " + std::to_string(idx);
      for(const auto & fp : isd.fields) {
        field_id_t fid = fp->fid;
        int item_size = fp->type_size;
        std::string field_name = region_name + " field " + std::to_string(fid);

        const auto data = isd.partition->get_raw_storage(fid, item_size);
        hsize_t size = data.size() / item_size;
        void * buffer = data.data();
        checkpoint_field<W>(
          checkpoint_file, field_name, buffer, size, item_size);
      }
      idx++;
    }
  }

  inline void checkpoint_all_fields(const std::string & file_name,
    bool = true) {
    checkpoint_data<true>(file_name);
  } // checkpoint_data

  inline void recover_all_fields(const std::string & file_name, bool = true) {
    checkpoint_data<false>(file_name);
  } // recover_data

private:
  int num_files;

  int world_size, rank, new_world_size, new_rank;
  int new_color;

  MPI_Comm mpi_hdf5_comm;
};

} // namespace io
} // namespace flecsi

#endif
