/*
    @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
   /@@/////  /@@          @@////@@ @@////// /@@
   /@@       /@@  @@@@@  @@    // /@@       /@@
   /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
   /@@////   /@@/@@@@@@@/@@       ////////@@/@@
   /@@       /@@/@@//// //@@    @@       /@@/@@
   /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
   //       ///  //////   //////  ////////  //

   Copyright (c) 2016, Triad National Security, LLC
   All rights reserved.
                                                                              */
#pragma once

/*!  @file */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <ostream>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <flecsi-config.h>

#if !defined(FLECSI_ENABLE_LEGION)
#error FLECSI_ENABLE_LEGION not defined! This file depends on Legion!
#endif

#include <flecsi/data/field.hh>

#include <hdf5.h>
#include <legion.h>

namespace flecsi {
namespace io {
using FieldNames = std::map<Legion::FieldID, std::string>;

/*----------------------------------------------------------------------------*
  HDF5 descriptor of one logical region, not called by users.
 *----------------------------------------------------------------------------*/
struct legion_hdf5_region_t {
  Legion::LogicalRegion logical_region;
  Legion::LogicalPartition logical_partition;
  std::string logical_region_name;
  FieldNames field_string_map;
};

/*----------------------------------------------------------------------------*
  HDF5 file interface.
 *----------------------------------------------------------------------------*/
struct legion_hdf5_t {

  //----------------------------------------------------------------------------//
  // Implementation of legion_hdf5_t::create.
  //----------------------------------------------------------------------------//
  static legion_hdf5_t create(const std::string & file_name) {
    return {{file_name.c_str(), true}};
  }

  //----------------------------------------------------------------------------//
  // Implementation of legion_hdf5_t:open.
  //----------------------------------------------------------------------------//
  static legion_hdf5_t open(const std::string & file_name) {
    return {{file_name.c_str(), false}};
  }

  //----------------------------------------------------------------------------//
  // Implementation of legion_hdf5_t:close.
  //----------------------------------------------------------------------------//
  bool close() {
    assert(hdf5_file_id);
    return hdf5_file_id.close();
  }

  //----------------------------------------------------------------------------//
  // Implementation of legion_hdf5_t::write_string.
  //----------------------------------------------------------------------------//
  bool write_string(const std::string & group_name,
    const std::string & dataset_name,
    const std::string & str) {

    [[maybe_unused]] herr_t status; // FIXME: report errors
    // TODO:FIXME
    // status = H5Eset_auto(NULL, NULL);
    // status = H5Gget_objinfo (hdf5_file_id, group_name, 0, NULL);

    const bool add = hdf5_groups.insert(group_name).second;
    const hid_t group_id =
      add ? H5Gcreate2(hdf5_file_id,
              group_name.c_str(),
              H5P_DEFAULT,
              H5P_DEFAULT,
              H5P_DEFAULT)
          : H5Gopen2(hdf5_file_id, group_name.c_str(), H5P_DEFAULT);
    if(group_id < 0) {
      flog(error) << (add ? "H5Gcreate2" : "H5Gopen2")
                  << " failed: " << group_id << std::endl;
      close();
      return false;
    }

    hid_t filetype = H5Tcopy(H5T_C_S1);
    status = H5Tset_size(filetype, H5T_VARIABLE);
    hid_t memtype = H5Tcopy(H5T_C_S1);
    status = H5Tset_size(memtype, H5T_VARIABLE);

    const hsize_t dim = 1;
    hid_t dataspace_id = H5Screate_simple(1, &dim, NULL);

    const auto data = str.c_str();
    hid_t dset = H5Dcreate2(group_id,
      dataset_name.c_str(),
      filetype,
      dataspace_id,
      H5P_DEFAULT,
      H5P_DEFAULT,
      H5P_DEFAULT);
    status = H5Dwrite(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data);

    H5Fflush(hdf5_file_id, H5F_SCOPE_LOCAL);
    status = H5Dclose(dset);
    status = H5Sclose(dataspace_id);
    status = H5Tclose(filetype);
    status = H5Tclose(memtype);
    status = H5Gclose(group_id);
    return true;
  }

  //----------------------------------------------------------------------------//
  // Implementation of legion_hdf5_t::read_string.
  //----------------------------------------------------------------------------//
  bool read_string(const std::string & group_name,
    const std::string & dataset_name,
    std::string & str) {

    [[maybe_unused]] herr_t status; // FIXME: report errors
    // TODO:FIXME
    // status = H5Eset_auto(NULL, NULL);
    // status = H5Gget_objinfo (hdf5_file_id, group_name, 0, NULL);

    hid_t group_id;
    group_id = H5Gopen2(hdf5_file_id, group_name.c_str(), H5P_DEFAULT);

    if(group_id < 0) {
      flog(error) << "H5Gopen2 failed: " << group_id << std::endl;
      close();
      return false;
    }

    hid_t dset = H5Dopen2(group_id, dataset_name.c_str(), H5P_DEFAULT);

    hid_t filetype = H5Dget_type(dset);
    hid_t memtype = H5Tcopy(H5T_C_S1);
    status = H5Tset_size(memtype, H5T_VARIABLE);

    char * data;
    status = H5Dread(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data);

    str += data;
    H5Fflush(hdf5_file_id, H5F_SCOPE_LOCAL);

    hid_t space = H5Dget_space(dset);
    status = H5Dvlen_reclaim(memtype, space, H5P_DEFAULT, &data);
    status = H5Dclose(dset);
    status = H5Tclose(filetype);
    status = H5Tclose(memtype);
    status = H5Gclose(group_id);
    return true;
  }

  //----------------------------------------------------------------------------//
  // Implementation of legion_hdf5_t::create_dataset.
  //----------------------------------------------------------------------------//
  bool create_dataset(const std::string & field_name, hsize_t size) {
    const hsize_t nsize = (size + (sizeof(double) - 1)) / sizeof(double);

    const hsize_t dims[2] = {nsize, 1};
    const hid_t dataspace_id = H5Screate_simple(2, dims, NULL);
    if(dataspace_id < 0) {
      flog(error) << "H5Screate_simple failed: " << dataspace_id << std::endl;
      close();
      return false;
    }
#if 0
    hid_t group_id = H5Gcreate2(file_id, (*lr_it).logical_region_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (group_id < 0) {
      printf("H5Gcreate2 failed: %lld\n", (long long)group_id);
      H5Sclose(dataspace_id);
      close();
      return false;
    }
#endif
    hid_t dataset = H5Dcreate2(hdf5_file_id,
      field_name.c_str(),
      H5T_IEEE_F64LE,
      dataspace_id,
      H5P_DEFAULT,
      H5P_DEFAULT,
      H5P_DEFAULT);
    if(dataset < 0) {
      flog(error) << "H5Dcreate2 failed: " << dataset << std::endl;
      //    H5Gclose(group_id);
      H5Sclose(dataspace_id);
      close();
      return false;
    }
    H5Dclose(dataset);
    //   H5Gclose(group_id);
    H5Sclose(dataspace_id);
    H5Fflush(hdf5_file_id, H5F_SCOPE_LOCAL);
    return true;
  }

  //----------------------------------------------------------------------------//
  // Implementation of legion_hdf5_t::legion_hdf5_t.
  //----------------------------------------------------------------------------//
  legion_hdf5_t(hdf5 h) : hdf5_file_id(std::move(h)) {}

  hdf5 hdf5_file_id;
  std::set<std::string> hdf5_groups;
};

/*----------------------------------------------------------------------------*
  Legion HDF5 checkpoint interface.
 *----------------------------------------------------------------------------*/
using hdf5_t = legion_hdf5_t;
using hdf5_region_t = legion_hdf5_region_t;
using launch_space_t = Legion::IndexSpace;

// This one task handles all I/O variations: read or Write, Attach or not.
template<bool W, bool A>
inline void
checkpoint_task(const Legion::Task * task,
  const std::vector<Legion::PhysicalRegion> & regions,
  Legion::Context ctx,
  Legion::Runtime * runtime) {

  const int point = task->index_point.point_data[0];

  const std::byte * task_args = (const std::byte *)task->args;

  const auto field_string_map_vector =
    util::serial_get<std::vector<FieldNames>>(task_args);
  const auto fname =
    util::serial_get<std::string>(task_args) + std::to_string(point);

  hdf5_t checkpoint_file({});
  if constexpr(A) {
    if constexpr(W) {
      // create files and datasets
      checkpoint_file = hdf5_t::create(fname);
      for(unsigned int rid = 0; rid < regions.size(); rid++) {
        const auto & rr = task->regions[rid];
        Legion::Rect<2> rect =
          runtime->get_index_space_domain(ctx, rr.region.get_index_space());
        size_t domain_size = rect.volume();

        for(Legion::FieldID fid : rr.privilege_fields) {
          checkpoint_file.create_dataset(
            std::to_string(fid), domain_size * sizeof(double));
        }
      }
      checkpoint_file.close();
    }
  }
  else
    checkpoint_file = (W ? hdf5_t::create : hdf5_t::open)(fname);

  for(unsigned int rid = 0; rid < regions.size(); rid++) {
    auto & rr = task->regions[rid];
    auto & pr = regions[rid];
    const auto f = [&, &m = field_string_map_vector[rid]](auto g) {
      auto & field_set = rr.privilege_fields;
      for(Legion::FieldID i : field_set)
        g(i, m.at(i));

      {
        log::devel_guard guard(io_tag);
        flog_devel(info) << (W ? "Checkpoint" : "Recover")
                         << " data to HDF5 file " << (A ? "" : "no ")
                         << "attach " << fname << " region_id " << rid
                         << " (dataset(fid) size= " << field_set.size() << ")"
                         << " field_string_map_vector(regions) size "
                         << field_string_map_vector.size() << std::endl;
      }
    };

    if constexpr(A) {
      Legion::PhysicalRegion attach_pr;
      Legion::LogicalRegion field_lr = pr.get_logical_region();
      Legion::LogicalRegion attach_lr = runtime->create_logical_region(
        ctx, field_lr.get_index_space(), field_lr.get_field_space());

      Legion::AttachLauncher hdf5_attach_launcher(
        EXTERNAL_HDF5_FILE, attach_lr, attach_lr);
      std::map<Legion::FieldID, const char *> field_map;
      f([&field_map](Legion::FieldID it, const std::string & n) {
        field_map.emplace(it, n.c_str());
      });

      hdf5_attach_launcher.attach_hdf5(
        fname.c_str(), field_map, LEGION_FILE_READ_WRITE);
      attach_pr = runtime->attach_external_resource(ctx, hdf5_attach_launcher);
      // cp_pr.wait_until_valid();

      Legion::CopyLauncher copy_launcher1;
      const Legion::LogicalRegion &src = W ? field_lr : attach_lr,
                                  &dest = W ? attach_lr : field_lr;
      copy_launcher1.add_copy_requirements(
        Legion::RegionRequirement(src, READ_ONLY, EXCLUSIVE, src),
        Legion::RegionRequirement(dest, WRITE_DISCARD, EXCLUSIVE, dest));
      for(const auto & fn : field_map) {
        copy_launcher1.add_src_field(0, fn.first);
        copy_launcher1.add_dst_field(0, fn.first);
      }
      runtime->issue_copy_operation(ctx, copy_launcher1);

      Legion::Future fu =
        runtime->detach_external_resource(ctx, attach_pr, true);
      fu.wait();
      runtime->destroy_logical_region(ctx, attach_lr);
    }
    else {
      Legion::Rect<2> rect =
        runtime->get_index_space_domain(ctx, rr.region.get_index_space());
      f([&](Legion::FieldID it, const std::string & n) {
        if constexpr(W)
          checkpoint_file.create_dataset(n, rect.volume() * sizeof(double));

        const Legion::FieldAccessor<W ? READ_ONLY : WRITE_DISCARD,
          double,
          2,
          Legion::coord_t,
          Realm::AffineAccessor<double, 2, Legion::coord_t>>
          acc_fid(pr, it);
        auto * const dset_data = acc_fid.ptr(rect.lo);
        hid_t dataset_id =
          H5Dopen2(checkpoint_file.hdf5_file_id, n.c_str(), H5P_DEFAULT);
        if(dataset_id < 0) {
          flog(error) << "H5Dopen2 failed: " << dataset_id << std::endl;
          H5Fclose(checkpoint_file.hdf5_file_id);
          assert(0);
        }
        [] {
          if constexpr(W)
            return H5Dwrite;
          else
            return H5Dread;
        }()(
          dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dset_data);
        H5Dclose(dataset_id);
      });
    }
  }
}

template<bool W = true> // whether to write or read the file
inline void
checkpoint_data(const std::string & file_name,
  Legion::IndexSpace launch_space,
  const std::vector<legion_hdf5_region_t> & hdf5_region_vector,
  bool attach_flag) {
  Legion::Runtime * runtime = Legion::Runtime::get_runtime();
  Legion::Context ctx = Legion::Runtime::get_context();

  const auto task_args = util::serial_buffer([&](auto & p) {
    util::serial_put(p, hdf5_region_vector.size());
    for(auto & h : hdf5_region_vector)
      util::serial_put(p, h.field_string_map);
    util::serial_put(p, file_name);
  });

  const auto task_id =
    attach_flag ? exec::leg::task_id<checkpoint_task<W, true>, loc | inner>
                : exec::leg::task_id<checkpoint_task<W, false>, loc | leaf>;

  Legion::IndexLauncher checkpoint_launcher(task_id,
    launch_space,
    Legion::TaskArgument((void *)(task_args.data()), task_args.size()),
    Legion::ArgumentMap());

  int idx = 0;
  for(auto & it : hdf5_region_vector) {
    checkpoint_launcher.add_region_requirement(
      Legion::RegionRequirement(it.logical_partition,
        0 /*projection ID*/,
        W ? READ_ONLY : WRITE_DISCARD,
        EXCLUSIVE,
        it.logical_region));

    for(auto & it : it.field_string_map) {
      checkpoint_launcher.region_requirements[idx].add_field(it.first);
    }
    idx++;
  }

  {
    log::devel_guard guard(io_tag);
    flog_devel(info) << "Start " << (W ? "checkpoint" : "recover") << " file "
                     << file_name << " regions size "
                     << hdf5_region_vector.size() << std::endl;
  }

  Legion::FutureMap fumap =
    runtime->execute_index_space(ctx, checkpoint_launcher);
  fumap.wait_all_results();
} // checkpoint_data

inline void
recover_data(const std::string & file_name,
  Legion::IndexSpace launch_space,
  const std::vector<legion_hdf5_region_t> & hdf5_region_vector,
  bool attach_flag) {
  checkpoint_data<false>(
    file_name, launch_space, hdf5_region_vector, attach_flag);
} // recover_data

struct io_interface_t {
  void add_process_topology(int num_files) {
    // TODO:  allow for num_files != # of ranks
    assert(num_files == (int)processes());
    auto & index_runtime_data = process_topology.get();

    FieldNames field_string_map;

    for(const auto p :
      run::context::instance().get_field_info_store<topo::index>()) {
      field_string_map[p->fid] = std::to_string(p->fid);
    }

    Legion::Runtime * runtime = Legion::Runtime::get_runtime();
    Legion::Context ctx = Legion::Runtime::get_context();
    Legion::Rect<1> file_color_bounds(0, num_files - 1);
    data::leg::unique_index_space process_topology_file_is =
      runtime->create_index_space(ctx, file_color_bounds);
#if 0 
    process_topology_file_ip = runtime->create_pending_partition(ctx, index_runtime_data.index_space, process_topology_file_is);
    int idx = 0; 
    int num_subregions = index_runtime_data.colors;
    for (int point = 0; point < hdf5_file.num_files; point++) {
      std::vector<IndexSpace> subspaces;
      for (int i = 0; i < num_subregions/hdf5_file.num_files; i++) {
        subspaces.push_back(runtime->get_index_subspace(ctx, index_runtime_data.color_partition.get_index_partition(), idx));
        idx ++;
      }
      runtime->create_index_space_union(ctx, process_topology_file_ip, point, subspaces);
    }
#else
    data::leg::unique_index_partition process_topology_file_ip =
      runtime->create_equal_partition(
        ctx, index_runtime_data.index_space, process_topology_file_is);
#endif
    data::leg::unique_logical_partition process_topology_file_lp =
      runtime->get_logical_partition(
        ctx, index_runtime_data.logical_region, process_topology_file_ip);

    file_map[&index_runtime_data] = {std::move(process_topology_file_is),
      std::move(process_topology_file_ip),
      std::move(process_topology_file_lp)};
  } // add_process_topology

  void checkpoint_process_topology(const std::string & file_name) {
    auto const & fid_vector =
      run::context::instance().get_field_info_store<topo::index>();

    auto & index_runtime_data = process_topology.get();
    {
      log::devel_guard guard(io_tag);
      flog_devel(info) << "Checkpoint default index topology, fields size "
                       << fid_vector.size() << std::endl;
    }

    const auto & file = file_map[&index_runtime_data];
    legion_hdf5_region_t checkpoint_region{index_runtime_data.logical_region,
      file.logical_partition,
      "process_topology",
      {}};
    for(const auto p : fid_vector) {
      checkpoint_region.field_string_map[p->fid] = std::to_string(p->fid);
    }

    checkpoint_data(
      file_name, file.index_space, {std::move(checkpoint_region)}, true);
  } // checkpoint_process_topology

  void recover_process_topology(const std::string & file_name) {
    auto const & fid_vector =
      run::context::instance().get_field_info_store<topo::index>();

    auto & index_runtime_data = process_topology.get();

    {
      log::devel_guard guard(io_tag);
      flog_devel(info) << "Recover default index topology, fields size "
                       << fid_vector.size() << std::endl;
    }

    const auto & file = file_map[&index_runtime_data];
    legion_hdf5_region_t recover_region{index_runtime_data.logical_region,
      file.logical_partition,
      "process_topology",
      {}};
    for(const auto p : fid_vector) {
      recover_region.field_string_map[p->fid] = std::to_string(p->fid);
    }

    recover_data(
      file_name, file.index_space, {std::move(recover_region)}, true);
  } // recover_process_topology

private:
  struct topology_data {
    data::leg::unique_index_space index_space;
    data::leg::unique_index_partition index_partition;
    data::leg::unique_logical_partition logical_partition;
  };
  std::map<const topo::index::core *, topology_data> file_map;
};

} // namespace io
} // namespace flecsi
