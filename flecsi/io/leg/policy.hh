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

inline void checkpoint_with_attach_task(const Legion::Task * task,
  const std::vector<Legion::PhysicalRegion> & regions,
  Legion::Context ctx,
  Legion::Runtime * runtime);

inline void checkpoint_without_attach_task(const Legion::Task * task,
  const std::vector<Legion::PhysicalRegion> & regions,
  Legion::Context ctx,
  Legion::Runtime * runtime);

inline void recover_with_attach_task(const Legion::Task * task,
  const std::vector<Legion::PhysicalRegion> & regions,
  Legion::Context ctx,
  Legion::Runtime * runtime);

inline void recover_without_attach_task(const Legion::Task * task,
  const std::vector<Legion::PhysicalRegion> & regions,
  Legion::Context ctx,
  Legion::Runtime * runtime);

/*----------------------------------------------------------------------------*
  HDF5 descriptor of one logical region, not called by users.
 *----------------------------------------------------------------------------*/
struct legion_hdf5_region_t {
  Legion::LogicalRegion logical_region;
  Legion::LogicalPartition logical_partition;
  std::string logical_region_name;
  std::map<Legion::FieldID, std::string> field_string_map;
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

    hid_t group_id;
    auto it = hdf5_groups.find(group_name);
    if(it != hdf5_groups.end()) {
      group_id = H5Gopen2(hdf5_file_id, group_name.c_str(), H5P_DEFAULT);
    }
    else {
      group_id = H5Gcreate2(hdf5_file_id,
        group_name.c_str(),
        H5P_DEFAULT,
        H5P_DEFAULT,
        H5P_DEFAULT);
      hdf5_groups.emplace(group_name);
    }
    if(group_id < 0) {
      if(it != hdf5_groups.end())
        flog(error) << "H5Gopen2 failed: " << group_id << std::endl;
      else
        flog(error) << "H5Gcreate2 failed: " << group_id << std::endl;
      close();
      return false;
    }

    hid_t filetype = H5Tcopy(H5T_C_S1);
    status = H5Tset_size(filetype, H5T_VARIABLE);
    hid_t memtype = H5Tcopy(H5T_C_S1);
    status = H5Tset_size(memtype, H5T_VARIABLE);

    hsize_t dims[1] = {1};
    hid_t dataspace_id = H5Screate_simple(1, dims, NULL);

    const char * data[1];
    data[0] = str.c_str();
    hid_t dset = H5Dcreate2(group_id,
      dataset_name.c_str(),
      filetype,
      dataspace_id,
      H5P_DEFAULT,
      H5P_DEFAULT,
      H5P_DEFAULT);
    status = H5Dwrite(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

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

    char * data[1];
    status = H5Dread(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    str = str + std::string(data[0]);
    H5Fflush(hdf5_file_id, H5F_SCOPE_LOCAL);

    hid_t space = H5Dget_space(dset);
    status = H5Dvlen_reclaim(memtype, space, H5P_DEFAULT, data);
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

inline void
checkpoint_data(const std::string & file_name,
  Legion::IndexSpace launch_space,
  std::vector<legion_hdf5_region_t> & hdf5_region_vector,
  bool attach_flag) {
  Legion::Runtime * runtime = Legion::Runtime::get_runtime();
  Legion::Context ctx = Legion::Runtime::get_context();

  std::vector<std::map<Legion::FieldID, std::string>> field_string_map_vector;
  for(legion_hdf5_region_t & it : hdf5_region_vector) {
    field_string_map_vector.push_back(it.field_string_map);
  }

  std::vector<std::byte> task_args;
  task_args = util::serial_put(std::tie(field_string_map_vector, file_name));

  auto task_id =
    attach_flag
      ? exec::leg::task_id<checkpoint_with_attach_task, loc | inner>
      : exec::leg::task_id<checkpoint_without_attach_task, loc | leaf>;

  Legion::IndexLauncher checkpoint_launcher(task_id,
    launch_space,
    Legion::TaskArgument((void *)(task_args.data()), task_args.size()),
    Legion::ArgumentMap());

  int idx = 0;
  for(legion_hdf5_region_t & it : hdf5_region_vector) {
    checkpoint_launcher.add_region_requirement(
      Legion::RegionRequirement(it.logical_partition,
        0 /*projection ID*/,
        READ_ONLY,
        EXCLUSIVE,
        it.logical_region));

    std::map<Legion::FieldID, std::string> & field_string_map =
      it.field_string_map;
    for(std::pair<const Legion::FieldID, std::string> & it : field_string_map) {
      checkpoint_launcher.region_requirements[idx].add_field(it.first);
    }
    idx++;
  }

  {
    log::devel_guard guard(io_tag);
    flog_devel(info) << "Start checkpoint file " << file_name
                     << " regions size " << hdf5_region_vector.size()
                     << std::endl;
  }

  Legion::FutureMap fumap =
    runtime->execute_index_space(ctx, checkpoint_launcher);
  fumap.wait_all_results();
} // checkpoint_data

inline void
recover_data(const std::string & file_name,
  Legion::IndexSpace launch_space,
  std::vector<legion_hdf5_region_t> & hdf5_region_vector,
  bool attach_flag) {
  Legion::Runtime * runtime = Legion::Runtime::get_runtime();
  Legion::Context ctx = Legion::Runtime::get_context();

  std::vector<std::map<Legion::FieldID, std::string>> field_string_map_vector;
  for(legion_hdf5_region_t & it : hdf5_region_vector) {
    field_string_map_vector.push_back(it.field_string_map);
  }

  std::vector<std::byte> task_args;
  task_args = util::serial_put(std::tie(field_string_map_vector, file_name));

  auto task_id =
    attach_flag ? exec::leg::task_id<recover_with_attach_task, loc | inner>
                : exec::leg::task_id<recover_without_attach_task, loc | leaf>;

  Legion::IndexLauncher recover_launcher(task_id,
    launch_space,
    Legion::TaskArgument((void *)(task_args.data()), task_args.size()),
    Legion::ArgumentMap());
  int idx = 0;
  for(legion_hdf5_region_t & it : hdf5_region_vector) {
    recover_launcher.add_region_requirement(
      Legion::RegionRequirement(it.logical_partition,
        0 /*projection ID*/,
        WRITE_DISCARD,
        EXCLUSIVE,
        it.logical_region));

    std::map<Legion::FieldID, std::string> & field_string_map =
      it.field_string_map;
    for(std::pair<const Legion::FieldID, std::string> & it : field_string_map) {
      recover_launcher.region_requirements[idx].add_field(it.first);
    }
    idx++;
  }

  {
    log::devel_guard guard(io_tag);
    flog_devel(info) << "Start recover file " << file_name << " regions size "
                     << hdf5_region_vector.size() << std::endl;
  }

  Legion::FutureMap fumap = runtime->execute_index_space(ctx, recover_launcher);
  fumap.wait_all_results();
} // recover_data

struct io_interface_t {
  void add_process_topology(int num_files) {
    // TODO:  allow for num_files != # of ranks
    assert(num_files == (int)processes());
    auto & index_runtime_data = process_topology.get();

    std::map<Legion::FieldID, std::string> field_string_map;

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

    std::vector<legion_hdf5_region_t> hdf5_region_vector;
    hdf5_region_vector.push_back(checkpoint_region);
    checkpoint_data(file_name, file.index_space, hdf5_region_vector, true);
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

    std::vector<legion_hdf5_region_t> hdf5_region_vector;
    hdf5_region_vector.push_back(recover_region);
    recover_data(file_name, file.index_space, hdf5_region_vector, true);
  } // recover_process_topology

private:
  struct topology_data {
    data::leg::unique_index_space index_space;
    data::leg::unique_index_partition index_partition;
    data::leg::unique_logical_partition logical_partition;
  };
  std::map<const topo::index::core *, topology_data> file_map;
};

inline void
checkpoint_with_attach_task(const Legion::Task * task,
  const std::vector<Legion::PhysicalRegion> & regions,
  Legion::Context ctx,
  Legion::Runtime * runtime) {

  const int point = task->index_point.point_data[0];

  const std::byte * task_args = (const std::byte *)task->args;

  std::vector<std::map<Legion::FieldID, std::string>> field_string_map_vector;

  field_string_map_vector =
    util::serial_get<std::vector<std::map<Legion::FieldID, std::string>>>(
      task_args);

  std::string fname = util::serial_get<std::string>(task_args);
  fname = fname + std::to_string(point);

  {
    // create files and datasets
    hdf5_t checkpoint_file = hdf5_t::create(fname);
    for(unsigned int rid = 0; rid < regions.size(); rid++) {
      Legion::Rect<2> rect = runtime->get_index_space_domain(
        ctx, task->regions[rid].region.get_index_space());
      size_t domain_size = rect.volume();

      for(Legion::FieldID fid : task->regions[rid].privilege_fields) {
        checkpoint_file.create_dataset(
          std::to_string(fid), domain_size * sizeof(double));
      }
    }
  }

  // write actual data to files
  for(unsigned int rid = 0; rid < regions.size(); rid++) {
    Legion::PhysicalRegion attach_dst_pr;
    Legion::LogicalRegion input_lr = regions[rid].get_logical_region();
    Legion::LogicalRegion attach_dst_lr = runtime->create_logical_region(
      ctx, input_lr.get_index_space(), input_lr.get_field_space());

    Legion::AttachLauncher hdf5_attach_launcher(
      EXTERNAL_HDF5_FILE, attach_dst_lr, attach_dst_lr);
    std::map<Legion::FieldID, const char *> field_map;
    std::set<Legion::FieldID> field_set = task->regions[rid].privilege_fields;
    std::map<Legion::FieldID, std::string>::iterator map_it;
    for(Legion::FieldID it : field_set) {
      map_it = field_string_map_vector[rid].find(it);
      if(map_it != field_string_map_vector[rid].end()) {
        field_map.insert(std::make_pair(it, (map_it->second).c_str()));
      }
      else {
        assert(0);
      }
    }

    {
      log::devel_guard guard(io_tag);
      flog_devel(info) << "Checkpoint data to HDF5 file attach " << fname
                       << " region_id " << rid
                       << " (dataset(fid) size= " << field_map.size() << ")"
                       << " field_string_map_vector(regions) size "
                       << field_string_map_vector.size() << std::endl;
    }

    hdf5_attach_launcher.attach_hdf5(
      fname.c_str(), field_map, LEGION_FILE_READ_WRITE);
    attach_dst_pr =
      runtime->attach_external_resource(ctx, hdf5_attach_launcher);
    // cp_pr.wait_until_valid();

    Legion::CopyLauncher copy_launcher1;
    copy_launcher1.add_copy_requirements(
      Legion::RegionRequirement(input_lr, READ_ONLY, EXCLUSIVE, input_lr),
      Legion::RegionRequirement(
        attach_dst_lr, WRITE_DISCARD, EXCLUSIVE, attach_dst_lr));
    for(Legion::FieldID it : field_set) {
      copy_launcher1.add_src_field(0, it);
      copy_launcher1.add_dst_field(0, it);
    }
    runtime->issue_copy_operation(ctx, copy_launcher1);

    Legion::Future fu =
      runtime->detach_external_resource(ctx, attach_dst_pr, true);
    fu.wait();
    runtime->destroy_logical_region(ctx, attach_dst_lr);
  }
} // checkpoint_with_attach_task

inline void
checkpoint_without_attach_task(const Legion::Task * task,
  const std::vector<Legion::PhysicalRegion> & regions,
  Legion::Context ctx,
  Legion::Runtime * runtime) {

  const int point = task->index_point.point_data[0];

  const std::byte * task_args = (const std::byte *)task->args;

  std::vector<std::map<Legion::FieldID, std::string>> field_string_map_vector;

  field_string_map_vector =
    util::serial_get<std::vector<std::map<Legion::FieldID, std::string>>>(
      task_args);

  std::string fname = util::serial_get<std::string>(task_args);
  fname = fname + std::to_string(point);

  hdf5_t checkpoint_file = hdf5_t::create(fname);
  hid_t file_id = checkpoint_file.hdf5_file_id;

  for(unsigned int rid = 0; rid < regions.size(); rid++) {
    Legion::Rect<2> rect = runtime->get_index_space_domain(
      ctx, task->regions[rid].region.get_index_space());
    size_t domain_size = rect.volume();

    std::set<Legion::FieldID> field_set = task->regions[rid].privilege_fields;
    for(Legion::FieldID it : field_set) {
      auto map_it = field_string_map_vector[rid].find(it);
      if(map_it != field_string_map_vector[rid].end()) {
        checkpoint_file.create_dataset(
          map_it->second, domain_size * sizeof(double));

        const Legion::FieldAccessor<READ_ONLY,
          double,
          2,
          Legion::coord_t,
          Realm::AffineAccessor<double, 2, Legion::coord_t>>
          acc_fid(regions[rid], it);
        const double * dset_data = acc_fid.ptr(rect.lo);
        hid_t dataset_id =
          H5Dopen2(file_id, (map_it->second).c_str(), H5P_DEFAULT);
        if(dataset_id < 0) {
          flog(error) << "H5Dopen2 failed: " << dataset_id << std::endl;
          H5Fclose(file_id);
          assert(0);
        }
        H5Dwrite(
          dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dset_data);
        H5Dclose(dataset_id);
      }
      else {
        assert(0);
      }
    }

    {
      log::devel_guard guard(io_tag);
      flog_devel(info) << "Checkpoint data to HDF5 file no attach " << fname
                       << " region_id " << rid
                       << " (dataset(fid) size= " << field_set.size() << ")"
                       << " field_string_map_vector(regions) size "
                       << field_string_map_vector.size() << std::endl;
    }
  }
} // checkpoint_without_attach_task

inline void
recover_with_attach_task(const Legion::Task * task,
  const std::vector<Legion::PhysicalRegion> & regions,
  Legion::Context ctx,
  Legion::Runtime * runtime) {
  const int point = task->index_point.point_data[0];

  const std::byte * task_args = (const std::byte *)task->args;

  std::vector<std::map<Legion::FieldID, std::string>> field_string_map_vector;

  field_string_map_vector =
    util::serial_get<std::vector<std::map<Legion::FieldID, std::string>>>(
      task_args);

  std::string fname = util::serial_get<std::string>(task_args);
  fname = fname + std::to_string(point);

  for(unsigned int rid = 0; rid < regions.size(); rid++) {
    Legion::PhysicalRegion attach_src_pr;
    Legion::LogicalRegion output_lr = regions[rid].get_logical_region();
    Legion::LogicalRegion attach_src_lr = runtime->create_logical_region(
      ctx, output_lr.get_index_space(), output_lr.get_field_space());

    Legion::AttachLauncher hdf5_attach_launcher(
      EXTERNAL_HDF5_FILE, attach_src_lr, attach_src_lr);
    std::map<Legion::FieldID, const char *> field_map;
    std::set<Legion::FieldID> field_set = task->regions[rid].privilege_fields;
    std::map<Legion::FieldID, std::string>::iterator map_it;
    for(Legion::FieldID it : field_set) {
      map_it = field_string_map_vector[rid].find(it);
      if(map_it != field_string_map_vector[rid].end()) {
        field_map.insert(std::make_pair(it, (map_it->second).c_str()));
      }
      else {
        assert(0);
      }
    }

    {
      log::devel_guard guard(io_tag);
      flog_devel(info) << "Recover data to HDF5 file attach " << fname
                       << " region_id " << rid
                       << " (dataset(fid) size= " << field_map.size() << ")"
                       << " field_string_map_vector(regions) size "
                       << field_string_map_vector.size() << std::endl;
    }

    hdf5_attach_launcher.attach_hdf5(
      fname.c_str(), field_map, LEGION_FILE_READ_WRITE);
    attach_src_pr =
      runtime->attach_external_resource(ctx, hdf5_attach_launcher);

    Legion::CopyLauncher copy_launcher2;
    copy_launcher2.add_copy_requirements(
      Legion::RegionRequirement(
        attach_src_lr, READ_ONLY, EXCLUSIVE, attach_src_lr),
      Legion::RegionRequirement(
        output_lr, WRITE_DISCARD, EXCLUSIVE, output_lr));
    for(Legion::FieldID it : field_set) {
      copy_launcher2.add_src_field(0, it);
      copy_launcher2.add_dst_field(0, it);
    }
    runtime->issue_copy_operation(ctx, copy_launcher2);

    Legion::Future fu =
      runtime->detach_external_resource(ctx, attach_src_pr, true);
    fu.wait();
    runtime->destroy_logical_region(ctx, attach_src_lr);
  }
} // recover_with_attach_task

inline void
recover_without_attach_task(const Legion::Task * task,
  const std::vector<Legion::PhysicalRegion> & regions,
  Legion::Context ctx,
  Legion::Runtime * runtime) {
  const int point = task->index_point.point_data[0];

  const std::byte * task_args = (const std::byte *)task->args;

  std::vector<std::map<Legion::FieldID, std::string>> field_string_map_vector;

  field_string_map_vector =
    util::serial_get<std::vector<std::map<Legion::FieldID, std::string>>>(
      task_args);

  std::string fname = util::serial_get<std::string>(task_args);
  fname = fname + std::to_string(point);

  hdf5_t checkpoint_file = hdf5_t::open(fname);
  hid_t file_id = checkpoint_file.hdf5_file_id;

  for(unsigned int rid = 0; rid < regions.size(); rid++) {
    Legion::Rect<2> rect = runtime->get_index_space_domain(
      ctx, task->regions[rid].region.get_index_space());

    std::set<Legion::FieldID> field_set = task->regions[rid].privilege_fields;
    for(Legion::FieldID it : field_set) {
      auto map_it = field_string_map_vector[rid].find(it);
      if(map_it != field_string_map_vector[rid].end()) {
        const Legion::FieldAccessor<WRITE_DISCARD,
          double,
          2,
          Legion::coord_t,
          Realm::AffineAccessor<double, 2, Legion::coord_t>>
          acc_fid(regions[rid], it);
        double * dset_data = acc_fid.ptr(rect.lo);
        hid_t dataset_id =
          H5Dopen2(file_id, (map_it->second).c_str(), H5P_DEFAULT);
        if(dataset_id < 0) {
          flog(error) << "H5Dopen2 failed: " << dataset_id << std::endl;
          H5Fclose(file_id);
          assert(0);
        }
        H5Dread(
          dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dset_data);
        H5Dclose(dataset_id);
      }
      else {
        assert(0);
      }
    }

    {
      log::devel_guard guard(io_tag);
      flog_devel(info) << "Recover data to HDF5 file no attach " << fname
                       << " region_id " << rid
                       << " (dataset(fid) size= " << field_set.size() << ")"
                       << " field_string_map_vector(regions) size "
                       << field_string_map_vector.size() << std::endl;
    }
  }
} // recover_without_attach_task

} // namespace io
} // namespace flecsi
