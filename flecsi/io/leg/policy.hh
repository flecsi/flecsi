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
#include <string>
#include <utility>
#include <vector>

#include <flecsi-config.h>

#if !defined(FLECSI_ENABLE_LEGION)
#error FLECSI_ENABLE_LEGION not defined! This file depends on Legion!
#endif

#include "flecsi/data.hh"
#include "flecsi/data/field.hh"
#include "flecsi/data/leg/policy.hh"
#include "flecsi/execution.hh"
#include "flecsi/io/hdf5.hh"
#include "flecsi/run/context.hh"
#include "flecsi/util/serialize.hh"

#include <hdf5.h>
#include <legion.h>

namespace flecsi {
namespace io {
using FieldSizes = std::map<Legion::FieldID, std::size_t>;

// This one task handles all I/O variations: read or Write, Attach or not.
template<bool W, bool A>
inline void
checkpoint_task(const Legion::Task * task,
  const std::vector<Legion::PhysicalRegion> & regions,
  Legion::Context ctx,
  Legion::Runtime * runtime) {

  const int point = task->index_point.point_data[0];

  const std::byte * task_args = (const std::byte *)task->args;

  const auto field_size_map_vector =
    util::serial_get<std::vector<FieldSizes>>(task_args);
  const auto fname =
    util::serial_get<std::string>(task_args) + std::to_string(point);

  hdf5 checkpoint_file({});
  if constexpr(A) {
    if constexpr(W) {
      // create files and datasets
      checkpoint_file = hdf5::create(fname);
      for(unsigned int rid = 0; rid < regions.size(); rid++) {
        const auto & rr = task->regions[rid];
        Legion::Rect<2> rect =
          runtime->get_index_space_domain(ctx, rr.region.get_index_space());
        size_t domain_size = rect.volume();
        auto & m = field_size_map_vector[rid];

        for(Legion::FieldID fid : rr.privilege_fields) {
          checkpoint_file.create_dataset(
            "field " + std::to_string(fid), domain_size, m.at(fid));
        }
      }
      checkpoint_file.close();
    }
  }
  else
    checkpoint_file = (W ? hdf5::create : hdf5::open)(fname);

  for(unsigned int rid = 0; rid < regions.size(); rid++) {
    auto & rr = task->regions[rid];
    auto & pr = regions[rid];
    const auto f = [&, &m = field_size_map_vector[rid]](auto g) {
      auto & field_set = rr.privilege_fields;
      for(Legion::FieldID i : field_set)
        g(i, m.at(i));

      {
        log::devel_guard guard(io_tag);
        flog_devel(info) << (W ? "Checkpoint" : "Recover")
                         << " data to HDF5 file " << (A ? "" : "no ")
                         << "attach " << fname << " region_id " << rid
                         << " (dataset(fid) size= " << field_set.size() << ")"
                         << " field_size_map_vector(regions) size "
                         << field_size_map_vector.size() << std::endl;
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
      // We need to create a map with C pointers to field names -
      // the Legion interface requires this.  But we also need for
      // the names to persist outside of the lambda below.  So the
      // names will be stored as strings in the field_names vector
      // to make sure their data persists.
      std::vector<std::string> field_names;
      field_names.reserve(rr.privilege_fields.size());
      f([&field_map, &field_names](Legion::FieldID fid, std::size_t) {
        field_names.emplace_back("field " + std::to_string(fid));
        field_map.emplace(fid, field_names.back().c_str());
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
      f([&](Legion::FieldID fid, std::size_t item_size) {
        std::string name = "field " + std::to_string(fid);
        if constexpr(W)
          checkpoint_file.create_dataset(name, rect.volume(), item_size);

        const Legion::FieldAccessor<W ? READ_ONLY : WRITE_DISCARD,
          int,
          2,
          Legion::coord_t,
          Realm::AffineAccessor<int, 2, Legion::coord_t>>
          acc_fid(pr, fid);
        auto * const dset_data = acc_fid.ptr(rect.lo);
        hid_t dataset_id =
          H5Dopen2(checkpoint_file.hdf5_file_id, name.c_str(), H5P_DEFAULT);
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
        }()(dataset_id,
          hdf5_type(item_size),
          H5S_ALL,
          H5S_ALL,
          H5P_DEFAULT,
          dset_data);
        H5Dclose(dataset_id);
      });
    }
  }
}

struct io_interface {

  explicit io_interface(int num_files)
    : launch_space([&] {
        // TODO:  allow for num_files != # of ranks
        assert(num_files == (int)processes());
        Legion::Rect<1> file_color_bounds(0, num_files - 1);
        return data::leg::run().create_index_space(
          data::leg::ctx(), file_color_bounds);
      }()),
      launch_partition(data::leg::run().create_equal_partition(data::leg::ctx(),
        process_topology->index_space,
        launch_space)) {}

  template<bool W = true> // whether to write or read the file
  inline void checkpoint_data(const std::string & file_name, bool attach_flag) {
    Legion::Runtime * runtime = Legion::Runtime::get_runtime();
    Legion::Context ctx = Legion::Runtime::get_context();
    auto & context = run::context::instance();
    auto & isd_vector = context.get_index_space_info();

    std::vector<FieldSizes> field_size_map_vector;
    for(auto & isd : isd_vector) {
      field_size_map_vector.emplace_back(make_field_size_map(*(isd.fields)));
    }
    const auto task_args = util::serial_buffer([&](auto & p) {
      util::serial_put(p, field_size_map_vector);
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
    for(auto & isd : isd_vector) {
      checkpoint_launcher.add_region_requirement(
        Legion::RegionRequirement(isd.partition->logical_partition,
          0 /*projection ID*/,
          W ? READ_ONLY : WRITE_DISCARD,
          EXCLUSIVE,
          isd.region->logical_region));

      for(auto & it : field_size_map_vector[idx]) {
        checkpoint_launcher.region_requirements[idx].add_field(it.first);
      }
      idx++;
    }

    {
      log::devel_guard guard(io_tag);
      flog_devel(info) << "Start " << (W ? "checkpoint" : "recover") << " file "
                       << file_name << " regions size " << isd_vector.size()
                       << std::endl;
    }

    Legion::FutureMap fumap =
      runtime->execute_index_space(ctx, checkpoint_launcher);
    fumap.wait_all_results();
  }

  inline void checkpoint_all_fields(const std::string & file_name,
    bool attach_flag = true) {
    checkpoint_data<true>(file_name, attach_flag);
  } // checkpoint_data

  inline void recover_all_fields(const std::string & file_name,
    bool attach_flag = true) {
    checkpoint_data<false>(file_name, attach_flag);
  } // recover_data

private:
  static FieldSizes make_field_size_map(const data::fields & fs) {
    FieldSizes fsm;
    for(const auto p : fs) {
      fsm.emplace(p->fid, p->type_size);
    }
    return fsm;
  }

private:
  data::leg::unique_index_space launch_space;
  data::leg::unique_index_partition launch_partition;
};

} // namespace io
} // namespace flecsi
