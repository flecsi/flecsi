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
/// \defgroup legion-io Legion I/O
/// \ingroup io
/// \{

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
    util::serial::get<std::vector<FieldNames>>(task_args);
  const auto fname =
    util::serial::get<std::string>(task_args) + std::to_string(point);

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
        auto & m = field_string_map_vector[rid];

        for(Legion::FieldID fid : rr.privilege_fields) {
          checkpoint_file.create_dataset(
            m.at(fid), domain_size * sizeof(double));
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
    namespace serial = util::serial;

    const auto task_args = serial::buffer([&](auto & p) {
      serial::put(p, hdf5_region_vector.size());
      for(auto & h : hdf5_region_vector)
        serial::put(p, h.field_string_map);
      serial::put(p, file_name);
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
  }

  template<class Topo, typename Topo::index_space Index = Topo::default_space()>
  void add_region(typename Topo::slot & slot) {
    auto & fs = run::context::instance().get_field_info_store<Topo, Index>();
    FieldNames fn;
    for(const auto p : fs) {
      // TODO:  handle types other than double
      if(p->name != "double")
        continue;
      auto & i = name_count[p->name];
      fn.emplace(p->fid, p->name + " #" + std::to_string(++i));
    }
    hdf5_region_vector.emplace_back(
      legion_hdf5_region_t{(slot->template get_region<Index>().logical_region),
        (slot->template get_partition<Index>().logical_partition),
        util::type<Topo>() + '[' + std::to_string(Index) + ']',
        std::move(fn)});
  }

  inline void checkpoint_all_fields(const std::string & file_name,
    bool attach_flag = true) {
    checkpoint_data<true>(file_name, attach_flag);
  } // checkpoint_data

  inline void recover_all_fields(const std::string & file_name,
    bool attach_flag = true) {
    checkpoint_data<false>(file_name, attach_flag);
  } // recover_data

  data::leg::unique_index_space launch_space;
  data::leg::unique_index_partition launch_partition;
  std::vector<legion_hdf5_region_t> hdf5_region_vector;
  std::map<std::string, unsigned> name_count;
};

/// \}
} // namespace io
} // namespace flecsi
