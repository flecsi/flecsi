// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_IO_LEG_POLICY_HH
#define FLECSI_IO_LEG_POLICY_HH

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include <hdf5.h>
#include <legion.h>

#include "flecsi/config.hh"
#include "flecsi/data.hh"
#include "flecsi/data/field.hh"
#include "flecsi/data/leg/policy.hh"
#include "flecsi/execution.hh"
#include "flecsi/io/hdf5.hh"
#include "flecsi/run/context.hh"
#include "flecsi/util/serialize.hh"

namespace flecsi {
namespace io {
/// \defgroup legion-io Legion I/O
/// \ingroup io
/// \{

using FieldSizes = std::map<Legion::FieldID, std::size_t>;

// This one task handles all I/O variations: read or Write, Attach or not.
template<bool W, bool A>
inline void
checkpoint_task(const Legion::Task * task,
  const std::vector<Legion::PhysicalRegion> & regions,
  Legion::Context ctx,
  Legion::Runtime * runtime) {
  using F = hdf5::file;

  const int point = task->index_point.point_data[0];

  const std::byte * task_args = (const std::byte *)task->args;

  const auto field_size_map_vector =
    util::serial::get<std::vector<FieldSizes>>(task_args);
  const auto fname =
    util::serial::get<std::string>(task_args) + std::to_string(point);

  F checkpoint_file({});
  if constexpr(A) {
    if constexpr(W) {
      // create files and datasets
      checkpoint_file = F::create(fname);
      for(unsigned int rid = 0; rid < regions.size(); rid++) {
        const auto & rr = task->regions[rid];
        Legion::Rect<2> rect =
          runtime->get_index_space_domain(ctx, rr.region.get_index_space());
        size_t domain_size = rect.volume();
        auto & m = field_size_map_vector[rid];
        std::string pfx = "region " + std::to_string(rid);
        for(Legion::FieldID fid : rr.privilege_fields) {
          std::string name = pfx + " field " + std::to_string(fid);
          checkpoint_file.create_dataset(name, domain_size, m.at(fid));
        }
      }
      checkpoint_file.close();
    }
  }
  else
    checkpoint_file = (W ? F::create : F::open)(fname);

  for(unsigned int rid = 0; rid < regions.size(); rid++) {
    auto & rr = task->regions[rid];
    auto & pr = regions[rid];
    const auto f = [&, &m = field_size_map_vector[rid]](auto g) {
      auto & field_set = rr.privilege_fields;
      for(Legion::FieldID i : field_set)
        g(i, m.at(i));

      {
        flog::devel_guard guard(io_tag);
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
      std::string pfx = "region " + std::to_string(rid);
      f([&field_map, &field_names, &pfx](Legion::FieldID fid, std::size_t) {
        std::string name = pfx + " field " + std::to_string(fid);
        field_names.emplace_back(name);
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
        Legion::RegionRequirement(src, LEGION_READ_ONLY, LEGION_EXCLUSIVE, src),
        Legion::RegionRequirement(
          dest, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, dest));
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
      std::string pfx = "region " + std::to_string(rid);
      f([&](Legion::FieldID fid, std::size_t item_size) {
        std::string name = pfx + " field " + std::to_string(fid);
        if constexpr(W)
          checkpoint_file.create_dataset(name, rect.volume(), item_size);

        const Legion::FieldAccessor<W ? LEGION_READ_ONLY : LEGION_WRITE_DISCARD,
          char,
          2,
          Legion::coord_t,
          Realm::AffineAccessor<char, 2, Legion::coord_t>>
          acc_fid(pr, fid, item_size);
        [] {
          if constexpr(W)
            return H5Dwrite;
          else
            return H5Dread;
        }()(hdf5::dataset(checkpoint_file.hdf5_file_id, name.c_str()),
          hdf5::datatype::bytes(item_size),
          H5S_ALL,
          H5S_ALL,
          H5P_DEFAULT,
          acc_fid.ptr(rect.lo));
      });
    }
  }
}

struct io_interface {

  explicit io_interface(Color ranks_per_file)
    : launch_space([&] {
        int num_files = util::ceil_div(processes(), ranks_per_file);
        // TODO:  allow for num_files != # of ranks
        assert(num_files == (int)processes());
        Legion::Rect<1> file_color_bounds(0, num_files - 1);
        return data::leg::run().create_index_space(
          data::leg::ctx(), file_color_bounds);
      }()),
      launch_partition(data::leg::run().create_equal_partition(data::leg::ctx(),
        process_topology->get_index_space(),
        launch_space)) {}

  template<bool W = true> // whether to write or read the file
  inline void checkpoint_data(const std::string & file_name, bool attach_flag) {
    Legion::Runtime * runtime = Legion::Runtime::get_runtime();
    Legion::Context ctx = Legion::Runtime::get_context();
    auto & context = run::context::instance();
    auto & isd_vector = context.get_index_space_info();
    namespace serial = util::serial;

    std::vector<FieldSizes> field_size_map_vector;
    for(auto & isd : isd_vector) {
      field_size_map_vector.emplace_back(make_field_size_map(isd.fields));
    }
    const auto task_args = serial::buffer([&](auto & p) {
      serial::put(p, field_size_map_vector);
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
    for(auto & isd : isd_vector) {
      checkpoint_launcher.add_region_requirement(
        Legion::RegionRequirement(isd.partition->logical_partition,
          0 /*projection ID*/,
          W ? LEGION_READ_ONLY : LEGION_WRITE_DISCARD,
          LEGION_EXCLUSIVE,
          isd.region->logical_region));

      for(auto & it : field_size_map_vector[idx]) {
        checkpoint_launcher.region_requirements[idx].add_field(it.first);
      }
      idx++;
    }

    {
      flog::devel_guard guard(io_tag);
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
    for(const auto & p : fs) {
      fsm.emplace(p->fid, p->type_size);
    }
    return fsm;
  }

  data::leg::shared_index_space launch_space;
  data::leg::shared_index_partition launch_partition;
};

/// \}
} // namespace io
} // namespace flecsi

#endif
