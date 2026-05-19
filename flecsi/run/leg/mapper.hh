// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUN_LEG_MAPPER_HH
#define FLECSI_RUN_LEG_MAPPER_HH

#include "../backend.hh"
#include "flecsi/util/color_map.hh"

#include <legion.h>
#include <mappers/default_mapper.h>

#include <iomanip>

namespace flecsi {
namespace run {
/// \addtogroup legion-runtime
/// \{

/*!
  FleCSI's mapper, named for its support for rank-matching for MPI tasks.
  \ns::run.
*/
class mpi_mapper_t : public Legion::Mapping::DefaultMapper {
public:
  /// See \c DefaultMapper for parameter meanings.
  mpi_mapper_t(Legion::Machine machine,
    Legion::Runtime * _runtime,
    Legion::Processor local)
    : Legion::Mapping::DefaultMapper(_runtime->get_mapper_runtime(),
        machine,
        local,
        "default"),
      target_mem(closest()) {
    memoize = true; // as set by -dm:memoize
    assert(target_mem.exists());
    {
      const auto sock = maybe_fb(Legion::Memory::SOCKET_MEM);
      const auto numa = closest(sock);
      for(auto p : processors(local_kind))
        if(closest(sock, p) == numa)
          numa_domain.push_back(p);
    }
  }

  void select_task_options(const Legion::Mapping::MapperContext ctx,
    const Legion::Task & task,
    Legion::Mapping::Mapper::TaskOptions & output) override {
    DefaultMapper::select_task_options(ctx, task, output);
    // Mysteriously, the top-level task has 16 bytes of argument.
    if(task.arglen == sizeof(task_idx))
      context::instance()
        .params.at(get1<task_idx>(task))
        .post(task.is_index_space
                ? util::equal_map(
                    task.index_domain.get_volume(), total_nodes)[node_id]
                    .size()
                : node_id == output.initial_proc.address_space());
  }

  void select_sharding_functor(Legion::Mapping::MapperContext,
    const Legion::Task &,
    const SelectShardingFunctorInput &,
    SelectShardingFunctorOutput & out) override {
    out.chosen_functor = block_shard;
  }
  void select_sharding_functor(Legion::Mapping::MapperContext,
    const Legion::Copy &,
    const SelectShardingFunctorInput &,
    SelectShardingFunctorOutput & out) override {
    out.chosen_functor = block_shard;
  }
  void select_sharding_functor(Legion::Mapping::MapperContext,
    const Legion::Partition &,
    const SelectShardingFunctorInput &,
    SelectShardingFunctorOutput & out) override {
    out.chosen_functor = block_shard;
  }
  void select_sharding_functor(Legion::Mapping::MapperContext,
    const Legion::Fill &,
    const SelectShardingFunctorInput &,
    SelectShardingFunctorOutput & out) override {
    out.chosen_functor = block_shard;
  }

  Legion::LayoutConstraintID default_policy_select_layout_constraints(
    Legion::Mapping::MapperContext ctx,
    Realm::Memory target_memory,
    const Legion::RegionRequirement & req,
    Legion::Mapping::DefaultMapper::MappingKind mapping_kind,
    bool /* constraint */,
    bool & force_new_instances) override {

    if((req.privilege == LEGION_REDUCE) && (mapping_kind != COPY_MAPPING)) {
      force_new_instances = true;
      std::pair<Legion::Memory::Kind, Legion::ReductionOpID> constraint_key(
        target_memory.kind(), req.redop);
      std::map<std::pair<Legion::Memory::Kind, Legion::ReductionOpID>,
        Legion::LayoutConstraintID>::const_iterator finder =
        reduction_constraint_cache.find(constraint_key);
      // No need to worry about field constraint checks here
      // since we don't actually have any field constraints
      if(finder != reduction_constraint_cache.end())
        return finder->second;
      Legion::LayoutConstraintSet constraints;
      default_policy_select_constraints(ctx, constraints, target_memory, req);

      Legion::LayoutConstraintID result =
        runtime->register_layout(ctx, constraints);
      reduction_constraint_cache[constraint_key] = result;
      return result;
    }

    force_new_instances = false; // delay WAR tasks to save memory
    return soa_constraint_id;
  }

  /// Use subregions rather than parent region.
  virtual Legion::LogicalRegion default_policy_select_instance_region(
    Legion::Mapping::MapperContext,
    Realm::Memory,
    const Legion::RegionRequirement & req,
    const Legion::LayoutConstraintSet &,
    bool /* force_new_instances */,
    bool) override {
    return req.region;
  }

#if 0
  /*!
   This function will create compacted PhysicalInstance.

   For example, it will return 1 instance for the unstructured 
   topology with compacted+shared+ghost partitions.
   This is currently unused feature that, potentially,
   will be used if we have different partitions for
   owned and not owned entries.

  */
  void create_compacted_instance(const Legion::Mapping::MapperContext ctx,
    const Legion::Task & task,
    Legion::Mapping::Mapper::MapTaskOutput & output,
    const Legion::LayoutConstraintSet & layout_constraints,
    const size_t & indx) {
    using namespace Legion;
    using namespace Legion::Mapping;

    // creating physical instance for the compacted storaged
    flog_assert(task.regions.size() > indx + 2,
      "ERROR:: wrong number of regions passed to the task wirth \
               the tag = compacted_storage");

    flog_assert((task.regions[indx].region.exists()),
      "ERROR:: pasing not existing REGION to the mapper");

    Legion::Mapping::PhysicalInstance result = get_instance(ctx,
      name(task),
      target_mem,
      layout_constraints,
      {task.regions[indx].region,
        task.regions[indx + 1].region,
        task.regions[indx + 2].region});
    for(int j = 0; j < 3; j++) {
      output.chosen_instances[indx + j].clear();
      output.chosen_instances[indx + j].push_back(result);
    } // for
  } // create_compacted_instance
#endif

  void create_instance(const Legion::Mapping::MapperContext ctx,
    const Legion::Task & task,
    Legion::Mapping::Mapper::MapTaskOutput & output,
    const Legion::LayoutConstraintSet & layout_constraints,
    const size_t & indx) {
    using namespace Legion;
    using namespace Legion::Mapping;

    const LogicalRegion r = task.regions[indx].region;
    if(!r.exists()) // for incomplete launch maps
      return;

    output.chosen_instances[indx].push_back(
      get_instance(ctx, name(task), target_mem, layout_constraints, {r}));
  } // create_instance

  /// Implement \c gpu and \c omp tags and reuse or create
  /// appropriate SoA instances.
  virtual void map_task(const Legion::Mapping::MapperContext ctx,
    const Legion::Task & task,
    const Legion::Mapping::Mapper::MapTaskInput & input,
    Legion::Mapping::Mapper::MapTaskOutput & output) override {

    using namespace Legion;
    using namespace Legion::Mapping;
    using namespace mapper;

    if(output.target_procs.empty()) { // replicated tasks are automatic
      output.chosen_variant =
        find_variant(ctx, task.task_id, processor_kind(task.tag));
      output.target_procs = numa_domain;
    }

    output.chosen_instances.resize(task.regions.size());

    if(task.regions.size() > 0) {
      std::vector<std::set<Legion::FieldID>> missing_fields(
        task.regions.size());
      runtime->filter_instances(ctx,
        task,
        output.chosen_variant,
        output.chosen_instances,
        missing_fields);

      for(size_t indx = 0; indx < task.regions.size(); indx++) {
        // Check to see if any of the valid instances satisfy this requirement
        std::vector<Legion::Mapping::PhysicalInstance> valid_instances;
        for(auto & vi : input.valid_instances[indx])
          if(vi.get_location() == target_mem)
            valid_instances.push_back(vi);

        std::set<FieldID> valid_missing_fields;
        runtime->filter_instances(ctx,
          task,
          indx,
          output.chosen_variant,
          valid_instances,
          valid_missing_fields);
        if(!runtime->acquire_and_filter_instances(ctx, valid_instances))
          flog_fatal(
            "FleCSI mapper failed to acquire valid instances in map_task");
        missing_fields[indx] = valid_missing_fields;
        output.chosen_instances[indx] = valid_instances;

        if(missing_fields[indx].empty()) {
#if 0 // this block is only used for compacted instances
          if(task.regions[indx].tag & mapper::exclusive_lr){
            for(int j = 1; j < 3; j++)
              output.chosen_instances[indx + j] = valid_instances; 
            indx = indx + 2;
          }
#endif
          continue;
        }
        // We could not find valid instances that totally satisfy the
        // requirement. We need to create instances

        if(task.regions[indx].privilege == REDUCE) {
          create_reduction_instance(
            ctx, task, output, indx, valid_missing_fields);
          continue;
        }

#if 0 // this block is only used for compacted instances
        if(task.regions[indx].tag & mapper::exclusive_lr) {
            std::vector<Legion::FieldID> all_fields;
            for(auto fid : task.regions[indx].privilege_fields) {
              all_fields.push_back(fid);
            } // for
            layout_constraints.add_constraint(
              Legion::FieldConstraint(all_fields, true));
            create_compacted_instance(
              ctx, task, output, layout_constraints, indx);
          indx = indx + 2;
          continue;
        }
#endif
        for(const auto & missing_field : missing_fields[indx])
          create_instance(ctx, task, output, constraints(missing_field), indx);
      } // end for

    } // end if

  } // map_task

  /// Assign processors, unconditionally implementing \c force_rank_match.
  virtual void slice_task(const Legion::Mapping::MapperContext,
    const Legion::Task & task,
    const Legion::Mapping::Mapper::SliceTaskInput & input,
    Legion::Mapping::Mapper::SliceTaskOutput & output) override {

    using namespace Legion;
    using namespace mapper;

    const Legion::Rect<1> r = input.domain;
    const auto me = r.lo[0];

#if 0 // this is not supported in FleCSI yet
      // when we launch subtasks
      // this tag is used to map nested tasks
      if(task.tag & subrank_launch) {
        // expect a 1-D index domain
        assert(input.domain.get_dim() == 1);
        // send the whole domain to our local processor
        output.slices.resize(1);
        output.slices[0].domain = input.domain;
        output.slices[0].proc = task.target_proc;
        return;
      }
#endif
    // Our sharding functor unconditionally implements rank-matching: for now,
    // we simply do nothing to disturb that situation.
    auto & procs = processors(processor_kind(task.tag));
    auto it = procs.begin();
    for(auto m : util::equal_map(r.volume(), procs.size()))
      if(!m.empty())
        output.slices.emplace_back(
          Domain(me + m.front(), me + m.back()), *it++, false, false);
  } // slice_task

  /// Reuse existing indirection instances and request reusable preimages.
  virtual void map_copy(const Legion::Mapping::MapperContext ctx,
    const Legion::Copy & copy,
    const Legion::Mapping::Mapper::MapCopyInput & input,
    Legion::Mapping::Mapper::MapCopyOutput & output) override {

    bool has_unrestricted = false;
    for(unsigned idx = 0; idx < copy.src_requirements.size(); idx++) {
      auto & output_src = output.src_instances[idx];
      auto & output_dst = output.dst_instances[idx];
      auto & copy_src_req = copy.src_requirements[idx];
      auto & copy_dst_req = copy.dst_requirements[idx];

      // Try to reuse existing instances
      output_src = input.src_instances[idx];
      if(!output_src.empty())
        runtime->acquire_and_filter_instances(ctx, output_src);

      // According to Legion documention: for the indirections and reductions we
      // need to create an actual physical instance
      if((copy_dst_req.privilege == LEGION_REDUCE) ||
         (idx < copy.src_indirect_requirements.size()) ||
         (idx < copy.dst_indirect_requirements.size())) {
        if(!copy_src_req.is_restricted())
          create_copy_instance<true /*is src*/>(
            ctx, copy, copy_src_req, output_src);
        // else: do nothing (if restricted we cannot create a new instance)
      }
      // Do a virtual mapping instead of creating new instances
      // We can use this optimization only for copies without indirections
      else
        output_src.push_back(
          Legion::Mapping::PhysicalInstance::get_virtual_instance());

      // Try to reuse existing instances
      output_dst = input.dst_instances[idx];
      if(!output_dst.empty())
        runtime->acquire_and_filter_instances(ctx, output_dst);
      if(!copy_dst_req.is_restricted())
        has_unrestricted = true;
    }
    // If the instances are unrestricted we can create copies of them
    if(has_unrestricted) {
      for(unsigned idx = 0; idx < copy.dst_requirements.size(); idx++) {
        auto & output_dst = output.dst_instances[idx];
        auto & copy_dst_req = copy.dst_requirements[idx];
        // Try to reuse existing instances
        output_dst = input.dst_instances[idx];
        if(!copy_dst_req.is_restricted())
          create_copy_instance<false /*is src*/>(
            ctx, copy, copy_dst_req, output_dst);
      }
    }

    using instances = std::vector<Legion::Mapping::PhysicalInstance>;
    const auto indirect = [&](
                            const std::vector<Legion::RegionRequirement> & req,
                            const std::vector<instances> & in,
                            instances & out,
                            auto src) {
      for(unsigned idx = 0; idx < req.size(); idx++) {
        auto & in1 = in[idx];
        auto & out1 = out[idx];
        // Try to reuse existing instances
        bool can_reuse_instance = false;
        if(!in1.empty()) {
          out1 = in1[0];
          can_reuse_instance = runtime->acquire_instance(ctx, out1);
        }
        // We could not find a valid existing instance --> create a new one
        if(!can_reuse_instance && !req[idx].is_restricted()) {
          std::vector<Legion::Mapping::PhysicalInstance> tmp;
          create_copy_instance<src>(ctx, copy, req[idx], tmp);
          assert(tmp.size() == 1);
          out1 = tmp.front();
        }
      }
    };

    // Gather copy
    indirect(copy.src_indirect_requirements,
      input.src_indirect_instances,
      output.src_indirect_instances,
      std::true_type());
    // Scatter copy (for generality; FleCSI does not use scatter operations):
    indirect(copy.dst_indirect_requirements,
      input.dst_indirect_instances,
      output.dst_indirect_instances,
      std::false_type());

    output.compute_preimages = true;
  } // map_copy

private:
  static std::string name(const Legion::Task & t) {
    std::ostringstream s;
    s << "task " << std::quoted(t.get_task_name());
    return s.str();
  }

  Legion::Memory::Kind maybe_fb(Legion::Memory::Kind k) const {
    using namespace Legion;
    return local_kind == Processor::TOC_PROC ? Memory::GPU_FB_MEM : k;
  }

  Legion::Memory closest(Legion::Memory::Kind k, Legion::Processor p) const {
    using namespace Legion;
    Machine::MemoryQuery q(machine);
    q.local_address_space().only_kind(k).best_affinity_to(p);
    return q.first();
  }
  Legion::Memory closest(Legion::Memory::Kind k) const {
    return closest(k, local_proc);
  }
  Legion::Memory closest() const {
    return closest(maybe_fb(Legion::Memory::SYSTEM_MEM));
  }

  static Legion::Processor::Kind processor_kind(Legion::MappingTagID t) {
    using namespace mapper;
    using P = Legion::Processor;
    switch(t & proc_mask) {
      case gpu:
        return P::TOC_PROC;
      case omp:
        return P::OMP_PROC;
      default:
        return P::LOC_PROC;
    }
  }
  const decltype(local_cpus) & processors(Legion::Processor::Kind k) const {
    using P = Legion::Processor;
    switch(k) {
      case P::TOC_PROC:
        return local_gpus;
      case P::OMP_PROC:
        return local_omps;
      default:
        return local_cpus;
    }
  }

  static Legion::LayoutConstraintSet constraints(Legion::FieldID f_id) {
    using namespace Legion;
    LayoutConstraintSet ret;
    ret.add_constraint(SpecializedConstraint());
    ret.add_constraint(soa_constraint);
    ret.add_constraint(
      FieldConstraint(std::vector<FieldID>{f_id}, false, false));
    return ret;
  }

  /*
   * create_copy_instance : similar to
   * DefaultMapper::default_create_copy_instance except that
   * it creates one physical instance per field
   */
  template<bool S>
  void create_copy_instance(Legion::Mapping::MapperContext ctx,
    const Legion::Copy & copy,
    const Legion::RegionRequirement & req,
    std::vector<Legion::Mapping::PhysicalInstance> & instances) {
    using namespace Legion;
    using namespace Legion::Mapping;

    // See if we have all the fields covered
    std::set<FieldID> missing_fields = req.privilege_fields;
    for(auto & phys_instance : instances) {
      phys_instance.remove_space_fields(missing_fields);
      if(missing_fields.empty())
        return;
    }
    // If we still have missing fields, we need to create new instances
    Memory target_memory = default_policy_select_target_memory(
      ctx, copy.parent_task->current_proc, req);

    for(const auto & missing_field : missing_fields)
      instances.emplace_back(get_instance(ctx,
        S ? "copy source" : "copy destination",
        target_memory,
        constraints(missing_field),
        {req.region}));
  } // create_copy_instance

  void create_reduction_instance(const Legion::Mapping::MapperContext ctx,
    const Legion::Task & task,
    Legion::Mapping::Mapper::MapTaskOutput & output,
    const size_t & idx,
    std::set<Legion::FieldID> & missing_fields) {

    Legion::Processor target_proc = output.target_procs[0];
    bool needs_field_constraint_check = false;

    const Legion::TaskLayoutConstraintSet & layout_constraints =
      runtime->find_task_layout_constraints(
        ctx, task.task_id, output.chosen_variant);

    size_t footprint;
    if(!default_create_custom_instances(ctx,
         target_proc,
         target_mem,
         task.regions[idx],
         idx,
         missing_fields,
         layout_constraints,
         needs_field_constraint_check,
         output.chosen_instances[idx],
         &footprint)) {
      default_report_failed_instance_creation(
        task, idx, target_proc, target_mem, footprint);
    }
  }

  Legion::VariantID find_variant(const Legion::Mapping::MapperContext ctx,
    Legion::TaskID task_id,
    Legion::Processor::Kind processor_kind) {
    return variant
      .try_emplace({task_id, processor_kind}, util::convert{[&] {
        std::vector<Legion::VariantID> variants;
        runtime->find_valid_variants(ctx, task_id, variants, processor_kind);
        return variants.at(0);
      }})
      .first->second;
  }

  Legion::Mapping::PhysicalInstance get_instance(
    const Legion::Mapping::MapperContext ctx,
    const std::string & op,
    const Legion::Memory & target_mem, // can be different for copies
    const Legion::LayoutConstraintSet & layout_constraints,
    const std::vector<Legion::LogicalRegion> & regions) const {
    Legion::Mapping::PhysicalInstance result;
    std::size_t instance_size = 0;
    bool created, res = runtime->find_or_create_physical_instance(ctx,
                    target_mem,
                    layout_constraints,
                    regions,
                    result,
                    created,
                    true /*acquire*/,
                    GC_NEVER_PRIORITY,
                    true,
                    &instance_size);
    if(!res)
      flog_fatal("FleCSI mapper failed to allocate instance of size "
                 << instance_size << " in memory " << target_mem << " for "
                 << op);
    return result;
  }

  std::map<std::pair<Legion::TaskID, Legion::Processor::Kind>,
    Legion::VariantID>
    variant;

  Legion::Memory target_mem;
  decltype(local_cpus) numa_domain;

  static inline Legion::ShardingID block_shard =
    [id = Legion::Runtime::generate_static_sharding_id()] {
      struct functor : Legion::ShardingFunctor {
      private:
        static auto map(const Legion::Domain & d, Color n) {
          const Legion::Rect<1> r = d;
          assert(!r.lo[0]);
          return util::equal_map(r.hi[0] + 1, n);
        }

        bool is_invertible() const override {
          return true;
        }
        Legion::ShardID shard(const Legion::DomainPoint & p,
          const Legion::Domain & d,
          std::size_t ns) override {
          return map(d, ns).bin(p.point_data[0]);
        }
        void invert(Legion::ShardID s,
          const Legion::Domain &,
          const Legion::Domain & d,
          std::size_t ns,
          std::vector<Legion::DomainPoint> & out) override {
          const auto r = map(d, ns)[s];
          out.assign(r.begin(), r.end());
        }
      };
      Legion::Runtime::preregister_sharding_functor(id, new functor());
      return id;
    }();

  // used consistently
  static inline const Legion::OrderingConstraint soa_constraint = {
    {Legion::DimensionKind::DIM_Y,
      Legion::DimensionKind::DIM_X,
      Legion::DimensionKind::DIM_F},
    true /*contiguous*/
  };
  // preregister the ordering contraint
  static inline const Legion::LayoutConstraintID soa_constraint_id = [] {
    Legion::LayoutConstraintRegistrar registrar;
    registrar.add_constraint(soa_constraint);
    return Legion::Runtime::preregister_layout(registrar);
  }();
};

/// Replace default mappers with \c mpi_mapper_t instances.  \ns::run.
inline void
mapper_registration(Legion::Machine machine,
  Legion::Runtime * rt,
  const std::set<Legion::Processor> & local_procs) {
  for(std::set<Legion::Processor>::const_iterator it = local_procs.begin();
    it != local_procs.end();
    it++) {
    rt->replace_default_mapper(new mpi_mapper_t(machine, rt, *it), *it);
  }
}

/// \}
} // namespace run
} // namespace flecsi

#endif
