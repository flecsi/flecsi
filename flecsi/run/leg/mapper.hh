// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUN_LEG_MAPPER_HH
#define FLECSI_RUN_LEG_MAPPER_HH

#include "../backend.hh"
#include "flecsi/config.hh"

#include <legion.h>
#include <legion/legion_mapping.h>
#include <mappers/default_mapper.h>

#include <iomanip>

namespace flecsi {

inline flog::devel_tag legion_mapper_tag("legion_mapper");

namespace run {
/// \addtogroup legion-runtime
/// \{

/*!
 The mpi_mapper_t - is a custom mapper that handles mpi-legion
 interoperability in FleCSI
*/

class mpi_mapper_t : public Legion::Mapping::DefaultMapper
{
public:
  /*!
   Contructor. Derives from the Legion's Default Mapper

   @param machine Machine type for Legion's Realm
   @param _runtime Legion runtime
   @param local processor type: currently supports only
           LOC_PROC and TOC_PROC

   This constructor is different from a constructor in Default Mapper
   because it sets up some information on which memory to use
   depending on the Processor type (local_sysmemory,
   local_zerobuffer, etc)
   */

  mpi_mapper_t(Legion::Machine machine,
    Legion::Runtime * _runtime,
    Legion::Processor local)
    : Legion::Mapping::DefaultMapper(_runtime->get_mapper_runtime(),
        machine,
        local,
        "default"),
      machine(machine) {
    using namespace Legion;
    using namespace Legion::Mapping;
    // Auto-enable 'memoize' (set to 'false' by default in the DefaultMapper)
    // there is no more need to specify '-dm:memoize' at runtime
    memoize = true;
    // Get our local memories
    {
      Machine::MemoryQuery sysmem_query(machine);
      sysmem_query.local_address_space();
      sysmem_query.only_kind(Memory::SYSTEM_MEM);
      local_sysmem = sysmem_query.first();
      assert(local_sysmem.exists());
    }
    if(!local_gpus.empty()) {
      Machine::MemoryQuery zc_query(machine);
      zc_query.local_address_space();
      zc_query.only_kind(Memory::Z_COPY_MEM);
      local_zerocopy = zc_query.first();
      assert(local_zerocopy.exists());
    }
    else {
      local_zerocopy = Memory::NO_MEMORY;
    }
    if(local_kind == Processor::TOC_PROC) {
      Machine::MemoryQuery fb_query(machine);
      fb_query.local_address_space();
      fb_query.only_kind(Memory::GPU_FB_MEM);
      fb_query.best_affinity_to(local_proc);
      local_framebuffer = fb_query.first();
      assert(local_framebuffer.exists());
    }
    else {
      local_framebuffer = Memory::NO_MEMORY;
    }
  } // end mpi_mapper_t

  /*!
    Destructor
   */
  virtual ~mpi_mapper_t() {}

  void select_task_options(const Legion::Mapping::MapperContext ctx,
    const Legion::Task & task,
    Legion::Mapping::Mapper::TaskOptions & output) override {
    DefaultMapper::select_task_options(ctx, task, output);
    // make sure the input provided to `map_task` includes all the valid
    // instances that the runtime knows
    output.valid_instances = true;
  }

  /* This is the method to choose default Layout constraints.
     FleCSI is currently uses SOA ordering, which is different from
     the one in Default Mapper
  */
  Legion::LayoutConstraintID default_policy_select_layout_constraints(
    Legion::Mapping::MapperContext ctx,
    Realm::Memory target_memory,
    const Legion::RegionRequirement & req,
    Legion::Mapping::DefaultMapper::MappingKind mapping_kind,
    bool /* constraint */,
    bool & force_new_instances) override {

    if((req.privilege == LEGION_REDUCE) && (mapping_kind != COPY_MAPPING)) {
      // Always make new reduction instances
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
      // Save the result
      reduction_constraint_cache[constraint_key] = result;
      return result;
    }

    // We always set force_new_instances to false since we are
    // deciding to optimize for minimizing memory usage instead
    // of avoiding Write-After-Read (WAR) dependences
    force_new_instances = false;
    std::vector<Legion::DimensionKind> ordering;
    ordering.push_back(Legion::DimensionKind::DIM_Y);
    ordering.push_back(Legion::DimensionKind::DIM_X);
    ordering.push_back(Legion::DimensionKind::DIM_F); // SOA
    Legion::OrderingConstraint ordering_constraint(
      ordering, true /*contiguous*/);
    Legion::LayoutConstraintSet layout_constraint;
    layout_constraint.add_constraint(ordering_constraint);

    // Do the registration
    Legion::LayoutConstraintID result =
      runtime->register_layout(ctx, layout_constraint);
    return result;
  }

  /*!
   Specialization of the default_policy_select_instance_region method for
   FleCSI. In case of FleCSI we want exact region that has been requested to be
   created. This is different from Default mapper which will map Parent region,
   if it exists.

   @param req Region requirement for which instance is going to be allocated
  */
  virtual Legion::LogicalRegion default_policy_select_instance_region(
    Legion::Mapping::MapperContext,
    Realm::Memory,
    const Legion::RegionRequirement & req,
    const Legion::LayoutConstraintSet &,
    bool /* force_new_instances */,
    bool) override {
    return req.region;
  } // default_policy_select_instance_region

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
    const Legion::Memory & target_mem,
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

    // compacting region requirements for exclusive, shared and ghost into one
    // instance
    Legion::Mapping::PhysicalInstance result = get_instance(ctx,
      task,
      target_mem,
      layout_constraints,
      indx,
      {task.regions[indx].region,
        task.regions[indx + 1].region,
        task.regions[indx + 2].region});
    for(size_t j = 0; j < 3; j++) {
      output.chosen_instances[indx + j].clear();
      output.chosen_instances[indx + j].push_back(result);
    } // for
  } // create_compacted_instance
#endif

  void create_instance(const Legion::Mapping::MapperContext ctx,
    const Legion::Task & task,
    Legion::Mapping::Mapper::MapTaskOutput & output,
    const Legion::Memory & target_mem,
    const Legion::LayoutConstraintSet & layout_constraints,
    const size_t & indx) {
    using namespace Legion;
    using namespace Legion::Mapping;

    const LogicalRegion r = task.regions[indx].region;
    if(!r.exists()) // for incomplete launch maps
      return;

    output.chosen_instances[indx].push_back(
      get_instance(ctx, task, target_mem, layout_constraints, indx, {r}));
  } // create_instance

  /*!
   Specialization of the map_task funtion for FleCSI.

   The function has some FleCSI-specific features:

   1) It specifies SOA ordering for new physical instances;

   2) It stores information about already created instances
      and avoids creating a new instance if possible;

   3) It has logic on how to create compacted instances;

    @param ctx Mapper Context
    @param task Legion's task
    @param output Output information about task mapping
   */

  virtual void map_task(const Legion::Mapping::MapperContext ctx,
    const Legion::Task & task,
    const Legion::Mapping::Mapper::MapTaskInput & input,
    Legion::Mapping::Mapper::MapTaskOutput & output) override {

    using namespace Legion;
    using namespace Legion::Mapping;
    using namespace mapper;

    if(task.tag == prefer_gpu && !local_gpus.empty()) {
      output.chosen_variant = find_variant(
        ctx, task.task_id, gpu_variants, Legion::Processor::TOC_PROC);
      output.target_procs.push_back(task.target_proc);
    }
    else if(task.tag == prefer_omp && !local_omps.empty()) {
      output.chosen_variant = find_variant(
        ctx, task.task_id, omp_variants, Legion::Processor::OMP_PROC);
      output.target_procs = local_omps;
    }
    else {
      output.chosen_variant = find_variant(
        ctx, task.task_id, cpu_variants, Legion::Processor::LOC_PROC);
      output.target_procs.resize(1, local_proc);
    }

    output.chosen_instances.resize(task.regions.size());

    if(task.regions.size() > 0) {

      Legion::Memory target_mem;
      //   =
      //     DefaultMapper::default_policy_select_target_memory(
      //       ctx, task.target_proc, task.regions[0]);

      if(task.tag == prefer_gpu && !local_gpus.empty())
        target_mem = local_framebuffer;
      else
        target_mem = local_sysmem;

      // creating ordering constraint (SOA )
      std::vector<Legion::DimensionKind> ordering;
      ordering.push_back(Legion::DimensionKind::DIM_Y);
      ordering.push_back(Legion::DimensionKind::DIM_X);
      ordering.push_back(Legion::DimensionKind::DIM_F); // SOA
      Legion::OrderingConstraint ordering_constraint(
        ordering, true /*contiguous*/);

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
          if(task.regions[indx].tag == mapper::exclusive_lr){
            for(size_t j = 1; j < 3; j++)
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
            ctx, task, output, target_mem, indx, valid_missing_fields);
          continue;
        }
        // Filling out the "layout_constraints"
        Legion::LayoutConstraintSet layout_constraints;
        // No specialization
        layout_constraints.add_constraint(Legion::SpecializedConstraint());
        layout_constraints.add_constraint(ordering_constraint);
        // Constrained for the target memory kind
        layout_constraints.add_constraint(
          Legion::MemoryConstraint(target_mem.kind()));

#if 0 // this block is only used for compacted instances
        if(task.regions[indx].tag == mapper::exclusive_lr) {
            std::vector<Legion::FieldID> all_fields;
            for(auto fid : task.regions[indx].privilege_fields) {
              all_fields.push_back(fid);
            } // for
            layout_constraints.add_constraint(
              Legion::FieldConstraint(all_fields, true));
            create_compacted_instance(
              ctx, task, output, target_mem, layout_constraints, indx);
          indx = indx + 2;
          continue;
        }
#endif
        // We need to create a new instance containing the missing fields
        layout_constraints.add_constraint(
          Legion::FieldConstraint(missing_fields[indx], true));
        create_instance(
          ctx, task, output, target_mem, layout_constraints, indx);
      } // end for

    } // end if

  } // map_task

  /* This is a FleCSI specialization for the slice_task method
    that specify how resources are choosen for the task.
    In case of the Index task, it will specify what processes
    should be used by each index point.
    In particular, it provides FleCSI specific logic for how to map MPI tasks
  */

  virtual void slice_task(const Legion::Mapping::MapperContext,
    const Legion::Task & task,
    const Legion::Mapping::Mapper::SliceTaskInput & input,
    Legion::Mapping::Mapper::SliceTaskOutput & output) override {

    using namespace Legion;
    using namespace mapper;

    switch(task.tag) {
#if 0 // this is not supported in FleCSI yet
      // when we launch subtasks
      // this tag is used to map nested tasks
      case subrank_launch:
        // expect a 1-D index domain
        assert(input.domain.get_dim() == 1);
        // send the whole domain to our local processor
        output.slices.resize(1);
        output.slices[0].domain = input.domain;
        output.slices[0].proc = task.target_proc;
        break;
#endif
      case force_rank_match: /* MPI tasks or tasks that need 1-to-1 matching
                                with MPI ranks*/
      {
        // Control replication has already subdivided the launch domain:
        assert(input.domain.get_dim() == 1);
        const Legion::Rect<1> r = input.domain;
        const auto me = r.lo[0];
        assert(r.hi[0] == me);

        output.slices.clear();
        // Find the CPU with the desired address space:
        Legion::Machine::ProcessorQuery pq =
          Legion::Machine::ProcessorQuery(machine).only_kind(
            Legion::Processor::LOC_PROC);
        for(Legion::Machine::ProcessorQuery::iterator it = pq.begin();
            it != pq.end();
            ++it) {
          Legion::Processor p = *it;
          if(p.address_space() == me) {
            auto & out = output.slices.emplace_back();
            out.domain = r;
            out.proc = p;
            break;
          }
        }
        assert(!output.slices.empty());
        break;
      }

      // general leaf tasks
      default:
        // We've already been control replicated, so just divide our points
        // over the local processors, depending on which kind we prefer
        if(task.tag == prefer_gpu && !local_gpus.empty())
          distribute_index_points_accross_local_procs(
            input, output, local_gpus);
        else if(task.tag == prefer_omp && !local_omps.empty())
          distribute_index_points_accross_local_procs(
            input, output, local_omps);
        else
          distribute_index_points_accross_local_procs(
            input, output, local_cpus);
    }

  } // slice_task

  virtual void map_copy(const Legion::Mapping::MapperContext ctx,
    const Legion::Copy & copy,
    const Legion::Mapping::Mapper::MapCopyInput & input,
    Legion::Mapping::Mapper::MapCopyOutput & output) override {
    DefaultMapper::map_copy(ctx, copy, input, output);

    // currently our copy_plans are reused which is why we
    // want the gather copies to be optimized for repeated use.
    output.compute_preimages = true;
  } // map_copy

private:
  /*
    Distribute the index points of a domain across the processors provided in
    `local_procs` in a round robin way
  */
  static void distribute_index_points_accross_local_procs(
    const Legion::Mapping::Mapper::SliceTaskInput & input,
    Legion::Mapping::Mapper::SliceTaskOutput & output,
    const std::vector<Legion::Processor> & local_procs) {
    using namespace Legion;
    using namespace mapper;
    unsigned local_index = 0;
    for(Domain::DomainPointIterator itr(input.domain); itr; itr++) {
      TaskSlice slice;
      slice.domain = Domain(itr.p, itr.p);
      slice.proc = local_procs[local_index];
      local_index = (local_index + 1) % local_procs.size();
      slice.recurse = false;
      slice.stealable = false;
      output.slices.push_back(slice);
    }
  }

  /*!
   This function will create PhysicalInstance for Reduction task
  */
  void create_reduction_instance(const Legion::Mapping::MapperContext ctx,
    const Legion::Task & task,
    Legion::Mapping::Mapper::MapTaskOutput & output,
    const Legion::Memory & target_mem,
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
  } // create reduction instance

  /*!
   This function will find a variant from a VariantID map for the task
  */
  Legion::VariantID find_variant(const Legion::Mapping::MapperContext ctx,
    Legion::TaskID task_id,
    std::map<Legion::TaskID, Legion::VariantID> & variant,
    Legion::Processor::Kind processor_kind) {

    std::map<Legion::TaskID, Legion::VariantID>::const_iterator finder =
      variant.find(task_id);
    if(finder != variant.end())
      return finder->second;
    std::vector<Legion::VariantID> variants;
    runtime->find_valid_variants(ctx, task_id, variants, processor_kind);
    return variant[task_id] = variants.at(0);
  }

  static void report_size(const Legion::Task & task,
    std::size_t indx,
    std::size_t instance_size) {
    flog_devel(info) << "task " << task.get_task_name()
                     << " allocates physical instance with size "
                     << instance_size << " for the region requirement #" << indx
                     << std::endl;

    if(instance_size > 1000000000) {
      flog_devel(error)
        << "task " << task.get_task_name()
        << " is trying to allocate physical instance with the size > than 1 Gb("
        << instance_size << " )"
        << " for the region requirement # " << indx << std::endl;
    } // if
  }
  Legion::Mapping::PhysicalInstance get_instance(
    const Legion::Mapping::MapperContext ctx,
    const Legion::Task & task,
    const Legion::Memory & target_mem,
    const Legion::LayoutConstraintSet & layout_constraints,
    std::size_t indx,
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
                 << instance_size << " in memory " << target_mem << " for task "
                 << std::quoted(task.get_task_name()));
    report_size(task, indx, instance_size);
    return result;
  }

  Realm::Machine machine;

protected:
  std::map<Legion::TaskID, Legion::VariantID> cpu_variants;
  std::map<Legion::TaskID, Legion::VariantID> gpu_variants;
  std::map<Legion::TaskID, Legion::VariantID> omp_variants;

  Legion::Memory local_sysmem, local_zerocopy, local_framebuffer;
};

/*!
 mapper_registration is used to replace DefaultMapper with mpi_mapper_t in
 FleCSI
 */

inline void
mapper_registration(Legion::Machine machine,
  Legion::HighLevelRuntime * rt,
  const std::set<Legion::Processor> & local_procs) {
  for(std::set<Legion::Processor>::const_iterator it = local_procs.begin();
      it != local_procs.end();
      it++) {
    rt->replace_default_mapper(new mpi_mapper_t(machine, rt, *it), *it);
  }
} // mapper registration

/// \}
} // namespace run
} // namespace flecsi

#endif
