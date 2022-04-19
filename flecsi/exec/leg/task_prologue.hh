// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_LEG_TASK_PROLOGUE_HH
#define FLECSI_EXEC_LEG_TASK_PROLOGUE_HH

#include <flecsi-config.h>

#include "flecsi/data/field.hh"
#include "flecsi/data/privilege.hh"
#include "flecsi/data/topology.hh"
#include "flecsi/exec/leg/future.hh"

#if !defined(FLECSI_ENABLE_LEGION)
#error FLECSI_ENABLE_LEGION not defined! This file depends on Legion!
#endif

#include <legion.h>

namespace flecsi {

namespace topo {
struct global_base;
} // namespace topo

namespace exec {

struct task_prologue_impl {
  std::vector<Legion::RegionRequirement> const & region_requirements() const {
    return region_reqs_;
  } // region_requirements

  std::vector<Legion::Future> && futures() && {
    return std::move(futures_);
  } // futures

  std::vector<Legion::FutureMap> const & future_maps() const {
    return future_maps_;
  } // future_maps

private:
  static Legion::PrivilegeMode privilege_mode(Privileges mode) {
    // Reduce the read and write permissions for each privilege separately:
    bool r = false, w = false;
    for(auto i = privilege_count(mode); i-- && !(r && w);) {
      const auto p = get_privilege(i, mode);
      r = r || privilege_read(p);
      w = w || privilege_write(p);
    }
    return r   ? w ? LEGION_READ_WRITE : LEGION_READ_ONLY
           : w ? privilege_discard(mode) ? LEGION_WRITE_DISCARD
                                         : LEGION_READ_WRITE
               : LEGION_NO_ACCESS;
  } // privilege_mode

protected:
  // This implementation can be generic because all topologies are expected to
  // provide get_region (and, with one exception, get_partition).
  template<typename D,
    Privileges P,
    class Topo,
    typename Topo::index_space Space>
  void visit(data::accessor<data::raw, D, P> &,
    const data::field_reference<D, data::raw, Topo, Space> & r) {
    const field_id_t f = r.fid();
    auto & t = r.topology();
    data::region & reg = t.template get_region<Space>();

    reg.ghost_copy<P>(r);

    const Legion::PrivilegeMode m = privilege_mode(P);
    const Legion::LogicalRegion lr = reg.logical_region;
    if constexpr(std::is_same_v<typename Topo::base, topo::global_base>)
      region_reqs_.emplace_back(lr, m, LEGION_EXCLUSIVE, lr);
    else
      region_reqs_.emplace_back(
        t.template get_partition<Space>().logical_partition,
        data::leg::def_proj,
        m,
        LEGION_EXCLUSIVE,
        lr);
    region_reqs_.back().add_field(f);
  } // visit

  template<class R, typename T, class Topo, typename Topo::index_space Space>
  void visit(data::reduction_accessor<R, T> &,
    const data::field_reference<T, data::dense, Topo, Space> & r) {
    const Legion::LogicalRegion lr =
      r.topology().template get_region<Space>().logical_region;
    static_assert(std::is_same_v<typename Topo::base, topo::global_base>);
    region_reqs_
      .emplace_back(lr,
        // Cast to Legion::ReductionOpID due to missing definition of REDOP_ID
        // in legion_redop.h
        Legion::ReductionOpID(fold::wrap<R, T>::REDOP_ID),
        EXCLUSIVE,
        lr)
      .add_field(r.fid());
  } // visit

  /*--------------------------------------------------------------------------*
    Futures
   *--------------------------------------------------------------------------*/
  template<class P, class T>
  void visit(P &, const future<T> & f) {
    futures_.push_back(f.legion_future_);
  }

  template<class P, class T>
  void visit(P &, const future<T, exec::launch_type_t::index> & f) {
    future_maps_.push_back(f.legion_future_);
  }

private:
  std::vector<Legion::RegionRequirement> region_reqs_;
  std::vector<Legion::Future> futures_;
  std::vector<Legion::FutureMap> future_maps_;
};

template<task_processor_type_t>
using task_prologue = task_prologue_impl;
} // namespace exec
} // namespace flecsi

#endif
