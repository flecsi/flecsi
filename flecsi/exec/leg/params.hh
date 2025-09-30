// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_LEG_PARAMS_HH
#define FLECSI_EXEC_LEG_PARAMS_HH

#include "flecsi/config.hh"
#include "flecsi/data/field.hh"
#include "flecsi/data/privilege.hh"
#include "flecsi/data/topology.hh"
#include "flecsi/exec/leg/future.hh"
#include "flecsi/exec/leg/tracer.hh"
#include "flecsi/util/array_ref.hh"

#include <legion.h>

namespace flecsi {

namespace topo {
struct global_base;
} // namespace topo

namespace exec {
/// \addtogroup legion-execution
/// \{

namespace leg {
using Indices = std::vector<
  std::pair<std::vector<Legion::RegionRequirement>::size_type, field_id_t>>;
}

namespace detail {
template<class T = void>
struct pointer_key {
  pointer_key(T * p) : p(p) {}
  bool operator<(const pointer_key & k) const {
    return std::less<>()(p, k.p);
  }

private:
  T * p;
};
} // namespace detail

struct task_prologue_impl : prolog_base {
  using prolog_base::prolog_base;

  std::vector<Legion::RegionRequirement> && region_requirements() && {
    return std::move(region_reqs_);
  } // region_requirements

  auto && region_indices() && {
    return std::move(which);
  }

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

  template<class P>
  static const data::borrow * get_projection(
    const topo::borrow_category<P> & b) {
    return &b.get_projection();
  }
  template<class T>
  static const data::borrow * get_projection(const T &) {
    return nullptr;
  }

  template<class A>
  void field(field_id_t f,
    bool rsz,
    data::leg::storage & s,
    const data::borrow * b,
    Legion::PrivilegeMode m,
    A && a) {
    const auto [it, add] =
      topo_req.try_emplace({&s, b, m}, region_reqs_.size());
    if(add)
      std::forward<A>(a)();
    auto & r = region_reqs_[which.emplace_back(it->second, f).first];
    if(!r.privilege_fields.count(f))
      r.add_field(f);
    if(rsz)
      r.add_flags(LEGION_SUPPRESS_WARNINGS_FLAG);
  }

protected:
  template<typename D,
    Privileges P,
    class Topo,
    typename Topo::index_space Space>
  void visit(data::accessor<data::raw, D, P> &,
    const data::field_reference<D, data::raw, Topo, Space> & r) {
    const field_id_t f = r.fid();
    auto & t = r.topology();
    data::region & reg = t.template get_region<Space>();
    auto & p = t.template get_partition<Space>();
    const data::borrow * b = get_projection(t);

    add_copy<P>(r);

    const Legion::PrivilegeMode m = privilege_mode(P);
    const Legion::LogicalRegion lr = reg.logical_region;
    field(f, reg.check_resize(f), p, b, m, [&] {
      if constexpr(std::is_same_v<typename Topo::base, topo::global_base>)
        region_reqs_.emplace_back(lr, m, LEGION_EXCLUSIVE, lr);
      else {
        data::borrow::attach(region_reqs_.emplace_back(p.logical_partition,
                               data::borrow::projection(b),
                               m,
                               LEGION_EXCLUSIVE,
                               lr),
          b);
      }
    });
  } // visit

  template<class R, typename T, class Topo, typename Topo::index_space Space>
  void visit(data::reduction_accessor<R, T> &,
    const data::field_reference<T, data::dense, Topo, Space> & r) {
    auto & t = r.topology();
    const Legion::LogicalRegion lr =
      t.template get_region<Space>().logical_region;
    static_assert(std::is_same_v<typename Topo::base, topo::global_base>);

    field(r.fid(),
      false,
      t.template get_partition<Space>(),
      nullptr,
      LEGION_REDUCE,
      [&] {
        region_reqs_.emplace_back(lr,
          // Cast to Legion::ReductionOpID due to missing definition of REDOP_ID
          // in legion_redop.h
          Legion::ReductionOpID(fold::wrap<R, T>::REDOP_ID),
          LEGION_EXCLUSIVE,
          lr);
      });
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
  /*--------------------------------------------------------------------------*
   Epilog
   *--------------------------------------------------------------------------*/
  template<class A>
  void visit(data::detail::save_for_epilog &, A & a) {
    // store the field to enact resizing at the end of the trace
    if(exec::is_tracing())
      epilog_wrappers.push_back([a]() { return trace::save_dynamic_field(a); });
    else // launch the reduction to check if resizing is required
      epilog_wrappers.push_back(
        [a]() { return a.get_elements().reduce_rsz_required(); });
  }

private:
  std::vector<Legion::RegionRequirement> region_reqs_;
  std::map<std::tuple<detail::pointer_key<data::leg::storage>,
             detail::pointer_key<const data::borrow>,
             Legion::PrivilegeMode>,
    decltype(region_reqs_.size())>
    topo_req;
  leg::Indices which;
  std::vector<Legion::Future> futures_;
  std::vector<Legion::FutureMap> future_maps_;
};

template<processor>
using task_prologue = task_prologue_impl;

/*!
  The bind_accessors type is called to walk the user task arguments inside of
  an executing legion task to properly complete the users accessors, i.e., by
  pointing the accessor \em view instances to the appropriate legion-mapped
  buffers.

  This is the other half of the wire protocol implemented by \c task_prologue.
 */
template<processor Proc>
struct bind_accessors {

  bind_accessors(Legion::Runtime * legion_runtime,
    Legion::Context & legion_context,
    std::vector<Legion::PhysicalRegion> const & regions,
    const leg::Indices & which,
    std::vector<Legion::Future> const & futures)
    : legion_runtime_(legion_runtime), legion_context_(legion_context),
      regions_(regions), which(which), futures_(futures) {}
  ~bind_accessors() {
    flog_assert(region == which.size(), "not enough parameters");
  }

protected:
  void visit(processor_space_t<Proc> & s) {
    const Legion::Task & t =
      *legion_runtime_->get_current_task(legion_context_);
    s.bind(t.index_domain.get_volume(), t.index_point.point_data[0]);
  }

  // All accessors are handled in terms of their underlying raw accessors.

  template<typename D, Privileges P>
  void visit(data::accessor<data::raw, D, P> & accessor) {
    auto [reg, f] = next();
    // For incomplete launch maps:
    if(!reg.get_logical_region().exists())
      return;

    const Legion::UnsafeFieldAccessor<D,
      data::leg::region_dimensions,
      Legion::coord_t,
      Realm::AffineAccessor<D, data::leg::region_dimensions, Legion::coord_t>>
      ac(reg, f);
    bind(reg, accessor, ac);
  }

  template<class R, typename D>
  void visit(data::reduction_accessor<R, D> & reduce) {
    auto [reg, f] = next();
    const Legion::ReductionAccessor<exec::fold::wrap<R, D>,
      false,
      data::leg::region_dimensions,
      Legion::coord_t,
      Realm::AffineAccessor<D, data::leg::region_dimensions, Legion::coord_t>>
      ac(reg, f, exec::fold::wrap<R, D>::REDOP_ID);
    bind(reg, reduce, ac);
  }

private:
  std::pair<const Legion::PhysicalRegion &, field_id_t> next() {
    flog_assert(region < which.size(), "too many parameters");
    const auto & [w, f] = which[region++];
    return {regions_[w], f};
  }

  template<typename A, typename LA>
  void bind(const Legion::PhysicalRegion & reg, A & acc, const LA & aa) const {
    const auto dom = legion_runtime_->get_index_space_domain(
      legion_context_, reg.get_logical_region().get_index_space());
    const Legion::Rect<data::leg::region_dimensions> r(dom);

    if(!dom.empty())
      acc.bind(util::span(aa.ptr(Legion::Domain::DomainPointIterator(dom).p),
        r.hi[1] - r.lo[1] + 1));
  }

protected:
  /*--------------------------------------------------------------------------*
   Futures
   *--------------------------------------------------------------------------*/
  template<typename D>
  void visit(future<D> & f) {
    f.legion_future_ = futures_[future_id++];
  }

private:
  Legion::Runtime * legion_runtime_;
  Legion::Context & legion_context_;
  size_t region = 0;
  const std::vector<Legion::PhysicalRegion> & regions_;
  const leg::Indices & which;
  size_t future_id = 0;
  const std::vector<Legion::Future> & futures_;

}; // struct bind_accessors

/// \}
} // namespace exec
} // namespace flecsi

#endif
