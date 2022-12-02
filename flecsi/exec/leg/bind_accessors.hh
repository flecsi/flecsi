// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_LEG_BIND_ACCESSORS_HH
#define FLECSI_EXEC_LEG_BIND_ACCESSORS_HH

#include <flecsi-config.h>

#include "flecsi/data/field.hh"
#include "flecsi/data/topology.hh"
#include "flecsi/exec/leg/future.hh"
#include "flecsi/util/array_ref.hh"
#include "flecsi/util/demangle.hh"

#if !defined(FLECSI_ENABLE_LEGION)
#error FLECSI_ENABLE_LEGION not defined! This file depends on Legion!
#endif

#include <legion.h>

#include <memory>

namespace flecsi {
/// \addtogroup legion-execution
/// \{

inline flog::devel_tag bind_accessors_tag("bind_accessors");

namespace exec::leg {

/*!
  The bind_accessors type is called to walk the user task arguments inside of
  an executing legion task to properly complete the users accessors, i.e., by
  pointing the accessor \em view instances to the appropriate legion-mapped
  buffers.

  This is the other half of the wire protocol implemented by \c task_prologue.
 */
template<task_processor_type_t ProcessorType>
struct bind_accessors {

  bind_accessors(Legion::Runtime * legion_runtime,
    Legion::Context & legion_context,
    std::vector<Legion::PhysicalRegion> const & regions,
    std::vector<Legion::Future> const & futures)
    : legion_runtime_(legion_runtime), legion_context_(legion_context),
      regions_(regions), futures_(futures) {}

  template<class A>
  void operator()(A & a) {
    std::apply([&](auto &... aa) { (visit(aa), ...); }, a);
  }

private:
  auto visitor() {
    return
      [&](auto & p, auto &&) { visit(p); }; // Clang 8.0.1 deems 'this' unused
  }

  // All accessors are handled in terms of their underlying raw accessors.

  template<typename D, Privileges P>
  void visit(data::accessor<data::raw, D, P> & accessor) {
    auto & reg = regions_[region++];

    const Legion::UnsafeFieldAccessor<D,
      data::leg::region_dimensions,
      Legion::coord_t,
      Realm::AffineAccessor<D, data::leg::region_dimensions, Legion::coord_t>>
      ac(reg, accessor.field());
    bind(reg, accessor, ac);
  }

  template<class P>
  std::enable_if_t<std::is_base_of_v<data::send_tag, P>> visit(P & p) {
    p.send(visitor());
  }

  template<class R, typename D>
  void visit(data::reduction_accessor<R, D> & reduce) {
    auto & reg = regions_[region++];
    const Legion::ReductionAccessor<exec::fold::wrap<R, D>,
      false,
      data::leg::region_dimensions,
      Legion::coord_t,
      Realm::AffineAccessor<D, data::leg::region_dimensions, Legion::coord_t>>
      ac(reg, reduce.field(), exec::fold::wrap<R, D>::REDOP_ID);
    bind(reg, reduce, ac);
  }

  template<typename A, typename LA>
  void bind(const Legion::PhysicalRegion & reg, A & acc, const LA & aa) const {
    const auto dom = legion_runtime_->get_index_space_domain(
      legion_context_, reg.get_logical_region().get_index_space());
    const auto r = dom.get_rect<data::leg::region_dimensions>();

    if(!dom.empty())
      acc.bind(util::span(aa.ptr(Legion::Domain::DomainPointIterator(dom).p),
        r.hi[1] - r.lo[1] + 1));
  }

  /*--------------------------------------------------------------------------*
   Futures
   *--------------------------------------------------------------------------*/
  template<typename D>
  void visit(future<D> & f) {
    f = {futures_[future_id++]};
  }

  // Note: due to how visitor() is implemented above the first
  // parameter can not be 'const &' here, otherwise template/overload
  // resolution fails (silently).
  template<typename T>
  static void visit(data::detail::scalar_value<T> & s) {
    if constexpr(ProcessorType == exec::task_processor_type_t::toc) {
#if defined(__NVCC__) || defined(__CUDACC__)
      cudaMemcpy(s.host, s.device, sizeof(T), cudaMemcpyDeviceToHost);
      return;
#elif defined(__HIPCC__)
      auto status =
        hipMemcpy(s.host, s.device, sizeof(T), hipMemcpyDeviceToHost);
      flog_assert(hipSuccess == status, "Error calling hipMemcpy");
      return;
#else
      flog_assert(false, "Cuda should be enabled when using toc task");
#endif
    }
    *s.host = *s.device;
  }

  /*--------------------------------------------------------------------------*
    Non-FleCSI Data Types
   *--------------------------------------------------------------------------*/

  template<typename D>
  static typename std::enable_if_t<!std::is_base_of_v<data::bind_tag, D>> visit(
    D &) {
    {
      flog::devel_guard guard(bind_accessors_tag);
      flog_devel(info) << "No setup for parameter of type " << util::type<D>()
                       << std::endl;
    }
  } // visit

  Legion::Runtime * legion_runtime_;
  Legion::Context & legion_context_;
  size_t region = 0;
  const std::vector<Legion::PhysicalRegion> & regions_;
  size_t future_id = 0;
  const std::vector<Legion::Future> & futures_;

}; // struct bind_accessors

/// \}
} // namespace exec::leg
} // namespace flecsi

#endif
