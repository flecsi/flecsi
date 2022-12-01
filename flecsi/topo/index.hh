// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_INDEX_HH
#define FLECSI_TOPO_INDEX_HH

#include "flecsi/data/accessor.hh"
#include "flecsi/exec/launch.hh"
#include "flecsi/topo/size.hh"

namespace flecsi {
namespace topo {
/// \addtogroup topology
/// \{

namespace zero {
inline std::size_t
function(std::size_t) {
  return 0;
}
inline constexpr auto partial = make_partial<function>();
} // namespace zero

// A partition with a field for dynamically resizing it.
struct repartition : with_size, data::prefixes, with_cleanup {
  // Construct a partition with an initial size.
  // f is passed as a task argument, so it must be serializable;
  // consider using make_partial.
  template<class F = decltype((zero::partial))>
  repartition(data::region & r, F && f = zero::partial)
    : with_size(r.size().first), prefixes(r, sizes().use([&f](auto ref) {
        execute<fill<std::decay_t<F>>>(ref, std::forward<F>(f));
      })) {}
  void resize() { // apply sizes stored in the field
    update(sizes());
  }

  template<auto>
  repartition & get_partition() {
    return *this;
  }

private:
  template<class F>
  static void fill(resize::Field::accessor<wo> a, F f) {
    a = std::move(f)(run::context::instance().color());
  }
};

using repartitioned = data::partitioned<repartition>;

template<class T, typename T::index_space S = T::default_space(), class F>
repartitioned
make_repartitioned(Color r, F f) {
  return {data::make_region<T, S>({r, data::logical_size}), std::move(f)};
}

// Stores the flattened elements of the ragged fields on an index space.
struct ragged_partition_base : repartition {
  using coloring = data::region &;

  ragged_partition_base(coloring c) : repartition(c), reg(&c) {}

  template<single_space>
  data::region & get_region() const {
    return *reg;
  }

  template<single_space>
  const repartition & get_partition() const {
    return *this;
  }
  template<single_space>
  repartition & get_partition() {
    return *this;
  }

  // Ragged ghost copies must be handled at the level of the host topology.
  template<class R>
  void ghost_copy(const R &) {}

private:
  data::region * reg; // note the address stability assumption
};
template<class>
struct ragged_partition_category : ragged_partition_base {
  using ragged_partition_base::ragged_partition_base;
};
template<>
struct detail::base<ragged_partition_category> {
  using type = ragged_partition_base;
};
template<PrivilegeCount N>
struct ragged_partition
  : specialization<ragged_partition_category, ragged_partition<N>> {
  template<single_space>
  static constexpr PrivilegeCount privilege_count = N;
};

template<class>
struct ragged;

template<class Topo, typename Topo::index_space S>
struct ragged_partitioned : data::region {
  using base_type = ragged_partition<Topo::template privilege_count<S>>;
  using Partition = typename base_type::core;

  explicit ragged_partitioned(Color r)
    : region({r, data::logical_size}, util::key_type<S, ragged<Topo>>()) {
    for(const auto & fi : run::context::field_info_store<ragged<Topo>, S>())
      part.try_emplace(fi->fid, *this);
  }
  ragged_partitioned(ragged_partitioned &&) = delete; // we store 'this'
  Partition & operator[](field_id_t i) {
    return part.at(i);
  }
  const Partition & operator[](field_id_t i) const {
    return part.at(i);
  }

private:
  std::map<field_id_t, Partition> part;
};

namespace detail {
template<class, class>
struct ragged_tuple;
template<class T, typename T::index_space... SS>
struct ragged_tuple<T, util::constants<SS...>> {
  using type =
    util::key_tuple<util::key_type<SS, ragged_partitioned<T, SS>>...>;
};
} // namespace detail

struct ragged_base {
  using coloring = std::nullptr_t;
};
template<class>
struct ragged_category : ragged_base {
  ragged_category() = delete; // used only for registering fields
};

template<class P>
struct ragged_elements {
  using index_spaces = typename P::index_spaces;
  using index_space = typename P::index_space;

  explicit ragged_elements(Color c) : ragged_elements(c, index_spaces()) {}

  template<index_space S>
  ragged_partitioned<P, S> & get() {
    return part.template get<S>();
  }

private:
  template<auto... VV>
  ragged_elements(Color n,
    util::constants<VV...> /* index_spaces, to deduce a pack */
    )
    : part((void(VV), n)...) {}

  typename detail::ragged_tuple<P, index_spaces>::type part;
};
template<class T>
struct ragged : specialization<ragged_category, ragged<T>> {
  using index_space = typename T::index_space;
  using index_spaces = typename T::index_spaces;
};

// Standardized interface for use by fields and accessors:
template<class P>
struct with_ragged {
  with_ragged(Color n) : ragged(n) {}

  ragged_elements<P> ragged;
};

template<>
struct detail::base<ragged_category> {
  using type = ragged_base;
};

// The user-facing variant of the color category supports ragged fields.
struct index_base {
  using coloring = Color;
};

template<class P>
struct index_category : index_base, color<P>, with_ragged<P>, with_cleanup {
  using index_base::coloring; // override color_base::coloring
  explicit index_category(coloring c) : color<P>({c, 1}), with_ragged<P>(c) {}
};
template<>
struct detail::base<index_category> {
  using type = index_base;
};

// A subtopology for holding topology-specific metadata per color.
template<class P>
struct meta : specialization<index_category, meta<P>> {};

template<class P>
struct with_meta { // for interface consistency
  with_meta(Color n) : meta(n) {}
  typename topo::meta<P>::core meta;
};

// A subtopology for holding internal arrays without ragged support.
struct array_base {
  using coloring = std::vector<std::size_t>;

protected:
  static std::size_t index(const coloring & c, std::size_t i) {
    return c[i];
  }
};
template<class P>
struct array_category : array_base, repartitioned {
  explicit array_category(const coloring & c)
    : partitioned(make_repartitioned<P>(c.size(), make_partial<index>(c))) {}
};
template<>
struct detail::base<array_category> {
  using type = array_base;
};

template<class P>
struct array : topo::specialization<array_category, array<P>> {};

/// \defgroup spec Predefined specializations
/// Specializations for topologies so simple that no others are needed.
/// \{

/*!
  The \c index type allows users to register data on an
  arbitrarily-sized set of indices that have an implicit one-to-one coloring.
  Its \c coloring type is just the size of that set.
 */
struct index : specialization<index_category, index> {
  static coloring color(Color size) {
    return size;
  } // color

}; // struct index

/// \}
/// \}
} // namespace topo

} // namespace flecsi

#endif
