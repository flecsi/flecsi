// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_FIELD_HH
#define FLECSI_DATA_FIELD_HH

#include "flecsi/data/topology_slot.hh"
#include "flecsi/run/backend.hh"
#include "flecsi/util/demangle.hh"
#include <flecsi/data/layout.hh>
#include <flecsi/data/privilege.hh>

namespace flecsi {
namespace topo {
template<class>
struct ragged; // defined in terms of field
}

namespace data {
/// \addtogroup data
/// \{

/// A data accessor.
/// Name via \c field::accessor.
/// Pass a \c field_reference to a task that accepts an accessor.
/// \tparam L data layout
/// \tparam T data type
/// \tparam Priv access privileges
template<layout L, typename T, Privileges Priv>
struct accessor;

/// A specialized accessor for changing the extent of dynamic layouts.
/// Name via \c field::mutator.
template<layout, class, Privileges>
struct mutator;

namespace detail {
template<class, layout>
struct field_base {};
template<class, layout, class Topo, typename Topo::index_space>
struct field_register;

// All field registration is ultimately defined in terms of raw fields.
template<class T, class Topo, typename Topo::index_space Space>
struct field_register<T, raw, Topo, Space> : field_info_t {
  explicit field_register(field_id_t i)
    : field_info_t{i, sizeof(T), util::type<T>()} {
    run::context::instance().add_field_info<Topo, Space>(this);
  }
  field_register() : field_register(fid_counter()) {}
  // Copying/moving is inadvisable because the context knows the address.
  field_register(const field_register &) = delete;
  field_register & operator=(const field_register &) = delete;
};

// Work around the fact that std::is_trivially_move_constructible_v checks
// for a trivial destructor in some implementations:
template<class T>
union move_check // movable only if trivially so
{
  T t;
  ~move_check();
};
template<class T>
inline constexpr bool is_trivially_move_constructible_v =
  std::is_move_constructible_v<move_check<T>>;
} // namespace detail

/// Identifies a field on a particular topology instance.
/// Declare a task parameter as an \c accessor to use the field.
/// \tparam T data type (merely for type safety)
/// \tparam L data layout (similarly)
/// \tparam Space topology-relative index space
template<class T, layout L, class Topo, typename Topo::index_space Space>
struct field_reference : convert_tag {
  using value_type = T;
  using Topology = Topo;
  using topology_t = typename Topo::core;
  static constexpr auto space = Space;

  field_reference(const field_info_t & info, topology_t & topology)
    : fid_(info.fid), topology_(&topology) {}

  field_id_t fid() const {
    return fid_;
  }
  topology_t & topology() const {
    return *topology_;
  } // topology_identifier

  // We can't forward-declare partition, so just deduce these:
  template<class S>
  static auto & get_region(S & topo) {
    return topo.template get_region<Space>();
  }
  template<class S>
  static auto & get_partition(S & topo) {
    return topo.template get_partition<Space>();
  }

  auto & get_region() const {
    return get_region(*topology_);
  }
  auto & get_partition() const {
    return get_partition(*topology_);
  }

  auto & get_ragged() const { // -> ragged_partition<...>::core
    return topology_->ragged.template get<Space>()[fid_];
  }

  template<layout L2, class T2 = T> // TODO: allow only safe casts
  auto cast() const {
    return field_reference<T2, L2, Topo, Space>({fid_, 0, {}}, *topology_);
  }

  /// \if core
  /// Use this reference and return it.
  /// \endif
  template<class F>
  const field_reference & use(F && f) const {
    std::forward<F>(f)(*this);
    return *this;
  }

private:
  field_id_t fid_;
  topology_t * topology_;
};

// This is the portion of field validity that can be checked and that
// doesn't exclude things like std::tuple<int>.  It's similar to (the
// proposed) trivial relocatability, but excludes std::vector.
template<class T>
inline constexpr bool portable_v =
  std::is_object_v<T> && !std::is_pointer_v<T> &&
  detail::is_trivially_move_constructible_v<T>;
/// \}
} // namespace data

/// \addtogroup data
/// \{

/// Helper type to define and access fields.
/// \tparam T field value type: a trivially copyable type with no pointers
///   or references
/// \tparam L data layout
template<class T, data::layout L = data::dense>
struct field : data::detail::field_base<T, L> {
  static_assert(data::portable_v<T>);
  using value_type = T;

  template<Privileges Priv>
  using accessor1 = data::accessor<L, T, Priv>;
  template<Privileges Priv>
  using mutator1 = data::mutator<L, T, Priv>;
  /// The accessor to use as a parameter to receive this sort of field.
  /// \tparam PP the appropriate number of privilege values, interpreted as
  ///   - exclusive
  ///   - shared, ghost
  ///   - exclusive, shared, ghost
  template<partition_privilege_t... PP>
  using accessor = accessor1<privilege_pack<PP...>>;
  /// The mutator to use as a parameter for this sort of field (usable only
  /// for certain layouts).
  /// \tparam PP as for \c accessor
  template<partition_privilege_t... PP>
  using mutator = mutator1<privilege_pack<PP...>>;

  template<class Topo, typename Topo::index_space S>
  using Register = data::detail::field_register<T, L, Topo, S>;
  template<class Topo, typename Topo::index_space S>
  using Reference = data::field_reference<T, L, Topo, S>;

  /// A field registration.
  ///
  /// \warning
  /// Field definitions are typically declared \c const. If placed in a header
  /// make sure to declare them as <tt>inline const</tt> to avoid breaking ODR.
  ///
  /// \tparam Topo (specialized) topology type
  /// \tparam Space index space
  template<class Topo, typename Topo::index_space Space = Topo::default_space()>
  struct definition : Register<Topo, Space> {
    using Field = field;

    /// Return a reference to a field instance.
    /// \param t topology instance (must be allocated)
    auto operator()(data::topology_slot<Topo> & t) const {
      return (*this)(t.get());
    }
    Reference<Topo, Space> operator()(typename Topo::core & t) const {
      return {*this, t};
    }
  };

  /// Fields cannot be constructed.  Use \c definition instead.
  field() = delete;
};
/// \}

namespace data {
/// \addtogroup data
/// \{
namespace detail {
template<class T>
struct field_base<T, dense> {
  using base_type = field<T, raw>;
};
template<class T>
struct field_base<T, single> {
  using base_type = field<T>;
};
template<class T>
struct field_base<T, ragged> {
  using base_type = field<T, raw>;
  using Offsets = field<std::size_t>;
};
template<class T>
struct field_base<T, sparse> {
  using key_type = std::size_t;
  using base_type = field<std::pair<key_type, T>, ragged>;
};

// Many compilers incorrectly require the 'template' for a base class.
template<class T, layout L, class Topo, typename Topo::index_space Space>
struct field_register : field<T, L>::base_type::template Register<Topo, Space> {
  using Base = typename field<T, L>::base_type::template Register<Topo, Space>;
  using Base::Base;
};
template<class T, class Topo, typename Topo::index_space Space>
struct field_register<T, ragged, Topo, Space>
  : field<T, ragged>::base_type::template Register<topo::ragged<Topo>, Space> {
  using Offsets = typename field<T, ragged>::Offsets;
  // We use the same field ID for the offsets:
  typename Offsets::template Register<Topo, Space> off{field_register::fid};
};
} // namespace detail

template<class F, Privileges Priv>
using field_accessor = // for convenience with decltype
  typename std::remove_reference_t<F>::Field::template accessor1<Priv>;
// Accessors that are always used with the same (internal) field can be
// automatically initialized with its ID rather than having to be serialized.
template<const auto & F, Privileges Priv>
struct accessor_member : field_accessor<decltype(F), Priv> {
  accessor_member() : accessor_member::accessor(F.fid) {}
  using accessor_member::accessor::operator=; // for single

  template<class G>
  void topology_send(G && g) {
    // Using get_base() works around a GCC 9 bug that claims that the
    // inheritance of various accessor types is ambiguous.
    std::forward<G>(g)(get_base(), F);
  }
  template<class G, class S>
  void topology_send(G && g, S && s) { // s: topology -> subtopology
    std::forward<G>(g)(get_base(),
      [&s](auto & t) { return F(std::invoke(std::forward<S>(s), t.get())); });
  }

  typename accessor_member::accessor & get_base() {
    return *this;
  }
};

/// \}
} // namespace data
} // namespace flecsi

#endif
