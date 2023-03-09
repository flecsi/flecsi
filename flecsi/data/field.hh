// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_FIELD_HH
#define FLECSI_DATA_FIELD_HH

#include "flecsi/data/topology_slot.hh"
#include "flecsi/run/backend.hh"
#include "flecsi/util/demangle.hh"
#include "flecsi/util/target.hh"
#include <flecsi/data/layout.hh>
#include <flecsi/data/privilege.hh>

namespace flecsi {
namespace topo {
struct with_cleanup; // defined in terms of cleanup

template<class>
struct ragged; // defined in terms of field
} // namespace topo

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

template<class R, typename T>
struct reduction_accessor;

/// A specialized accessor for changing the extent of dynamic layouts.
/// Name via \c field::mutator.
template<layout, class, Privileges>
struct mutator;

namespace launch {
template<class P>
struct mapping;
}

// Each field can have a destructor (for individual field values) registered
// that is invoked when the field is recreated or the region is destroyed.
struct cleanup {
  using function = std::function<void()>;

  void operator()(field_id_t f, function d) {
    fields.insert_or_assign(f, std::move(d));
  }

private:
  struct finalizer {
    finalizer(function f) noexcept : f(std::move(f)) {}
    finalizer(finalizer && o) noexcept {
      f.swap(o.f); // guarantee o.f is empty
    }
    ~finalizer() {
      if(f)
        f();
    }
    finalizer & operator=(finalizer o) noexcept {
      f.swap(o.f);
      return *this;
    }

  private:
    function f;
  };

  std::map<field_id_t, finalizer> fields;
};

namespace detail {
template<class T, auto S, class = void> // core topology type
struct cleanup {
  static data::cleanup & get(T & t) {
    return t.template get_partition<S>().cleanup;
  }
};
template<class T, auto S>
struct cleanup<T,
  S,
  std::enable_if_t<std::is_base_of_v<topo::with_cleanup, T>>> {
  static data::cleanup & get(T & t) {
    return t.cleanup;
  }
};
template<auto S, class T>
data::cleanup &
get_cleanup(T & t) {
  return cleanup<T, S>::get(t);
}

template<class, layout>
struct field_base {};
template<class, layout, class Topo, typename Topo::index_space>
struct field_register;

// All field registration is ultimately defined in terms of raw fields.
template<class T, class Topo, typename Topo::index_space Space>
struct field_register<T, raw, Topo, Space> {
  explicit field_register(field_id_t i) : fid(i) {
    run::context::add_field_info<Topo, Space, T>(i);
  }
  field_register() : field_register(fid_counter()) {}
  field_id_t fid;
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

struct particle_base {
  using size_type = std::size_t; // full-width since have only the one block
  struct link {
    // Only the first element of each run participates in the linked list.
    size_type prev, // head element has count of used elements instead
      next; // size of array is sentinel
  };
};

template<class T>
struct particle : particle_base {
  particle(size_type s, size_type p, size_type n) : free{p, n}, skip(s) {}
  // This class is indestructible; we run T's destructor when necessary.
  template<class... AA>
  FLECSI_INLINE_TARGET link emplace(AA &&... aa) {
    struct guard {
      FLECSI_INLINE_TARGET guard(particle & p) : p(p), ret(p.free) {}
      FLECSI_INLINE_TARGET ~guard() {
        if(fail)
          p.free = ret;
      }
      particle & p;
      link ret;
      bool fail = true;
    } g(*this);
    new(&data) T(std::forward<AA>(aa)...);
    g.fail = false;
    return g.ret;
  }
  void reset() noexcept {
    data.~T();
  }
  void reset(size_type p, size_type n) noexcept {
    reset();
    free = {p, n};
  }

  union
  {
    T data; // pointer-interconvertible if standard-layout
    link free;
  };
  // To avoid a separate field allocation, this member has several meanings:
  // If first element (never part of a run): index of head of free list
  // Otherwise, if data exists: 0
  // Otherwise, if first or last of its free run: length of the run
  // Otherwise: unused
  // If the first slot is free, it is always the head of the free list.
  size_type skip;
};
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

  // construct references just from field IDs.
  field_reference(field_id_t f, topology_t & t) : fid_(f), topology_(&t) {}

  field_id_t fid() const {
    return fid_;
  }
  topology_t & topology() const {
    return *topology_;
  } // topology_identifier

  // Some of these types vary across topologies:
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

  auto & get_ragged() const {
    // A ragged_partition<...>::core, or borrowing of same:
    return topology_->ragged.template get<Space>()[fid_];
  }
  void cleanup(std::function<void()> f) const {
    detail::get_cleanup<Space> (*topology_)(fid_, std::move(f));
  }

  template<layout L2, class T2 = T> // TODO: allow only safe casts
  auto cast() const {
    return field_reference<T2, L2, Topo, Space>(fid_, *topology_);
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

/// Identifies a field on a \c\ref mapping.
/// Declare a task parameter as a `multi<accessor<...>>` to use the field.
template<class T, layout L, class Topo, typename Topo::index_space S>
struct multi_reference : convert_tag {
  using Map = launch::mapping<Topo>;

  multi_reference(field_id_t f, Map & m) : f(f), m(&m) {}
  // Note that no correspondence between f and m is checked.
  multi_reference(const field_reference<T, L, Topo, S> & f, Map & m)
    : multi_reference(f.fid(), m) {}

  Map & map() const {
    return *m;
  }

  // i indexes into the depth of the map rather than being a color directly.
  field_reference<T, L, typename Map::Borrow, S> data(Color i) const {
    return {f, map()[i]};
  }

private:
  field_id_t f;
  Map * m;
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
/// \tparam T field value type:
///   - if any non-MPI tasks use the field, \c T must be a trivially copyable
///     type with no pointers or references
///   - if any instance of the field is resized, \c T must be trivially
///     relocatable; this weaker property is not formally recognized by the
///     language, but common implementations of \c std::vector and \c
///     std::unique_ptr qualify
/// \tparam L data layout
template<class T, data::layout L = data::dense>
struct field : data::detail::field_base<T, L> {
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
  /// Instances may be freely copied; they must all be created before any
  /// instance of \a Topo.
  ///
  /// \warning
  /// Field definitions are typically declared \c const. If placed in a header
  /// make sure to declare them as <tt>inline const</tt> to avoid breaking ODR.
  ///
  /// \tparam Topo (specialized) topology type
  /// \tparam Space index space
  template<class Topo, typename Topo::index_space Space = Topo::default_space()>
  struct definition : Register<Topo, Space> {
    using Topology = Topo;
    using Field = field;

    /// Return a reference to a field instance.
    /// \param t topology instance (must be allocated)
    auto operator()(data::topology_slot<Topo> & t) const {
      return (*this)(t.get());
    }
    Reference<Topo, Space> operator()(typename Topo::core & t) const {
      return {this->fid, t};
    }
    // For indirect and borrow topologies:
    template<template<class> class C, class P>
    std::enable_if_t<std::is_same_v<typename P::Base, Topo>,
      Reference<P, Space>>
    operator()(C<P> & t) const {
      return {this->fid, t};
    }
    /// Return a reference to a mapped field instance.
    data::multi_reference<T, L, Topo, Space> operator()(
      data::launch::mapping<Topo> & m) const {
      return {this->fid, m};
    }
  };

  /// Fields cannot be constructed.  Use \c definition instead.
  field() = delete;

#ifdef DOXYGEN
  /// Reduction accessor corresponding to this field type.
  /// This is only implemented for a Dense layout.
  /// \sa data::reduction_accessor
  template<class R>
  using reduction = reduction_accessor<R, T>;
#endif
};
/// \}

namespace data {
/// \addtogroup data
/// \{
namespace detail {
template<class T>
struct field_base<T, dense> {
  using base_type = field<T, raw>;
  template<class R>
  using reduction = reduction_accessor<R, T>;
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
template<class T>
struct field_base<T, data::particle> {
  using base_type = field<particle<T>, raw>;
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
  using base_type = typename accessor_member::accessor;
  accessor_member() : base_type(F.fid) {}
  using base_type::operator=; // for single

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

  base_type & get_base() {
    return *this;
  }
  const base_type & get_base() const {
    return *this;
  }
};

namespace detail {
template<class T>
struct scalar_value;
struct init_needed {};
} // namespace detail

/// \}
} // namespace data
} // namespace flecsi

#endif
