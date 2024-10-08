// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_ACCESSOR_HH
#define FLECSI_DATA_ACCESSOR_HH

#include "flecsi/data/topology_accessor.hh"
#include "flecsi/exec/fwd.hh"
#include "flecsi/exec/launch.hh"
#include "flecsi/topo/size.hh"
#include "flecsi/util/array_ref.hh"
#include <flecsi/data/field.hh>

#include <algorithm>
#include <iterator>
#include <memory>
#include <stack>

namespace flecsi {
namespace data {
/// \addtogroup data
/// \{

template<class>
struct multi;

namespace detail {
template<class F>
void
require_host(F && f) {
  host_only h; // lvalue required
  std::forward<F>(f)(h, [](auto &) { return nullptr; });
}

template<Privileges P, class S, class F>
void
construct(S && s, F && f) {
  if constexpr(privilege_discard(P) &&
               !std::is_trivially_default_constructible_v<
                 std::remove_reference_t<
                   decltype(*std::declval<S>()().begin())>>) {
    require_host(std::forward<F>(f));
    auto && a = std::forward<S>(s)();
    std::uninitialized_default_construct(a.begin(), a.end());
  }
}

template<class T, layout L, Privileges P, bool Span>
void
destroy_task(typename field<T, L>::template accessor1<P> a) {
  const auto && s = [&] {
    if constexpr(Span)
      return a.span();
    else
      return a;
  }();
  std::destroy(s.begin(), s.end());
}
template<Privileges P,
  bool Span = true,
  class T,
  layout L,
  class Topo,
  typename Topo::index_space S>
void
destroy(const field_reference<T, L, Topo, S> & r) {
  execute<destroy_task<T, L, privilege_repeat<rw, privilege_count(P)>, Span>,
    portable_v<T> ? loc | leaf : flecsi::mpi>(r);
}
template<class T, Privileges P>
using element_t = std::conditional_t<privilege_write(P), T, const T>;
template<class T, Privileges P, bool M>
using particle_raw =
  typename field<T, data::particle>::base_type::template accessor1<
    !M && get_privilege(0, P) == wo ? privilege_pack<rw> : P>;

template<class A, class = void>
struct multi_buffer {};
template<class A>
struct multi_buffer<A, util::voided<typename A::TaskBuffer>> {
  using TaskBuffer = std::vector<typename A::TaskBuffer>;
  void buffer(TaskBuffer & b) {
    const auto aa = static_cast<multi<A> &>(*this).accessors();
    // NB: Some of these will be unused because the accessors are discarded.
    b.resize(aa.size());
    auto i = b.begin();
    for(auto & a : aa)
      a.buffer(*i++);
  }
};
} // namespace detail

// All accessors are ultimately implemented in terms of those for the raw
// layout, minimizing the amount of backend-specific code required.

/// Accessor for a single value.  \gpu.
template<typename DATA_TYPE, Privileges PRIVILEGES>
struct accessor<single, DATA_TYPE, PRIVILEGES> : bind_tag, send_tag {
  using value_type = DATA_TYPE;
  // We don't actually inherit from base_type; we don't want its interface.
  using base_type = accessor<dense, DATA_TYPE, PRIVILEGES>;
  using element_type = typename base_type::element_type;

  explicit accessor(std::size_t s) : base(s) {}
  accessor(const base_type & b) : base(b) {}

  /// Get the value.
  FLECSI_INLINE_TARGET element_type & get() const {
    return base(0);
  } // data
  /// Convert to the value.
  FLECSI_INLINE_TARGET operator element_type &() const {
    return get();
  } // value

  /// Assign to the value.
  FLECSI_INLINE_TARGET const accessor & operator=(
    const DATA_TYPE & value) const {
    return const_cast<accessor &>(*this) = value;
  } // operator=
  /// Assign to the value.
  FLECSI_INLINE_TARGET accessor & operator=(const DATA_TYPE & value) {
    get() = value;
    return *this;
  } // operator=

  /// Get the value.
  FLECSI_INLINE_TARGET element_type & operator*() const {
    return get();
  }
  /// Access a member of the value.
  FLECSI_INLINE_TARGET element_type * operator->() const {
    return &get();
  } // operator->

  base_type & get_base() {
    return base;
  }
  const base_type & get_base() const {
    return base;
  }
  template<class F>
  void send(F && f) {
    std::forward<F>(f)(
      get_base(), [](const auto & r) { return r.template cast<dense>(); });
  }

private:
  base_type base;
}; // struct accessor

/// Accessor for computing reductions. This class is supported for GPU
/// execution.
/// Name via \c field::reduction.
/// Usable only with the global topology.  Pass a normal \c field_reference.
/// The previous field value contributes to the result.
/// \tparam R \ref fold "reduction operation"
/// \tparam T data type
template<class R, typename T>
struct reduction_accessor : bind_tag {
  using element_type = T;
  using size_type = typename util::span<element_type>::size_type;

  explicit reduction_accessor(field_id_t f) : f(f) {}

  /// Prepare to update en element.
  /// \return a callable that merges its \p T argument into the field element
  FLECSI_INLINE_TARGET
  auto operator[](size_type index) const {
    return [&v = s[index]](const T & r) { v = R::combine(v, r); };
  }

  field_id_t field() const {
    return f;
  }

  void bind(util::span<element_type> x) { // for bind_accessors
    s = x;
  }

  /// Access the underlying elements.
  FLECSI_INLINE_TARGET
  util::span<element_type> span() const {
    return s;
  }

private:
  field_id_t f;
  util::span<element_type> s;
};

/// Accessor for potentially uninitialized memory.
/// \gpu.
template<typename DATA_TYPE, Privileges PRIVILEGES>
struct accessor<raw, DATA_TYPE, PRIVILEGES> : bind_tag {
  using value_type = DATA_TYPE;
  using element_type = detail::element_t<DATA_TYPE, PRIVILEGES>;

  explicit accessor(field_id_t f) : f(f) {}

  field_id_t field() const {
    return f;
  }

  /// Get the allocated memory.
  /// \return \c util::span
  FLECSI_INLINE_TARGET
  auto span() const {
    return s;
  }

  void bind(util::span<element_type> x) { // for bind_accessors
    s = x;
  }

private:
  field_id_t f;
  util::span<element_type> s;
}; // struct accessor

/// Accessor for ordinary fields.
/// \gpu.
/// \see \link accessor<raw,DATA_TYPE,PRIVILEGES> the base class\endlink
template<class T, Privileges P>
struct accessor<dense, T, P> : accessor<raw, T, P>, send_tag {
  using base_type = accessor<raw, T, P>;
  using base_type::base_type;
  using size_type = typename decltype(base_type(0).span())::size_type;

  accessor(const base_type & b) : base_type(b) {}

  /// Index with bounds checking (except with \c NDEBUG).
  FLECSI_INLINE_TARGET
  typename accessor::element_type & operator()(size_type index) const {
    const auto s = this->span();
    assert(index < s.size() && "index out of range");
    return s[index];
  } // operator()

  /// Index without bounds checking (even without \c NDEBUG).
  FLECSI_INLINE_TARGET
  typename accessor::element_type & operator[](size_type index) const {
    return this->span()[index];
  }

  base_type & get_base() {
    return *this;
  }
  const base_type & get_base() const {
    return *this;
  }
  template<class F>
  void send(F && f) {
    std::forward<F>(f)(get_base(), [](const auto & r) {
      // TODO: use just one task for all fields
      if constexpr(privilege_discard(P) && !std::is_trivially_destructible_v<T>)
        r.cleanup([r] { detail::destroy<P>(r); });
      return r.template cast<raw>();
    });
    detail::construct<P>([this] { return this->span(); }, std::forward<F>(f));
  }
};

// The offsets privileges are separate because they are writable for mutators
// but read-only for even writable accessors.

/// Accessor for ragged fields.  \gpu.
/// \tparam P if write-only, rows do not change size but their elements are
///   reinitialized
/// \if core
/// \tparam OP for offsets
/// \endif
/// \see \link accessor<raw,DATA_TYPE,PRIVILEGES> the base class\endlink
template<class T, Privileges P, Privileges OP>
struct ragged_accessor
  : accessor<raw, T, privilege_pack<privilege_merge(P)>>,
    send_tag,
    util::with_index_iterator<const ragged_accessor<T, P, OP>> {
  using base_type = typename ragged_accessor::accessor;
  using typename base_type::element_type;
  using Offsets = accessor<dense, std::size_t, OP>;
  using Offset = typename Offsets::value_type;
  using size_type = typename Offsets::size_type;
  using row = util::span<element_type>;

  using base_type::base_type;
  ragged_accessor(const base_type & b) : base_type(b) {}

  /// Get the row at an index point.
  /// \return \c util::span
  FLECSI_INLINE_TARGET row operator[](size_type i) const {
    // Without an extra element, we must store one endpoint implicitly.
    // Storing the end usefully ignores any overallocation.
    return this->span().first(off(i)).subspan(i ? off(i - 1) : 0);
  }
  /// Return the number of rows.
  FLECSI_INLINE_TARGET size_type size() const noexcept {
    return off.span().size();
  }
  /// Return the number of elements across all rows.
  FLECSI_INLINE_TARGET Offset total() const noexcept {
    const auto s = off.span();
    return s.empty() ? 0 : s.back();
  }

  /// Get the elements without any row structure.
  FLECSI_INLINE_TARGET util::span<element_type> span() const {
    return get_base().span().first(total());
  }

  base_type & get_base() {
    return *this;
  }
  FLECSI_INLINE_TARGET const base_type & get_base() const {
    return *this;
  }

  Offsets & get_offsets() {
    return off;
  }
  const Offsets & get_offsets() const {
    return off;
  }
  template<class F>
  void send(F && f) {
    f(get_base(), [](const auto & r) {
      const field_id_t i = r.fid();
      // The dirty flag on the ordinary associated region is used (in a
      // trivial sense) for the offsets that are never copied directly.  Use
      // the flag on the ragged-elements region for the actual ghost copy:
      auto & t = r.get_elements();
      if constexpr(privilege_count(P) > 1) {
        region & reg = t.template get_region<topo::elements>();
        [[maybe_unused]] const copy_plan * const cp = reg.ghost_copy<P>(r);
        flog_assert(!cp, "copy_plan selected for ragged field");
        reg.ghost<P>(i); // restore any dirty flag cleared by the copy itself
      }
      if constexpr(!std::is_trivially_destructible_v<T>)
        r.cleanup([r] { detail::destroy<P>(r); });
      // Resize after the ghost copy (which can add elements and can perform
      // its own resize) rather than in the mutator before getting here:
      if constexpr(privilege_write(OP))
        t.resize();
      return field_reference<T,
        raw,
        topo::policy_t<std::remove_reference_t<decltype(t)>>,
        topo::elements>(i, t);
    });
    std::forward<F>(f)(get_offsets(),
      [](const auto & r) { return r.template cast<dense, Offset>(); });
    // These do nothing on the caller side:
    if constexpr(privilege_discard(OP)) {
      const auto s = off.span();
      std::fill(s.begin(), s.end(), 0);
    }
    else
      detail::construct<P>([this] { return span(); }, std::forward<F>(f));
  }

private:
  Offsets off{this->field()};
};

template<class T, Privileges P>
struct accessor<ragged, T, P>
  : ragged_accessor<T, P, privilege_repeat<ro, privilege_count(P)>> {
  using accessor::ragged_accessor::ragged_accessor;
};

/// Mutator for ragged fields.
/// Sizes are \ref topo::repartition::resize "applied" before launching the
/// task.  New sizes are \ref topo::with_size::growth "chosen" for the next
/// task based on usage.
/// Cannot be used while tracing or in a GPU task.
/// \tparam P if write-only, all rows are discarded
template<class T, Privileges P>
struct mutator<ragged, T, P>
  : bind_tag, send_tag, util::with_index_iterator<const mutator<ragged, T, P>> {
  static_assert(std::is_nothrow_move_constructible_v<T>,
    "the data type should not throw from a move constructor.");
  static_assert(std::is_nothrow_move_assignable_v<T>,
    "the data type should not throw from a move assignment.");
  static_assert(std::is_nothrow_destructible_v<T>,
    "the data type should not throw from a destructor.");

  using base_type = ragged_accessor<T,
    P,
    privilege_repeat<privilege_discard(P) ? wo : rw, privilege_count(P)>>;
  using size_type = typename base_type::size_type;

  struct Overflow {
    size_type del;
    std::vector<T> buffer;
  };

  using TaskBuffer = std::vector<Overflow>;

private:
  using base_row = typename base_type::row;
  using span_iterator = typename base_row::iterator; // T*
  using buffer_iterator = typename std::vector<T>::iterator;

  // Simple wrapper for basic access to a row and its overflow buffer.
  // Additionally provides a public interface for certain vector-like operations
  // on the span.
  struct raw_row {
    using size_type = mutator::size_type;

    base_row span;
    Overflow * overflow;

    T & operator[](size_type i) const noexcept {
      assert(i < size() && "index out of range");
      return i < span.size() ? span[i] : overflow->buffer[i - span.size()];
    }

    size_type size() const noexcept {
      return active_size() + overflow->buffer.size();
    }

    auto active_end() const noexcept {
      return span.end() - overflow->del;
    }

    auto active_size() const noexcept {
      return span.size() - overflow->del;
    }

  }; // struct raw_row

public:
  /// A row handle.  Provides the \c std::vector interface, with the important
  /// difference that there is no guarantee of contiguity between memory
  /// locations.
  struct row : util::with_index_iterator<const row>, private raw_row {
    using value_type = T;
    using typename raw_row::size_type;
    using difference_type = typename base_row::difference_type;
    using typename row::with_index_iterator::iterator;

    row(const raw_row & r) : raw_row(r) {}

    /// \name std::vector operations
    /// \{

    void assign(size_type count, const T & value) const {
      clear();
      resize(count, value);
    }

    template<class I, class = std::enable_if_t<!std::is_integral_v<I>>>
    void assign(I first, I last) const {
      clear();
      insert(this->begin(), first, last);
    }

    void assign(std::initializer_list<T> ilist) const {
      assign(ilist.begin(), ilist.end());
    }

    using raw_row::operator[];

    T & front() const noexcept {
      assert(size() > 0 && "cannot access the front of an empty row");
      return *this->begin();
    }

    T & back() const noexcept {
      assert(size() > 0 && "cannot access the back of an empty row");
      return this->end()[-1];
    }

    using raw_row::size;

    [[nodiscard]] bool empty() const noexcept {
      return this->size() == 0;
    }

    size_type max_size() const noexcept {
      return overflow->buffer.max_size();
    }

    size_type capacity() const noexcept {
      return span.size() + overflow->buffer.capacity();
    }

    void reserve(size_type n) const {
      if(n > span.size()) {
        overflow->buffer.reserve(n - span.size());
      }
    }

    void shrink_to_fit() const {
      // The span is always "shrunk to fit"
      overflow->buffer.shrink_to_fit();
    }

    void clear() const noexcept {
      erase(this->begin(), this->end());
    }

    iterator insert(iterator pos, const T & value) const {
      return emplace(pos, value);
    }

    iterator insert(iterator pos, T && value) const {
      return emplace(pos, std::move(value));
    }

    iterator insert(iterator pos, size_type count, const T & value) const {
      assert(pos <= this->end() && "cannot insert beyond end of the row");
      auto const middle = this->end();
      { // scope to unwind push_back calls if we fail partway through
        auto guard = get_exception_guard();
        size_type const scount = std::min(count, overflow->del);
        size_type const bcount = count - scount;
        std::uninitialized_fill_n(active_end(), scount, value);
        overflow->del -= scount;
        overflow->buffer.insert(overflow->buffer.end(), bcount, value);
        guard.reset = false;
      } // end of unwinding scope
      std::rotate(pos, middle, this->end());
      return pos;
    }

    template<class I, class = std::enable_if_t<!std::is_integral_v<I>>>
    iterator insert(iterator pos, I first, I last) const {
      assert(pos <= this->end() && "cannot insert beyond end of the row");
      auto const middle = this->end();
      { // scope to unwind push_back calls if we fail partway through
        auto guard = get_exception_guard();
        for(; (first != last) && (overflow->del > 0); ++first) {
          emplace_back_span(*first);
        }
        overflow->buffer.insert(overflow->buffer.end(), first, last);
        guard.reset = false;
      } // end of unwinding scope
      std::rotate(pos, middle, this->end());
      return pos;
    }

    iterator insert(iterator pos, std::initializer_list<T> ilist) const {
      return insert(pos, ilist.begin(), ilist.end());
    }

    template<class... AA>
    iterator emplace(iterator pos, AA &&... aa) const {
      assert(pos <= this->end() && "cannot insert beyond end of the row");
      auto const middle = this->end();
      emplace_back(std::forward<AA>(aa)...);
      std::rotate(pos, middle, this->end());
      return pos;
    }

    iterator erase(iterator pos) const noexcept {
      return erase(pos, pos + 1);
    }

    iterator erase(iterator first, iterator last) const noexcept {
      assert(first <= last && "range reversed");
      assert(
        last.index() <= size() && "cannot erase beyond the end of the row");
      if(first.index() >= span.size()) {
        // entirely in buffer
        overflow->buffer.erase(buffer_iter(first), buffer_iter(last));
      }
      else if(first != last) { // first std::move would fail
        bool const erase_buffer = (last.index() >= span.size());
        auto const sfirst = span_iter(first);
        auto const slast = erase_buffer ? span.end() : span_iter(last);
        size_type const span_erase_count = slast - sfirst;
        auto const new_end = active_end() - span_erase_count;
        auto const transfer_first =
          erase_buffer ? buffer_iter(last) : overflow->buffer.begin();
        size_type const transfer_count = std::min<size_type>(
          span_erase_count, overflow->buffer.end() - transfer_first);
        auto const transfer_last = transfer_first + transfer_count;
        std::move(slast, active_end(), sfirst);
        std::move(transfer_first, transfer_last, new_end);
        std::destroy(new_end + transfer_count, active_end());
        overflow->del += span_erase_count - transfer_count;
        overflow->buffer.erase(overflow->buffer.begin(), transfer_last);
      }
      return first;
    }

    void push_back(const T & t) const {
      emplace_back(t);
    }

    void push_back(T && t) const {
      emplace_back(std::move(t));
    }

    template<class... AA>
    T & emplace_back(AA &&... aa) const {
      if(overflow->del > 0) {
        emplace_back_span(std::forward<AA>(aa)...);
      }
      else {
        // case 2: append to buffer
        overflow->buffer.emplace_back(std::forward<AA>(aa)...);
      }
      return back();
    }

    void pop_back() const noexcept {
      erase(this->end() - 1);
    }

    void resize(size_type count) const {
      resize_(count);
    }

    void resize(size_type count, const T & value) const {
      resize_(count, value);
    }
    /// \}

    // No swap: it would swap the handles, not the contents

  private:
    using raw_row::active_end;
    using raw_row::active_size;
    using raw_row::overflow;
    using raw_row::span;

    // Allows us to unwind any appends if an exception occurs, without using
    // try/catch blocks (which aren't allowed on GPUs).
    struct exception_guard {
      const row & r;
      size_type initial_size;
      bool reset{true};
      ~exception_guard() {
        if(reset) {
          r.resize(initial_size);
        }
      }
    };

    auto get_exception_guard() const {
      return exception_guard{*this, size()};
    }

    /// Translates a row::iterator to a span iterator.  Changes the type from
    /// row::iterator to the iterator type expected by the span.
    auto span_iter(iterator iter) const noexcept {
      assert(iter.index() < span.size());
      return span.begin() + iter.index();
    }

    /// Translates a row::iterator to a buffer iterator.  Changes the type from
    /// row::iterator to the iterator type expected by the buffer, and shifts
    /// by span.size() to account for the fact that some elements are in the
    /// span instead of the buffer.
    auto buffer_iter(iterator iter) const noexcept {
      assert(iter.index() >= span.size());
      return overflow->buffer.begin() + (iter.index() - span.size());
    }

    template<class... AA>
    void emplace_back_span(AA &&... aa) const {
      assert(overflow->del > 0 && "no space to add new elements");
      new(&*active_end()) T(std::forward<AA>(aa)...);
      --overflow->del;
    }

    // We know this is only ever called with no arguments or `T const &`
    template<class... AA>
    void resize_(size_type count, AA const &... aa) const {
      static_assert(
        sizeof...(AA) <= 1, "resize_ can take only 0 or 1 arguments");
      if(count < size()) {
        erase(this->begin() + count, this->end());
      }
      else {
        auto guard = get_exception_guard();
        std::size_t num_add = count - size();
        for(; (overflow->del > 0) && (num_add > 0); --num_add) {
          emplace_back_span(aa...);
        }
        overflow->buffer.resize(overflow->buffer.size() + num_add, aa...);
        guard.reset = false;
      }
    }

  }; // struct row

  mutator(const base_type & b, const topo::resize::policy & p)
    : acc(b), grow(p) {}

  /// Get the row at an index point.
  row operator[](size_type i) const {
    return raw_get(i);
  }
  /// Get the number of rows.
  size_type size() const noexcept {
    return acc.size();
  }

  base_type & get_base() {
    return acc;
  }
  const base_type & get_base() const {
    return acc;
  }
  auto & get_size() {
    return sz;
  }
  const topo::resize::policy & get_grow() const {
    return grow;
  }
  void buffer(TaskBuffer & b) { // for unbind_accessors
    over = &b;
  }
  template<class F>
  void send(F && f) {
    detail::require_host(f);
    f(get_base(), util::identity());
    std::forward<F>(f)(
      get_size(), [](const auto & r) { return r.get_elements().sizes(); });
    if(over)
      over->resize(acc.size()); // no-op on caller side
  }

  /// \cond core
  /*! Repack the data within the raw layout.  Shrink the storage for each row
   * that got shorter. Expand the storage for each row that got longer.
   * Rearrange data to fit into the new layout, taking care not to overwrite
   * something that's not yet been relocated.
   *
   * The total number of elements across all rows is allowed to change, but the
   * total memory required is not allowed to increase.  That is, the slack space
   * at the end of the raw layout can increase or decrease, but if you run out
   * of slack space at the end of the raw layout then the commit operation fails
   * because there's not enough memory allocated for the new layout.
   */
  void commit() const {
    const auto all = acc.get_base().span(); // the raw layout as a span
    const size_type num_rows = size();
    // Compute new endpoints
    std::vector<size_type> new_endpoints;
    new_endpoints.reserve(num_rows);
    {
      size_type end = 0;
      for(size_type irow{0}; irow < num_rows; ++irow) {
        new_endpoints.push_back(end += raw_get(irow).size());
      }
      assert(end <= all.size() &&
             "Not enough storage to commit changes for ragged mutator.");
    }
    // Get old endpoints
    auto & old_endpoints = acc.get_offsets();
    // I keep getting off-by-one errors due to the structure of the endpoints
    // arrays, so wrap them
    auto new_offsets = [&new_endpoints](size_type const irow) {
      return irow == 0 ? 0 : new_endpoints[irow - 1];
    };
    auto old_offsets = [&old_endpoints](size_type const irow) {
      return irow == 0 ? 0 : old_endpoints[irow - 1];
    };
    // Define row shift operations
    auto shift_row = [&all](auto const old_offset,
                       auto const num_active,
                       auto const new_offset,
                       auto left) {
      auto const shift_distance =
        left ? old_offset - new_offset : new_offset - old_offset;
      auto const min_length = std::min(shift_distance, num_active);
      auto const old_first = all.begin() + old_offset;
      auto const old_last = old_first + num_active;
      if constexpr(left) {
        auto const split = old_first + min_length;
        std::uninitialized_move(old_first, split, old_first - shift_distance);
        std::move(split, old_last, split - shift_distance);
        std::destroy(old_last - min_length, old_last);
      }
      else {
        auto const split = old_last - min_length;
        std::uninitialized_move(split, old_last, split + shift_distance);
        std::move_backward(old_first, split, old_last);
        std::destroy(old_first, old_first + min_length);
      }
    };
    // Shift the existing span data within the raw layout (neglecting buffers
    // for the moment)
    for(size_type irow{0}; irow < num_rows; /*advancement handled below*/) {
      auto const new_offset = new_offsets(irow);
      auto const old_offset = old_offsets(irow);
      if(new_offset == old_offset) { // This row stays put
        ++irow;
      }
      else if(new_offset < old_offset) { // This row shifts left
        shift_row(old_offset,
          raw_get(irow).active_size(),
          new_offset,
          std::true_type());
        ++irow;
      }
      else { // This row begins a shift-right block
        // Find the end of this shift-right block (may include up to last row)
        // -- A shift-left or stay-put block implies that there's enough space
        //    at the end of this block for all the shift-right operations to
        //    happen.
        size_type irow_end{irow};
        while((++irow_end < num_rows) &&
              (new_offsets(irow_end) > old_offsets(irow_end)))
          ;
        // Shift
        for(size_type j{irow_end - 1}; j >= irow; --j) {
          shift_row(old_offsets(j),
            raw_get(j).active_size(),
            new_offsets(j),
            std::false_type());
        }
        // Advance to the next non-shift-right row
        irow = irow_end;
      }
    }
    // Store buffers
    for(size_type irow{0}; irow < num_rows; ++irow) {
      auto & overflow = (*over)[irow];
      std::uninitialized_move(overflow.buffer.begin(),
        overflow.buffer.end(),
        all.begin() + (new_offsets(irow + 1) - overflow.buffer.size()));
      overflow.buffer.clear();
      overflow.del = 0;
    }
    // Update endpoints
    std::copy(
      new_endpoints.begin(), new_endpoints.end(), old_endpoints.span().begin());
    // Set new size for later resizing of backing storage
    sz = grow(acc.total(), all.size());
  } // commit()
  /// \endcond

private:
  raw_row raw_get(size_type i) const {
    return {get_base()[i], &(*over)[i]};
  }

  base_type acc;
  topo::resize::accessor<wo> sz;
  topo::resize::policy grow;
  TaskBuffer * over = nullptr;
}; // struct mutator<ragged, T, P>

// Many compilers incorrectly require the 'template' for a base class.

/// Accessor for sparse fields.  \gpu.
/// \tparam P cannot be write-only
/// \see \link ragged_accessor the base class\endlink
template<class T, Privileges P>
struct accessor<sparse, T, P>
  : field<T, sparse>::base_type::template accessor1<P>,
    util::with_index_iterator<const accessor<sparse, T, P>> {
  static_assert(!privilege_discard(P),
    "sparse accessor requires read permission");

private:
  using Field = field<T, sparse>;
  using FieldBase = typename Field::base_type;

public:
  using value_type = typename FieldBase::value_type;
  using base_type = typename FieldBase::template accessor1<P>;
  using element_type =
    std::tuple_element_t<1, typename base_type::element_type>;

private:
  using base_row = typename base_type::row;

public:
  /// A mapping backed by a row.  Use the \c ragged interface to iterate.
  struct row {
    using key_type = typename Field::key_type;
    FLECSI_INLINE_TARGET row(base_row s) : s(s) {}
    /// Find an element.
    /// \param c must exist
    FLECSI_INLINE_TARGET element_type & operator()(key_type c) const {
      return util::partition_point(s, [c](const value_type & v) {
        return v.first < c;
      })->second;
    }

  private:
    base_row s;
  };

  using base_type::base_type;
  accessor(const base_type & b) : base_type(b) {}

  /// Get the row at an index point.
  FLECSI_INLINE_TARGET row operator[](typename accessor::size_type i) const {
    return get_base()[i];
  }

  base_type & get_base() {
    return *this;
  }
  FLECSI_INLINE_TARGET const base_type & get_base() const {
    return *this;
  }
  template<class F>
  void send(F && f) {
    std::forward<F>(f)(get_base(),
      [](const auto & r) { return r.template cast<ragged, value_type>(); });
  }
};

/// Mutator for sparse fields.
/// Cannot be used while tracing or in a GPU task.
/// \tparam P if write-only, all rows are discarded
template<class T, Privileges P>
struct mutator<sparse, T, P>
  : bind_tag, send_tag, util::with_index_iterator<const mutator<sparse, T, P>> {
private:
  using Field = field<T, sparse>;

public:
  using base_type = typename Field::base_type::template mutator1<P>;
  using size_type = typename base_type::size_type;
  using TaskBuffer = typename base_type::TaskBuffer;

private:
  using base_row = typename base_type::row;
  using base_iterator = typename base_row::iterator;

public:
  /// A row handle.
  struct row {
    using key_type = typename Field::key_type;
    using value_type = typename base_row::value_type;
    using size_type = typename base_row::size_type;

    /// Bidirectional iterator over key-value pairs.
    /// \warning Unlike \c std::map::iterator, this sort of iterator is
    ///   invalidated by insertions/deletions.
    struct iterator {
    public:
      using value_type = std::pair<const key_type &, T &>;
      using difference_type = std::ptrdiff_t;
      using reference = value_type;
      using pointer = void;
      // We could easily implement random access, but std::map doesn't.
      using iterator_category = std::bidirectional_iterator_tag;

      iterator(base_iterator i = {}) : i(i) {}

      /// Get an element.
      /// \return a pair of references
      value_type operator*() const {
        return {i->first, i->second};
      }

      iterator & operator++() {
        ++i;
        return *this;
      }
      iterator operator++(int) {
        iterator ret = *this;
        ++*this;
        return ret;
      }
      iterator & operator--() {
        --i;
        return *this;
      }
      iterator operator--(int) {
        iterator ret = *this;
        --*this;
        return ret;
      }

      bool operator==(const iterator & o) const {
        return i == o.i;
      }
      bool operator!=(const iterator & o) const {
        return i != o.i;
      }

      base_iterator get_base() const {
        return i;
      }

    private:
      base_iterator i;
    };

    row(base_row r) : r(r) {}

    /// \name std::map operations
    /// \{
    T & operator[](key_type c) const {
      return try_emplace(c).first->second;
    }

    iterator begin() const noexcept {
      return r.begin();
    }
    iterator end() const noexcept {
      return r.end();
    }

    [[nodiscard]] bool empty() const noexcept {
      return r.empty();
    }
    size_type size() const noexcept {
      return r.size();
    }
    size_type max_size() const noexcept {
      return r.max_size();
    }

    void clear() const noexcept {
      r.clear();
    }
    std::pair<iterator, bool> insert(const value_type & p) const {
      auto [i, hit] = lookup(p.first);
      if(!hit)
        i = r.insert(i, p); // assignment is no-op
      return {i, !hit};
    }
    // TODO: insert(U&&), insert(value_type&&)
    template<class I>
    void insert(I a, I b) const {
      for(; a != b; ++a)
        insert(*a);
    }
    void insert(std::initializer_list<value_type> l) const {
      insert(l.begin(), l.end());
    }
    template<class U>
    std::pair<iterator, bool> insert_or_assign(key_type c, U && u) const {
      auto [i, hit] = lookup(c);
      if(hit)
        i->second = std::forward<U>(u);
      else
        i = r.insert(i, {c, std::forward<U>(u)}); // assignment is no-op
      return {i, !hit};
    }
    // We don't support emplace since we can't avoid moving the result.
    template<class... AA>
    std::pair<iterator, bool> try_emplace(key_type c, AA &&... aa) const {
      auto [i, hit] = lookup(c);
      if(!hit)
        i = r.insert(i,
          {std::piecewise_construct,
            std::make_tuple(c),
            std::forward_as_tuple(
              std::forward<AA>(aa)...)}); // assignment is no-op
      return {i, !hit};
    }

    iterator erase(iterator i) const {
      return r.erase(i.get_base());
    }
    iterator erase(iterator i, iterator j) const {
      return r.erase(i.get_base(), j.get_base());
    }
    size_type erase(key_type c) const {
      const auto [i, hit] = lookup(c);
      if(hit)
        r.erase(i);
      return hit;
    }
    // No swap: it would swap the handles, not the contents

    size_type count(key_type c) const {
      return lookup(c).second;
    }
    iterator find(key_type c) const {
      const auto [i, hit] = lookup(c);
      return hit ? i : end();
    }
    std::pair<iterator, iterator> equal_range(key_type c) const {
      const auto [i, hit] = lookup(c);
      return {i, i + hit};
    }
    iterator lower_bound(key_type c) const {
      return lower(c);
    }
    iterator upper_bound(key_type c) const {
      const auto [i, hit] = lookup(c);
      return i + hit;
    }
    /// \}

  private:
    base_iterator lower(key_type c) const {
      return std::partition_point(
        r.begin(), r.end(), [c](const value_type & v) { return v.first < c; });
    }
    std::pair<base_iterator, bool> lookup(key_type c) const {
      const auto i = lower(c);
      return {i, i != r.end() && i->first == c};
    }

    // We simply keep the (ragged) row sorted; this avoids the complexity of
    // two lookaside structures and is efficient for small numbers of inserted
    // elements and for in-order initialization.
    base_row r;
  };

  mutator(const base_type & b) : rag(b) {}

  /// Get the row at an index point.
  row operator[](size_type i) const {
    return get_base()[i];
  }
  /// Get the number of rows.
  size_type size() const noexcept {
    return rag.size();
  }

  base_type & get_base() {
    return rag;
  }
  const base_type & get_base() const {
    return rag;
  }
  template<class F>
  void send(F && f) {
    std::forward<F>(f)(get_base(), [](const auto & r) {
      return r.template cast<ragged, typename base_row::value_type>();
    });
  }
  void buffer(TaskBuffer & b) { // for unbind_accessors
    rag.buffer(b);
  }

  void commit() const {
    rag.commit();
  }

private:
  base_type rag;
};

/// Accessor for particle fields.  \gpu.
/// Provides bidirectional iterators over the existing particles.
/// \tparam P if write-only, particles are not created or destroyed but they
///   are reinitialized
template<class T, Privileges P, bool M>
struct particle_accessor : detail::particle_raw<T, P, M>, send_tag {
  static_assert(privilege_count(P) == 1, "particles cannot be ghosts");
  using base_type = detail::particle_raw<T, P, M>;
  using size_type = typename decltype(base_type(0).span())::size_type;
  using Particle = typename base_type::value_type;
  // Override base class aliases:
  using value_type = T;
  using element_type = detail::element_t<T, P>;

  struct iterator {
    using reference = element_type &;
    using value_type = element_type;
    using pointer = element_type *;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::bidirectional_iterator_tag;

    FLECSI_INLINE_TARGET iterator() noexcept : iterator(nullptr, 0) {}
    FLECSI_INLINE_TARGET iterator(const particle_accessor * a, size_type i)
      : a(a), i(i) {}

    FLECSI_INLINE_TARGET reference operator*() const {
      return a->span()[i].data;
    }
    FLECSI_INLINE_TARGET pointer operator->() const {
      return &**this;
    }

    FLECSI_INLINE_TARGET iterator & operator++() {
      const auto s = a->span();
      if(++i != s.size())
        i += s[i].skip;
      return *this;
    }
    FLECSI_INLINE_TARGET iterator operator++(int) {
      iterator ret = *this;
      ++*this;
      return ret;
    }
    FLECSI_INLINE_TARGET iterator & operator--() {
      if(--i)
        i -= a->span()[i].skip;
      return *this;
    }
    FLECSI_INLINE_TARGET iterator operator--(int) {
      iterator ret = *this;
      --*this;
      return ret;
    }

    FLECSI_INLINE_TARGET bool operator==(const iterator & o) const {
      return i == o.i;
    }
    FLECSI_INLINE_TARGET bool operator!=(const iterator & o) const {
      return i != o.i;
    }
    FLECSI_INLINE_TARGET bool operator<(const iterator & o) const {
      return i < o.i;
    }
    FLECSI_INLINE_TARGET bool operator<=(const iterator & o) const {
      return i <= o.i;
    }
    FLECSI_INLINE_TARGET bool operator>(const iterator & o) const {
      return i > o.i;
    }
    FLECSI_INLINE_TARGET bool operator>=(const iterator & o) const {
      return i >= o.i;
    }

    FLECSI_INLINE_TARGET size_type location() const noexcept {
      return i;
    }

  private:
    const particle_accessor * a;
    size_type i;
  };

  using base_type::base_type;
  particle_accessor(const base_type & b) : base_type(b) {}

  // This interface is a subset of that proposed for std::hive.

  /// Get the number of extant particles.
  FLECSI_INLINE_TARGET size_type size() const {
    const auto s = this->span();
    const auto n = s.size();
    const auto i = n ? s.front().skip : 0;
    return i == n ? n : s[i].free.prev;
  }
  /// Get the maximum number of particles.
  FLECSI_INLINE_TARGET size_type capacity() const {
    return this->span().size();
  }
  /// Test whether any particles exist.
  [[nodiscard]] FLECSI_INLINE_TARGET bool empty() const {
    return !size();
  }

  FLECSI_INLINE_TARGET iterator begin() const {
    const auto s = this->span();
    return {this, s.empty() || s.front().skip ? 0 : 1 + first_skip()};
  }
  FLECSI_INLINE_TARGET iterator end() const {
    return {this, capacity()};
  }

  /// Get an iterator that refers to a particle.
  /// \c T must be standard-layout.
  FLECSI_INLINE_TARGET iterator get_iterator_from_pointer(
    element_type * the_pointer) const {
    static_assert(std::is_standard_layout_v<Particle>);
    const auto * const p = reinterpret_cast<Particle *>(the_pointer);
    const auto ret = p - this->span().data();
    // Some skipfield values are unused, so this isn't reliable:
    assert(!p->skip != !ret && "field slot is empty");
    return iterator(this, ret);
  }

  base_type & get_base() {
    return *this;
  }
  FLECSI_INLINE_TARGET const base_type & get_base() const {
    return *this;
  }

  template<class F>
  void send(F && f) {
    std::forward<F>(f)(get_base(), [](const auto & r) {
      if constexpr(privilege_discard(P) && !std::is_trivially_destructible_v<T>)
        r.cleanup([r] { detail::destroy<P, false>(r); });
      return r.template cast<raw, Particle>();
    });
  }

protected:
  FLECSI_INLINE_TARGET size_type
  first_skip() const { // after special first element
    const auto s = this->span();
    return s.size() == 1 ? 0 : s[1].skip;
  }
};

template<class T, Privileges P>
struct accessor<particle, T, P> : particle_accessor<T, P, false> {
  using accessor::particle_accessor::particle_accessor;

  template<class F>
  void send(F && f) {
    accessor::particle_accessor::send(std::forward<F>(f));
    detail::construct<P>(
      [this]() -> auto & { return *this; }, std::forward<F>(f));
  }
};

/// Mutator for particle fields.
/// \gpu; however, insertions and deletions are not thread-safe.
/// Iterators are invalidated only if their particle is removed.
/// \tparam P if write-only, all particles are discarded
template<class T, Privileges P>
struct mutator<particle, T, P> : particle_accessor<T, P, true> {
  using base_type = particle_accessor<T, P, true>;
  using typename base_type::iterator;
  using typename base_type::value_type;
  using Skip = typename base_type::base_type::value_type::size_type;

  // We don't support automatic resizing since we don't control the region.
  // TODO: Support manual resizing (by updating the skipfield appropriately)
  using base_type::base_type;

  // Since, by design, the skipfield data structure requires no lookaside
  // tables, accessor could provide it instead of having this class at all.
  // However, the natural wo semantics differ between the two cases.

  /// Remove all particles.
  FLECSI_INLINE_TARGET void clear() const {
    if(!std::is_trivially_destructible_v<T>)
      std::destroy(this->begin(), this->end());
    init();
  }

  /// Add a particle.
  /// \see emplace
  FLECSI_INLINE_TARGET iterator insert(const value_type & v) const {
    return emplace(v);
  }
  /// Add a particle by moving.
  /// \see emplace
  FLECSI_INLINE_TARGET iterator insert(value_type && v) const {
    return emplace(std::move(v));
  }
  /// Create an element, in constant time.
  /// It is unspecified where it appears in the sequence.
  /// \return an iterator to the new element
  template<class... AA>
  FLECSI_INLINE_TARGET iterator emplace(AA &&... aa) const {
    const auto s = this->span();
    const auto n = s.size();
    assert(n && "no particle space");
    Skip &ptr = s.front().skip, ret = ptr;
    assert(ret != n && "too many particles");
    auto * const head = &s[ret];
    Skip & skip = head->skip;
    const auto link = head->emplace(std::forward<AA>(aa)...);
    if(!ret || skip == 1) // erasing a run
      ptr = link.next;
    else { // shrinking a run
      auto & h2 = s[++ptr];
      h2.skip = head[skip - 1].skip = skip - 1;
      h2.free.next = link.next;
    }
    if(ret)
      skip = 0;
    if(ptr != n) // update size
      s[ptr].free.prev = link.prev + 1;
    return {this, ret};
  }

  /// Remove an element, in constant time.
  /// \return an iterator past the removed element
  FLECSI_INLINE_TARGET iterator erase(const iterator & it) const {
    const auto s = this->span();
    const auto n = s.size();
    assert(n && "no particles to erase");
    Skip & ptr = s.front().skip;
    const typename mutator::size_type i = it.location();
    auto * const rm = &s[i];

    const Skip end = i > 1 ? rm[-1].skip : 0, // adjacent empty run lengths
      beg = i && i < n - 1 ? rm[1].skip : 0;
    if(i)
      rm[-end].skip = rm[beg].skip = beg + end + 1; // set up new run

    if(end)
      rm->reset(); // no links in middle of run
    // Update free list:
    if(beg) {
      const auto & f = rm[1].free;
      Skip & up = ptr == i + 1 ? ptr : s[f.prev].free.next;
      if(end)
        up = f.next; // remove right-hand run from list
      else {
        --up;
        rm->reset(f.prev, f.next);
      }
    }
    else if(!end) { // new run
      Skip & up = ptr ? ptr : s.front().free.next; // keep element 0 first
      if(!ptr && up < n)
        s[up].free.prev = i;
      rm->reset(ptr && ptr < n ? /* size: */ s[ptr].free.prev : ptr, up);
      up = i;
    }
    --s[ptr].free.prev;

    return iterator(this, 1 + (i ? i + beg : this->first_skip()));
  }

  void commit() const {}

  template<class F>
  void send(F && f) {
    base_type::send(std::forward<F>(f));
    if constexpr(privilege_discard(P))
      init(); // no-op on caller side
  }

private:
  FLECSI_INLINE_TARGET void init() const {
    const auto s = this->span();
    if(const auto n = s.size()) {
      auto & a = s.front();
      a.free = {0, 1};
      a.skip = 0;
      if(n > 1) {
        auto & b = s[1];
        b.free = {0, n};
        b.skip = s.back().skip = n - 1;
      }
    }
  }
};

namespace detail {
template<class T>
struct scalar_value : bind_tag {
  const T * device;
  T * host;

  // The backend knows what value of P to provide when processing this as a
  // "task parameter" and thus whether 'device' is really a device pointer.
  template<exec::task_processor_type_t P>
  void copy() const {
    if constexpr(P == exec::task_processor_type_t::toc) {
#if defined(__NVCC__) || defined(__CUDACC__)
      auto status = cudaMemcpy(host, device, sizeof(T), cudaMemcpyDeviceToHost);
      flog_assert(cudaSuccess == status, "Error calling cudaMemcpy");
      return;
#elif defined(__HIPCC__)
      auto status = hipMemcpy(host, device, sizeof(T), hipMemcpyDeviceToHost);
      flog_assert(hipSuccess == status, "Error calling hipMemcpy");
      return;
#else
      flog_assert(false, "CUDA or HIP should be enabled when using toc task");
#endif
    }
    *host = *device;
  }
};

template<auto & F>
struct scalar_access : bind_tag {

  typedef
    typename std::remove_reference_t<decltype(F)>::Field::value_type value_type;

  template<class Func, class S>
  void topology_send(Func && f, S && s) {
    accessor_member<F, privilege_pack<ro>> acc;
    acc.topology_send(f, std::forward<S>(s));
    // A single accessor can be empty if it is part of a multi
    if(auto * const d = acc.get_base().get_base().span().data()) {
      scalar_value<value_type> dummy{{}, d, &scalar_};
      std::forward<Func>(f)(dummy, [](auto &) { return nullptr; });
    }
  }

  FLECSI_INLINE_TARGET const value_type * operator->() const {
    return &scalar_;
  }

  FLECSI_INLINE_TARGET const value_type & operator*() const {
    return scalar_;
  }

private:
  value_type scalar_;
};
} // namespace detail

/// \cond core

/// Metadata provided with this type is made available on the host if
/// read-only even if fields are stored on a device.  Use `*sa` or `sa->` to
/// access it.
/// \tparam P produces an ordinary accessor if writable
template<const auto & F, Privileges P>
using scalar_access = std::conditional_t<privilege_merge(P) == ro,
  detail::scalar_access<F>,
  accessor_member<F, P>>;

/// \endcond

/// A sequence of accessors obtained from a \c launch::mapping.
/// Pass a \c multi_reference or \c mapping to a task that accepts one.
/// <!-- This gets the short name since users must
///      declare parameters with it. -->
/// \tparam A an \c accessor, \c mutator, or \c topology_accessor
///   specialization
template<class A>
struct multi : detail::multi_buffer<A>, send_tag, bind_tag {
  multi(Color n, const A & a)
    : vp(std::make_shared<std::vector<round>>(n, round{{}, a})) {}
  multi(const multi &) = default; // implement move as copy

  Color depth() const {
    return vp->size();
  }

  /// Get the components for each color.
  /// \code for(auto [c,a] : m.components()) \endcode
  /// \return a range of color-accessor pairs
  auto components() const {
    return util::transform_view(
      util::span(*vp), [](const round & r) -> std::pair<Color, const A &> {
        return {r.row, r.a};
      });
  }
  // Usable on caller side:
  auto accessors() {
    return xform(*vp);
  }
  auto accessors() const {
    return xform(std::as_const(*vp));
  }

  template<class F>
  void send(F && f) {
    auto & v = *vp;
    Color i = 0;
    for(auto & [c, a] : v) {
      f(c.get_base(), [&](auto & r) {
        flog_assert(r.map().depth() == Color(v.size()),
          "launch map has depth " << r.map().depth() << ", not " << v.size());
        return r.map().claims(i);
      });
      f(a, [&](auto & r) -> decltype(auto) { return r.data(i); });
      ++i;
    }
    // no-op on caller side:
    v.erase(std::remove_if(v.begin(),
              v.end(),
              [](const round & r) {
                return !r.row.get_base().get_base().span().empty() &&
                       r.row == borrow::nil;
              }),
      v.end());
  }

private:
  struct round {
    accessor_member<topo::claims::field, privilege_pack<ro>> row;
    A a;
  };
  template<class V>
  static auto xform(V & v) {
    return util::transform_view(
      util::span(v), [](auto & r) -> auto & { return r.a; });
  }

  // Avoid losing contents when moved into a user parameter.
  // We could use TaskBuffer for the purpose, but not on the caller side.
  std::shared_ptr<std::vector<round>> vp;
};

/// \}
} // namespace data

template<data::layout L, class T, Privileges P>
struct exec::detail::task_param<data::accessor<L, T, P>> {
  template<class Topo, typename Topo::index_space S>
  static auto replace(const data::field_reference<T, L, Topo, S> & r) {
    return data::accessor<L, T, P>(r.fid());
  }
};
template<data::layout L, class T, Privileges P>
struct exec::detail::task_param<data::mutator<L, T, P>> {
  template<class Topo, typename Topo::index_space S>
  static auto replace(const data::field_reference<T, L, Topo, S> & r) {
    return data::mutator<L, T, P>(r.fid());
  }
};
template<class T, Privileges P>
struct exec::detail::task_param<data::mutator<data::ragged, T, P>> {
  using type = data::mutator<data::ragged, T, P>;
  template<class Topo, typename Topo::index_space S>
  static type replace(
    const data::field_reference<T, data::ragged, Topo, S> & r) {
    flog_assert(
      !exec::is_tracing(), "ragged mutators cannot be used while tracing");
    return {exec::replace_argument<typename type::base_type::base_type>(
              r.template cast<data::raw>()),
      r.get_elements().template get_partition<topo::elements>().growth};
  }
};
template<class T, Privileges P>
struct exec::detail::task_param<data::mutator<data::sparse, T, P>> {
  using type = data::mutator<data::sparse, T, P>;
  template<class Topo, typename Topo::index_space S>
  static type replace(
    const data::field_reference<T, data::sparse, Topo, S> & r) {
    return exec::replace_argument<typename type::base_type>(
      r.template cast<data::ragged,
        typename field<T, data::sparse>::base_type::value_type>());
  }
};
template<class R, typename T>
struct exec::detail::task_param<data::reduction_accessor<R, T>> {
  template<class Topo, typename Topo::index_space S>
  static auto replace(
    const data::field_reference<T, data::dense, Topo, S> & r) {
    return data::reduction_accessor<R, T>(r.fid());
  }
};
template<class A>
struct exec::detail::task_param<data::multi<A>> {
  using type = data::multi<A>;
  template<class T, data::layout L, class Topo, typename Topo::index_space S>
  static type replace(const data::multi_reference<T, L, Topo, S> & r) {
    return mk(r);
  }
  template<class P>
  static type replace(data::launch::mapping<P> & m) {
    return mk(m);
  }

private:
  template<class T>
  static type mk(T & t) {
    return {t.map().depth(), exec::replace_argument<A>(t.data(0))};
  }
};

} // namespace flecsi

#endif
