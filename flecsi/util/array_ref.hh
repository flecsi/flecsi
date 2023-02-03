// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_ARRAY_REF_HH
#define FLECSI_UTIL_ARRAY_REF_HH

#include <array>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

#include "flecsi/util/common.hh"
#include "flecsi/util/target.hh"

namespace flecsi {
namespace util {
/// \addtogroup ranges Ranges
/// Range and iterator tools, mostly backported from the standard library.
/// \ingroup utils
/// \{

/// A workalike for std::span from C++20 (only dynamic-extent, without ranges
/// support).
/// This class is supported for GPU execution.
template<class T>
struct span {
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;

  using iterator = pointer; // implementation-defined
  using reverse_iterator = std::reverse_iterator<iterator>;

  FLECSI_INLINE_TARGET constexpr span() noexcept : span(nullptr, nullptr) {}
  FLECSI_INLINE_TARGET constexpr span(pointer p, size_type sz)
    : span(p, p + sz) {}
  FLECSI_INLINE_TARGET constexpr span(pointer p, pointer q) : p(p), q(q) {}
  template<std::size_t N>
  FLECSI_INLINE_TARGET constexpr span(element_type (&a)[N]) : span(a, N) {}
  /// \warning Destroying \a C leaves this object dangling if it owns its
  ///   elements.  This implementation does not check for "borrowing".
  template<class C,
    class = std::enable_if_t<std::is_convertible_v<
      std::remove_pointer_t<decltype(void(std::size(std::declval<C &&>())),
        std::data(std::declval<C &&>()))> (*)[],
      T (*)[]>>>
  FLECSI_INLINE_TARGET constexpr span(C && c)
    : span(std::data(c), std::size(c)) {}
  FLECSI_INLINE_TARGET
  constexpr iterator begin() const noexcept {
    return p;
  }

  FLECSI_INLINE_TARGET
  constexpr iterator end() const noexcept {
    return q;
  }

  FLECSI_INLINE_TARGET
  constexpr reverse_iterator rbegin() const noexcept {
    return reverse_iterator(end());
  }

  FLECSI_INLINE_TARGET
  constexpr reverse_iterator rend() const noexcept {
    return reverse_iterator(begin());
  }

  FLECSI_INLINE_TARGET
  constexpr reference front() const {
    return *begin();
  }

  FLECSI_INLINE_TARGET
  constexpr reference back() const {
    return end()[-1];
  }

  FLECSI_INLINE_TARGET
  constexpr reference operator[](size_type i) const {
    return begin()[i];
  }

  FLECSI_INLINE_TARGET
  constexpr pointer data() const noexcept {
    return begin();
  }

  // FIXME: Spurious overflow for extremely large ranges
  FLECSI_INLINE_TARGET
  constexpr size_type size() const noexcept {
    return end() - begin();
  }

  FLECSI_INLINE_TARGET
  constexpr size_type size_bytes() const noexcept {
    return sizeof(element_type) * size();
  }

  FLECSI_INLINE_TARGET
  constexpr bool empty() const noexcept {
    return begin() == end();
  }

  FLECSI_INLINE_TARGET
  constexpr span first(size_type n) const {
    return {begin(), n};
  }

  FLECSI_INLINE_TARGET
  constexpr span last(size_type n) const {
    return {end() - n, n};
  }
  FLECSI_INLINE_TARGET
  constexpr span subspan(size_type i, size_type n = -1) const {
    return {begin() + i, n == size_type(-1) ? size() - i : n};
  }

private:
  pointer p, q;
};

template<class C>
span(C &) -> span<typename C::value_type>;
template<class C>
span(const C &) -> span<const typename C::value_type>;

/// Copy a span into a std::vector.
template<class T>
auto
to_vector(span<T> s) {
  // Work around GCC<8 having no deduction guide for vector:
  return std::vector<typename span<T>::value_type>(s.begin(), s.end());
}

// In these classes, the dimensions of a multidimensional array are labeled
// with integers starting with 0 for the least-significant index, which
// chooses among elements that are adjacent in memory.  Note that C arrays are
// declared and used in the other order:
//   int x[/* length 2 */][/* length 1 */][/* length 0 */];
// and that std::extent<decltype(x)/*,0*/> is length #2.
// (It's undefined to use these views with a truly multidimensional array.)

namespace detail {
/// A multi-dimensional view of an array.
/// This class is supported for GPU execution.
template<class T, Dimension D>
struct mdbase {
  static_assert(D > 0);
  using size_type = std::size_t;

  /// Construct a view of a one-dimensional array.
  /// Here, `x` and `y` are analogous:\code
  /// int x[2][3][4],y0[2*3*4];
  /// mdbase<int,3> y(y0,{4,3,2});
  /// \endcode
  /// \param p pointer to first element of (sub)array
  /// \param sz sizes, least significant first
  constexpr mdbase(T * p, std::array<size_type, D> sz) noexcept
    : mdbase(p, [&sz] {
        for(int d = 1; d < D; ++d) // premultiply to convert to strides
          sz[d] *= sz[d - 1];
        return sz.data();
      }()) {}

  /// Get one size of the view.
  /// \param i dimension (0 for least significant)
  constexpr size_type length(Dimension i) const noexcept {
    return step(i + 1) / step(i);
  }

protected:
  // The plain pointer can copy most data from a higher-dimensional object.
  FLECSI_INLINE_TARGET
  constexpr mdbase(T * p, const size_type * s) noexcept : p(p), strides() {
    for(int d = 0; d < D; d++)
      strides[d] = s[d];
  }

  constexpr size_type step(unsigned short i) const noexcept {
    assert(i <= D);
    return i == 0 ? 1 : strides[i - 1];
  }

  T * p;
  size_type strides[D]; // last element only for bounds checking
};
} // namespace detail

/// A variation of \c mdspan with reversed indices (distinguished by `()`).
template<class T, Dimension D>
struct mdcolex : detail::mdbase<T, D> {
  using mdcolex::mdbase::mdbase;
  using typename mdcolex::mdbase::size_type;

  /// Select an element of the view.
  /// \param inds indices (each smaller than `length(0)`, `length(1)`,
  ///   &hellip;)
  /// \return T&
  template<class... I>
  constexpr decltype(auto) operator()(I... inds) const noexcept {
    static_assert(sizeof...(inds) == D);
    Dimension d = 0;
    size_type i = 0;
    ((assert(size_type(inds) < this->length(d)),
       i += size_type(inds) * this->step(d++)),
      ...); // guarantee evaluation order
    return this->p[i];
  }
};

/// A small, approximate subset of mdspan as proposed for C++23.
/// This class is supported for GPU execution.
/// \tparam D dimension
template<class T, unsigned short D>
struct mdspan : detail::mdbase<T, D> {
  using mdspan::mdbase::mdbase;
  using typename mdspan::mdbase::size_type;
  friend mdspan<T, D + 1>;

  /// Select a subset of the view.
  /// \param i index (must be smaller than `length(D-1)`)
  /// \return `mdspan<T,D-1>` or `T&` if `D` is 1
  FLECSI_INLINE_TARGET
  constexpr decltype(auto) operator[](size_type i) const noexcept {
    assert(i < this->length(D - 1));
    const auto q = this->p + i * this->step(D - 1);
    if constexpr(D > 1)
      return mdspan<T, D - 1>(q, this->strides);
    else
      return *q;
  }
};

/// \cond core

/// A very simple emulation of std::ranges::iota_view from C++20.
/// This class is supported for GPU execution.
template<class I>
struct iota_view {
  static_assert(!std::is_const_v<I>, "integer type must not be qualified");
  struct iterator {
    using value_type = I;
    using reference = I;
    using pointer = void;
    using difference_type = I;
    using iterator_category = std::input_iterator_tag;

    FLECSI_INLINE_TARGET constexpr iterator(I i = I()) : i(i) {}

    FLECSI_INLINE_TARGET constexpr I operator*() const {
      return i;
    }
    FLECSI_INLINE_TARGET constexpr I operator[](difference_type n) {
      return i + n;
    }

    FLECSI_INLINE_TARGET constexpr iterator & operator++() {
      ++i;
      return *this;
    }
    FLECSI_INLINE_TARGET constexpr iterator operator++(int) {
      const iterator ret = *this;
      ++*this;
      return ret;
    }
    FLECSI_INLINE_TARGET constexpr iterator & operator--() {
      --i;
      return *this;
    }
    FLECSI_INLINE_TARGET constexpr iterator operator--(int) {
      const iterator ret = *this;
      --*this;
      return ret;
    }
    FLECSI_INLINE_TARGET constexpr iterator & operator+=(difference_type n) {
      i += n;
      return *this;
    }
    FLECSI_INLINE_TARGET friend constexpr iterator operator+(difference_type n,
      iterator i) {
      i += n;
      return i;
    }
    FLECSI_INLINE_TARGET constexpr iterator operator+(difference_type n) const {
      return n + *this;
    }
    FLECSI_INLINE_TARGET constexpr iterator & operator-=(difference_type n) {
      i -= n;
      return *this;
    }
    FLECSI_INLINE_TARGET constexpr iterator operator-(difference_type n) const {
      iterator ret = *this;
      ret -= n;
      return ret;
    }
    FLECSI_INLINE_TARGET constexpr difference_type operator-(
      const iterator & r) const {
      return i - r.i;
    }

    FLECSI_INLINE_TARGET constexpr bool operator==(
      const iterator & r) const noexcept {
      return i == r.i;
    }
    FLECSI_INLINE_TARGET constexpr bool operator!=(
      const iterator & r) const noexcept {
      return !(*this == r);
    }
    FLECSI_INLINE_TARGET constexpr bool operator<(
      const iterator & r) const noexcept {
      return i < r.i;
    }
    FLECSI_INLINE_TARGET constexpr bool operator>(
      const iterator & r) const noexcept {
      return r < *this;
    }
    FLECSI_INLINE_TARGET constexpr bool operator<=(
      const iterator & r) const noexcept {
      return !(*this > r);
    }
    FLECSI_INLINE_TARGET constexpr bool operator>=(
      const iterator & r) const noexcept {
      return !(*this < r);
    }

  private:
    I i;
  };

  iota_view() = default;
  constexpr iota_view(I b, I e) : b(b), e(e) {}
  FLECSI_INLINE_TARGET
  constexpr iterator begin() const noexcept {
    return b;
  }
  FLECSI_INLINE_TARGET
  constexpr iterator end() const noexcept {
    return e;
  }
  FLECSI_INLINE_TARGET
  constexpr bool empty() const {
    return b == e;
  }

  FLECSI_INLINE_TARGET
  constexpr explicit operator bool() const {
    return !empty();
  }

  FLECSI_INLINE_TARGET
  constexpr auto size() const {
    return e - b;
  }

  FLECSI_INLINE_TARGET
  constexpr decltype(auto) front() const {
    return *begin();
  }

  FLECSI_INLINE_TARGET
  constexpr decltype(auto) back() const {
    return *--end();
  }

  FLECSI_INLINE_TARGET
  constexpr decltype(auto) operator[](I i) const {
    return begin()[i];
  }

private:
  iterator b, e;
};

// A generic iterator implementation in terms of subscripting.
template<class C>
struct index_iterator {
private:
  C * c;
  std::size_t i;

public:
  using reference = decltype((*c)[i]);
  using value_type = std::remove_reference_t<reference>;
  using pointer =
    std::conditional_t<std::is_reference_v<reference>, value_type *, void>;
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::random_access_iterator_tag;

  FLECSI_INLINE_TARGET index_iterator() noexcept : index_iterator(nullptr, 0) {}
  FLECSI_INLINE_TARGET index_iterator(C * p, std::size_t i) : c(p), i(i) {}

  FLECSI_INLINE_TARGET decltype(auto) operator*() const {
    return (*this)[0];
  }
  FLECSI_INLINE_TARGET auto operator->() const {
    return &**this;
  }

  FLECSI_INLINE_TARGET decltype(auto) operator[](difference_type n) const {
    return (*c)[i + n];
  }

  FLECSI_INLINE_TARGET index_iterator & operator++() {
    ++i;
    return *this;
  }
  FLECSI_INLINE_TARGET index_iterator operator++(int) {
    index_iterator ret = *this;
    ++*this;
    return ret;
  }
  FLECSI_INLINE_TARGET index_iterator & operator--() {
    --i;
    return *this;
  }
  FLECSI_INLINE_TARGET index_iterator operator--(int) {
    index_iterator ret = *this;
    --*this;
    return ret;
  }

  FLECSI_INLINE_TARGET index_iterator & operator+=(difference_type n) {
    i += n;
    return *this;
  }
  FLECSI_INLINE_TARGET friend index_iterator operator+(difference_type n,
    index_iterator i) {
    return i += n;
  }
  FLECSI_INLINE_TARGET index_iterator operator+(difference_type n) const {
    return n + *this;
  }
  FLECSI_INLINE_TARGET index_iterator & operator-=(difference_type n) {
    i -= n;
    return *this;
  }
  FLECSI_INLINE_TARGET friend index_iterator operator-(difference_type n,
    index_iterator i) {
    return i -= n;
  }
  FLECSI_INLINE_TARGET index_iterator operator-(difference_type n) const {
    return n - *this;
  }
  FLECSI_INLINE_TARGET difference_type operator-(
    const index_iterator & o) const {
    return i - o.i;
  }

  FLECSI_INLINE_TARGET bool operator==(const index_iterator & o) const {
    return i == o.i;
  }
  FLECSI_INLINE_TARGET bool operator!=(const index_iterator & o) const {
    return i != o.i;
  }
  FLECSI_INLINE_TARGET bool operator<(const index_iterator & o) const {
    return i < o.i;
  }
  FLECSI_INLINE_TARGET bool operator<=(const index_iterator & o) const {
    return i <= o.i;
  }
  FLECSI_INLINE_TARGET bool operator>(const index_iterator & o) const {
    return i > o.i;
  }
  FLECSI_INLINE_TARGET bool operator>=(const index_iterator & o) const {
    return i >= o.i;
  }

  std::size_t index() const noexcept {
    return i;
  }
};

template<class D> // CRTP, but D might be const
struct with_index_iterator {
  using iterator = index_iterator<D>;

  FLECSI_INLINE_TARGET
  iterator begin() const noexcept {
    return {derived(), 0};
  }

  FLECSI_INLINE_TARGET
  iterator end() const noexcept {
    const auto * p = derived();
    return {p, p->size()};
  }

private:
  FLECSI_INLINE_TARGET
  D * derived() const noexcept {
    return static_cast<D *>(this);
  }
};

/// A very simple emulation of std::ranges::transform_view from C++20.
/// This class is supported for GPU execution.
template<class C, class F>
struct transform_view {
private:
  C c;
  F f;

public:
  template<bool Const>
  struct iterator {
  private:
    using base_iterator = decltype(std::begin(
      std::declval<std::conditional_t<Const, const C, C> &>()));
    using traits = std::iterator_traits<base_iterator>;

  public:
    using difference_type = typename traits::difference_type;
    // TODO: notice a reference return from F and upgrade iterator_category
    using reference = decltype(std::declval<const F &>()(
      std::declval<typename traits::reference>()));
    using value_type = std::decay_t<reference>;
    using pointer = void;
    // We provide all the operators, but we don't assume a real reference:
    using iterator_category = std::input_iterator_tag;

    constexpr iterator() noexcept
      : iterator({}, nullptr) {} // null F won't be used
    constexpr iterator(base_iterator p, const F * f) noexcept : p(p), f(f) {}

    FLECSI_INLINE_TARGET constexpr iterator & operator++() {
      ++p;
      return *this;
    }
    FLECSI_INLINE_TARGET constexpr iterator operator++(int) {
      const iterator ret = *this;
      ++*this;
      return ret;
    }
    FLECSI_INLINE_TARGET constexpr iterator & operator--() {
      --p;
      return *this;
    }
    FLECSI_INLINE_TARGET constexpr iterator operator--(int) {
      const iterator ret = *this;
      --*this;
      return ret;
    }
    FLECSI_INLINE_TARGET constexpr iterator & operator+=(difference_type n) {
      p += n;
      return *this;
    }
    FLECSI_INLINE_TARGET friend constexpr iterator operator+(difference_type n,
      iterator i) {
      i += n;
      return i;
    }
    FLECSI_INLINE_TARGET constexpr iterator operator+(difference_type n) const {
      return n + *this;
    }
    FLECSI_INLINE_TARGET constexpr iterator & operator-=(difference_type n) {
      p -= n;
      return *this;
    }
    FLECSI_INLINE_TARGET constexpr iterator operator-(difference_type n) const {
      iterator ret = *this;
      ret -= n;
      return ret;
    }
    FLECSI_INLINE_TARGET constexpr difference_type operator-(
      const iterator & i) const {
      return p - i.p;
    }

    FLECSI_INLINE_TARGET constexpr bool operator==(
      const iterator & i) const noexcept {
      return p == i.p;
    }
    FLECSI_INLINE_TARGET constexpr bool operator!=(
      const iterator & i) const noexcept {
      return !(*this == i);
    }
    FLECSI_INLINE_TARGET constexpr bool operator<(
      const iterator & i) const noexcept {
      return p < i.p;
    }
    FLECSI_INLINE_TARGET constexpr bool operator>(
      const iterator & i) const noexcept {
      return i < *this;
    }
    FLECSI_INLINE_TARGET constexpr bool operator<=(
      const iterator & i) const noexcept {
      return !(*this > i);
    }
    FLECSI_INLINE_TARGET constexpr bool operator>=(
      const iterator & i) const noexcept {
      return !(*this < i);
    }

    FLECSI_INLINE_TARGET
    constexpr reference operator*() const {
      return (*f)(*p);
    }
    // operator-> makes sense only for a true 'reference'
    FLECSI_INLINE_TARGET
    constexpr reference operator[](difference_type n) const {
      return *(*this + n);
    }

  private:
    base_iterator p;
    const F * f;
  }; // struct iterator

  /// Wrap a container.
  constexpr transform_view(C c, F f = {}) : c(std::move(c)), f(std::move(f)) {}

  FLECSI_INLINE_TARGET
  constexpr iterator<false> begin() noexcept {
    return {std::begin(c), &f};
  }
  FLECSI_INLINE_TARGET
  constexpr iterator<true> begin() const noexcept {
    return {std::begin(c), &f};
  }
  FLECSI_INLINE_TARGET
  constexpr iterator<false> end() noexcept {
    return {std::end(c), &f};
  }
  FLECSI_INLINE_TARGET
  constexpr iterator<true> end() const noexcept {
    return {std::end(c), &f};
  }

  FLECSI_INLINE_TARGET
  constexpr bool empty() const {
    return std::begin(c) == std::end(c);
  }
  FLECSI_INLINE_TARGET
  constexpr explicit operator bool() const {
    return !empty();
  }

  FLECSI_INLINE_TARGET
  constexpr auto size() const {
    return std::distance(std::begin(c), std::end(c));
  }

  FLECSI_INLINE_TARGET
  constexpr decltype(auto) front() {
    return *begin();
  }
  FLECSI_INLINE_TARGET
  constexpr decltype(auto) front() const {
    return *begin();
  }

  FLECSI_INLINE_TARGET
  constexpr decltype(auto) back() {
    return *--end();
  }
  FLECSI_INLINE_TARGET
  constexpr decltype(auto) back() const {
    return *--end();
  }

  FLECSI_INLINE_TARGET
  constexpr decltype(auto) operator[](
    typename iterator<false>::difference_type i) {
    return begin()[i];
  }
  FLECSI_INLINE_TARGET
  constexpr decltype(auto) operator[](
    typename iterator<true>::difference_type i) const {
    return begin()[i];
  }

}; // struct transform_view

/// A simple emulation of \c std::stride_view from C++23 for random-access
/// underlying ranges.
/// \param o initial offset (not part of C++23)
template<class R>
constexpr auto
stride_view(R && r,
  typename std::iterator_traits<decltype(std::begin(
    std::declval<R>()))>::difference_type n,
  decltype(n) o = 0) {
  using I = std::make_unsigned_t<decltype(n)>;
  const I sz = ceil_div<I>(std::size(r) - o, n); // before moving
  return transform_view(
    iota_view<I>(0, sz), [r = std::forward<R>(r), n, o](I i) -> decltype(auto) {
      return r[i * n + o];
    });
}

/// \endcond

/// A view of part of a range.  Analogous to a combination of
/// This class is supported for GPU execution.
/// \c std::take_view and \c std::drop_view from C++20.
template<class R>
struct substring_view {
  using iterator = decltype(std::begin(std::declval<R>()));
  using difference_type =
    typename std::iterator_traits<iterator>::difference_type;

  /// Select a substring of a range.
  /// \param r input (stored in this object)
  /// \param i starting index
  /// \param n number of elements (must be in bounds)
  constexpr substring_view(R r, difference_type i, difference_type n)
    : alive(std::move(r)), b(std::next(std::begin(alive), i)), n(n) {}

  FLECSI_INLINE_TARGET
  constexpr iterator begin() const {
    return b;
  }
  FLECSI_INLINE_TARGET
  constexpr iterator end() const {
    return std::next(b, n);
  }

  FLECSI_INLINE_TARGET
  constexpr bool empty() const {
    return !n;
  }
  FLECSI_INLINE_TARGET
  constexpr explicit operator bool() const {
    return n;
  }

  FLECSI_INLINE_TARGET
  constexpr std::make_unsigned_t<difference_type> size() const {
    return n;
  }

  FLECSI_INLINE_TARGET
  constexpr decltype(auto) front() const {
    return *b;
  }
  FLECSI_INLINE_TARGET
  constexpr decltype(auto) back() const {
    return b[n - 1];
  }

  FLECSI_INLINE_TARGET
  constexpr decltype(auto) operator[](difference_type i) const {
    return b[i];
  }

private:
  R alive;
  iterator b;
  difference_type n;
};

/// \}
} // namespace util
} // namespace flecsi

#endif
