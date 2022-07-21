// Copyright (c) 2016, Triad National Security, LLC
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

  constexpr span() noexcept : span(nullptr, nullptr) {}
  constexpr span(pointer p, size_type sz) : span(p, p + sz) {}
  constexpr span(pointer p, pointer q) : p(p), q(q) {}
  template<std::size_t N>
  constexpr span(element_type (&a)[N]) : span(a, N) {}
  /// \warning Destroying \a C leaves this object dangling if it owns its
  ///   elements.  This implementation does not check for "borrowing".
  template<class C,
    class = std::enable_if_t<std::is_convertible_v<
      std::remove_pointer_t<decltype(void(std::size(std::declval<C &&>())),
        std::data(std::declval<C &&>()))> (*)[],
      T (*)[]>>>
  constexpr span(C && c) : span(std::data(c), std::size(c)) {}
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

/// \cond core

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
/// \endcond

/// A small, approximate subset of mdspan as proposed for C++23.
/// \tparam D dimension
template<class T, unsigned short D>
struct mdspan : detail::mdbase<T, D> {
  using mdspan::mdbase::mdbase;
  using typename mdspan::mdbase::size_type;
  friend struct mdspan<T, D + 1>;

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
template<class I>
struct iota_view {
  struct iterator {
    using value_type = I;
    using reference = I;
    using pointer = void;
    using difference_type = I;
    using iterator_category = std::input_iterator_tag;

    constexpr iterator(I i = I()) : i(i) {}

    constexpr I operator*() const {
      return i;
    }
    constexpr I operator[](difference_type n) {
      return i + n;
    }

    constexpr iterator & operator++() {
      ++i;
      return *this;
    }
    constexpr iterator operator++(int) {
      const iterator ret = *this;
      ++*this;
      return ret;
    }
    constexpr iterator & operator--() {
      --i;
      return *this;
    }
    constexpr iterator operator--(int) {
      const iterator ret = *this;
      --*this;
      return ret;
    }
    constexpr iterator & operator+=(difference_type n) {
      i += n;
      return *this;
    }
    friend constexpr iterator operator+(difference_type n, iterator i) {
      i += n;
      return i;
    }
    constexpr iterator operator+(difference_type n) const {
      return n + *this;
    }
    constexpr iterator & operator-=(difference_type n) {
      i -= n;
      return *this;
    }
    constexpr iterator operator-(difference_type n) const {
      iterator ret = *this;
      ret -= n;
      return ret;
    }
    constexpr difference_type operator-(const iterator & r) const {
      return i - r.i;
    }

    constexpr bool operator==(const iterator & r) const noexcept {
      return i == r.i;
    }
    constexpr bool operator!=(const iterator & r) const noexcept {
      return !(*this == r);
    }
    constexpr bool operator<(const iterator & r) const noexcept {
      return i < r.i;
    }
    constexpr bool operator>(const iterator & r) const noexcept {
      return r < *this;
    }
    constexpr bool operator<=(const iterator & r) const noexcept {
      return !(*this > r);
    }
    constexpr bool operator>=(const iterator & r) const noexcept {
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

  index_iterator() noexcept : index_iterator(nullptr, 0) {}
  index_iterator(C * p, std::size_t i) : c(p), i(i) {}

  decltype(auto) operator*() const {
    return (*this)[0];
  }
  auto operator->() const {
    return &**this;
  }

  decltype(auto) operator[](difference_type n) const {
    return (*c)[i + n];
  }

  index_iterator & operator++() {
    ++i;
    return *this;
  }
  index_iterator operator++(int) {
    index_iterator ret = *this;
    ++*this;
    return ret;
  }
  index_iterator & operator--() {
    --i;
    return *this;
  }
  index_iterator operator--(int) {
    index_iterator ret = *this;
    --*this;
    return ret;
  }

  index_iterator & operator+=(difference_type n) {
    i += n;
    return *this;
  }
  friend index_iterator operator+(difference_type n, index_iterator i) {
    return i += n;
  }
  index_iterator operator+(difference_type n) const {
    return n + *this;
  }
  index_iterator & operator-=(difference_type n) {
    i -= n;
    return *this;
  }
  friend index_iterator operator-(difference_type n, index_iterator i) {
    return i -= n;
  }
  index_iterator operator-(difference_type n) const {
    return n - *this;
  }
  difference_type operator-(const index_iterator & o) const {
    return i - o.i;
  }

  bool operator==(const index_iterator & o) const {
    return i == o.i;
  }
  bool operator!=(const index_iterator & o) const {
    return i != o.i;
  }
  bool operator<(const index_iterator & o) const {
    return i < o.i;
  }
  bool operator<=(const index_iterator & o) const {
    return i <= o.i;
  }
  bool operator>(const index_iterator & o) const {
    return i > o.i;
  }
  bool operator>=(const index_iterator & o) const {
    return i >= o.i;
  }

  std::size_t index() const noexcept {
    return i;
  }
};

template<class D> // CRTP, but D might be const
struct with_index_iterator {
  using iterator = index_iterator<D>;

  iterator begin() const noexcept {
    return {derived(), 0};
  }
  iterator end() const noexcept {
    const auto * p = derived();
    return {p, p->size()};
  }

private:
  D * derived() const noexcept {
    return static_cast<D *>(this);
  }
};

/// A very simple emulation of std::ranges::transform_view from C++20.
template<class C, class F>
struct transform_view {
private:
  // create an iterator type from the passed-in container
  using base_iterator = decltype(std::begin(std::declval<C &>()));
  C c;
  F f;

public:
  struct iterator {
  private:
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

    constexpr iterator & operator++() {
      ++p;
      return *this;
    }
    constexpr iterator operator++(int) {
      const iterator ret = *this;
      ++*this;
      return ret;
    }
    constexpr iterator & operator--() {
      --p;
      return *this;
    }
    constexpr iterator operator--(int) {
      const iterator ret = *this;
      --*this;
      return ret;
    }
    constexpr iterator & operator+=(difference_type n) {
      p += n;
      return *this;
    }
    friend constexpr iterator operator+(difference_type n, iterator i) {
      i += n;
      return i;
    }
    constexpr iterator operator+(difference_type n) const {
      return n + *this;
    }
    constexpr iterator & operator-=(difference_type n) {
      p -= n;
      return *this;
    }
    constexpr iterator operator-(difference_type n) const {
      iterator ret = *this;
      ret -= n;
      return ret;
    }
    constexpr difference_type operator-(const iterator & i) const {
      return p - i.p;
    }

    constexpr bool operator==(const iterator & i) const noexcept {
      return p == i.p;
    }
    constexpr bool operator!=(const iterator & i) const noexcept {
      return !(*this == i);
    }
    constexpr bool operator<(const iterator & i) const noexcept {
      return p < i.p;
    }
    constexpr bool operator>(const iterator & i) const noexcept {
      return i < *this;
    }
    constexpr bool operator<=(const iterator & i) const noexcept {
      return !(*this > i);
    }
    constexpr bool operator>=(const iterator & i) const noexcept {
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
  constexpr iterator begin() const noexcept {
    return {std::begin(c), &f};
  }
  FLECSI_INLINE_TARGET
  constexpr iterator end() const noexcept {
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
  constexpr decltype(auto) front() const {
    return *begin();
  }

  FLECSI_INLINE_TARGET
  constexpr decltype(auto) back() const {
    return *--end();
  }

  FLECSI_INLINE_TARGET
  constexpr decltype(auto) operator[](
    typename std::iterator_traits<base_iterator>::difference_type i) const {
    return begin()[i];
  }

}; // struct transform_view

/// \endcond

/// \}
} // namespace util
} // namespace flecsi

#endif
