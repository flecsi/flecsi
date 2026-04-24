// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_COMMON_HH
#define FLECSI_UTIL_COMMON_HH

#include "flecsi/config.hh"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstdint> // for ID types
#include <cstdio>
#include <ios>
#include <map>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace flecsi {
/// Type for spatial dimensions and counts thereof.  \ns.
/// \ingroup utils
using Dimension = unsigned short;

namespace util {
/// \addtogroup utils
/// \{

/// Local (color-specific) topology entity ID type.
/// Often provided in a index-space-specific convertible wrapper.
using id = FLECSI_ID_TYPE;
static_assert(std::is_unsigned_v<id>,
  "topology entity ID type must be unsigned");

/// Global topology entity ID type.
using gid = FLECSI_GID_TYPE;

/// Interpret a type as itself in functional contexts.
struct identity {
  template<class T>
  T && operator()(T && x) {
    return std::forward<T>(x);
  }
};

/// \cond core

// A move-only subset of std::any.
struct any_base {
  virtual ~any_base() = default;
  virtual void * get(const std::type_info &) = 0;
};

template<class T>
struct any_impl : any_base {
  any_impl(T t) : t(std::move(t)) {}
  void * get(const std::type_info & i) override {
    if(i != typeid(T))
      throw std::bad_cast();
    return &t;
  }
  T t;
};

struct any {
  template<class T>
  std::decay_t<T> & emplace(T && t) {
    auto * const q = new any_impl<std::decay_t<T>>(std::forward<T>(t));
    p.reset(q);
    return q->t;
  }

  explicit operator bool() const {
    return !!p;
  }
  template<class T>
  T & get() {
    return *static_cast<T *>(p->get(typeid(T)));
  }
  template<class T>
  const T & get() const {
    return const_cast<any &>(*this).get<T>();
  }
  template<class T>
  T && get() && {
    return std::move(get<T>());
  }

private:
  std::unique_ptr<any_base> p;
};

template<bool Const, class T>
using maybe_const = std::conditional_t<Const, const T, T>;

// Defer a functor call until a conversion to its return type is needed.
template<class F>
struct convert {
  F f;
  operator decltype(std::declval<const F &>()())() const & {
    return f();
  }
  operator decltype(std::declval<F &>()())() & {
    return f();
  }
  operator decltype(std::declval<F>()())() && {
    return std::move(f)();
  }
};

template<class T>
constexpr std::enable_if_t<std::is_unsigned_v<T>, T>
ceil_div(T a, T b) {
  return a / b + !!(a % b); // avoids overflow in (a+(b-1))/b
}

//! P.O.D.
template<typename T>
constexpr T
square(const T & a) {
  return a * a;
}

/// A counter with a maximum.
template<auto M>
struct counter {
  using type = decltype(M);

  constexpr explicit counter(type l) : last(l) {}

  [[nodiscard]] const type & operator()() {
    assert(last < M && "counter overflow");
    return ++last;
  }

private:
  type last;
};

template<class T>
struct ref_count {
  explicit ref_count(T n) : c(n) {}
  ~ref_count() {
    assert(!*this && "abandoned reference count");
  }

  void operator+=(T n) {
    c.fetch_add(n, std::memory_order_relaxed);
  }
  void operator++() {
    *this += 1;
  }
  // Note the reversed sense of these two.
  [[nodiscard]] bool operator--() {
    return c.fetch_sub(1, std::memory_order_release) == 1;
  }
  explicit operator bool() const {
    return c.load(std::memory_order_acquire);
  }

private:
  std::atomic<T> c;
};

/// Sort a std::vector and remove duplicates.
template<typename T>
void
force_unique(std::vector<T> & v) {
  std::sort(v.begin(), v.end());
  auto first = v.begin();
  auto last = std::unique(first, v.end());
  v.erase(last, v.end());
}

/// Apply force_unique to each element of a std::map. Note that force_unique
/// is currently only implemented for std::vector.
template<typename K, typename T>
void
unique_each(std::map<K, T> & m) {
  for(auto & v : m)
    force_unique(v.second);
}

/// Apply force_unique to each element of a std::vector. Note that force_unique
/// is currently only implemented for std::vector.
template<typename T>
void
unique_each(std::vector<T> & vv) {
  for(auto & v : vv)
    force_unique(v);
}

struct FILE {
  FILE(const char * n, const char * m) : f(std::fopen(n, m)) {
    if(!f)
      throw std::ios::failure(
        "cannot open file", {errno, std::system_category()});
  }
  FILE(FILE && f) noexcept : f(std::exchange(f.f, {})) {}
  ~FILE() {
    if(f)
      std::fclose(f); // NB: error lost
  }
  FILE & operator=(FILE src) & noexcept {
    std::swap(f, src.f);
    return *this;
  }

  operator ::FILE *() const noexcept {
    return f;
  }

private:
  ::FILE * f;
};

/// \endcond
/// \}
} // namespace util
} // namespace flecsi

#endif
