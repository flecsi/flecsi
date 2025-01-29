// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_LAUNCH_HH
#define FLECSI_EXEC_LAUNCH_HH

#include "flecsi/data/field.hh"
#include "flecsi/exec/task_attributes.hh"

#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant> // monostate

namespace flecsi {
namespace data {
// Types inherit from these tags to indicate their task execution semantics.

// A task parameter that needs additional initialization after the task has
// been launched.
struct bind_tag {};
// A task parameter that provides a member function send to decompose itself
// into lower-level types.  Its one argument is a backend-specific callback
// that accepts a (subsidiary) task parameter and another function to call to
// transform the corresponding task argument (used only on the caller side).
struct send_tag {};
} // namespace data

namespace exec {
/// \addtogroup execution
/// \{
namespace detail {
// We care about value category, so we want to use perfect forwarding.
// However, such a template is a better match for some arguments than any
// single non-template overload, so we use SFINAE to detect that we have
// no replacement defined for an argument.
// XREF: more specializations in accessor.hh
template<class>
struct task_param {};
// A is what the user gives us when calling execute(), P is what the user
// defined function/task expects. P may not be the same as A, for example, user
// pass a field_reference as an argument to execute() but the task expects an
// data accessor as its formal parameter. For instance, replace_argument
// replaces a field_reference with an accessor. This is done through various
// specialization of the exec::detail::task_param<> template.
template<class P, class A, class V = void> // A is a reference type
struct replace_argument {
  static_assert(!std::is_void_v<V>,
    "mismatch between task parameter and argument");
};
// Allow specialization as well as use of convert_tag:
template<class T>
struct must_convert
  : std::integral_constant<bool, std::is_base_of_v<data::convert_tag, T>> {};
template<class P, class A>
struct replace_argument<P,
  A,
  std::enable_if_t<!must_convert<std::decay_t<A>>::value>> {
  static constexpr bool special = false;
  static A replace(A a) {
    return static_cast<A>(a);
  }
};
template<class P, class A>
struct replace_argument<P,
  A,
  decltype(void(task_param<P>::replace(std::declval<A>())))> {
  static constexpr bool special = true;
  static decltype(auto) replace(A a) {
    return task_param<P>::replace(static_cast<A>(a));
  }
};

template<class T>
struct must_bind : std::integral_constant<bool,
                     std::is_base_of_v<data::bind_tag, T> ||
                       std::is_base_of_v<data::send_tag, T>> {};
template<class T>
constexpr bool must_bind_v = must_bind<T>::value;

// For each parameter-type/argument pair we have either an Index (the size of
// a required index launch, or nothing for an empty vector), std::monostate
// (for a required single
// launch), or std::nullptr_t (don't care).
using Index = std::optional<Color>;

template<class P, class A>
struct launch {
  static auto get(const A &) {
    return nullptr;
  }
};
template<class P,
  class T,
  data::layout L,
  class Topo,
  typename Topo::index_space S>
struct launch<P, data::field_reference<T, L, Topo, S>> {
  static Index get(const data::field_reference<T, L, Topo, S> & r) {
    return r.topology().colors();
  }
};
template<class P,
  class T,
  data::layout L,
  class Topo,
  typename Topo::index_space S>
struct launch<P, data::multi_reference<T, L, Topo, S>> {
  static Index get(const data::multi_reference<T, L, Topo, S> & r) {
    return r.map().colors();
  }
};

template<class T>
struct launch_combine {
  launch_combine(const T & t) : t(t) {} // for CTAD
  // TIP: fold-expression allows different types at each level
  template<class U>
  auto & operator|(const launch_combine<U> & c) const {
    if constexpr(std::is_same_v<T, std::nullptr_t>)
      return c;
    else {
      if constexpr(!std::is_same_v<U, std::nullptr_t>) {
        static_assert(std::is_same_v<T, U>, "implied launch types conflict");
        if constexpr(!std::is_same_v<T, std::monostate>) {
          if(!t)
            return c;
          if(c.t && *t != *c.t)
            flog_fatal(
              "implied launch sizes " << *t << " and " << *c.t << " conflict");
        }
      }
      return *this;
    }
  }
  const T & value() const {
    return t;
  }
  auto get() const {
    if constexpr(std::is_same_v<T, Index>)
      return t.value_or(0);
    else
      return std::monostate();
  }

private:
  T t;
};

template<bool M = false, class... PP, class... AA>
auto
launch_size(std::tuple<PP...> *, const AA &... aa) {
  return (launch_combine([] {
    // An MPI task has a known launch domain:
    if constexpr(M)
      return Index(run::context::instance().processes());
    else
      return nullptr;
  }()) | ... |
          launch_combine(launch<std::decay_t<PP>, AA>::get(aa)));
}

template<class D>
struct bind_base { // decomposes parameters only
protected:
  auto visitor() {
    return [this](auto & p, auto &&) { d().visit(p); };
  }

  template<class T>
  void visit(std::vector<T> & v) {
    for(auto & t : v)
      d().visit(t);
  }
  template<class... TT>
  void visit(std::tuple<TT...> & t) {
    std::apply(
      [&](auto &&... xx) { (d().visit(std::forward<decltype(xx)>(xx)), ...); },
      t);
  }

  // The const gives a different parameter type (avoiding Clang bug #49583)
  // and makes this a worse overload than that for send_tag.
  template<class P>
  static std::enable_if_t<!must_bind_v<P>> visit(const P &) {}

private:
  D & d() {
    return static_cast<D &>(*this);
  }
};
} // namespace detail
// Replaces certain task arguments before conversion to the parameter type.
template<class P, class T>
decltype(auto)
replace_argument(T && t) {
  return detail::replace_argument<std::decay_t<P>, T &&>::replace(
    std::forward<T>(t));
}

namespace detail {
template<class... PP, class... AA>
auto
replace_arguments(std::tuple<PP...> * /* to deduce PP */, AA &&... aa) {
  // Specify the template arguments explicitly to produce references to
  // unchanged arguments.
  return std::tuple<decltype(exec::replace_argument<PP>(std::forward<AA>(
    aa)))...>(exec::replace_argument<PP>(std::forward<AA>(aa))...);
}
} // namespace detail

// Return the number of task invocations for the given parameter tuple and
// arguments, or std::monostate() if a single launch is appropriate.
template<TaskAttributes A, class P, class... AA>
auto
launch_size(const AA &... aa) {
  return detail::launch_size<mask_to_processor_type(A) == processor::mpi>(
    static_cast<P *>(nullptr), aa...)
    .get();
}

enum class launch_type_t : size_t { single, index };

/// An explicit launch domain size.
struct launch_domain {
  Color size_;
};

/// \cond core
/// A simple version of C++20's \c bind_front.
/// \endcond
template<auto & F, class... AA>
struct partial : std::tuple<AA...> {
  using Base = typename partial::tuple;
  using Base::Base;
  // Clang 8.0.1--10.0 considers the inherited tuple::tuple() non-constexpr:
  constexpr partial() = default;
  constexpr partial(const Base & b) : Base(b) {}
  constexpr partial(Base && b) : Base(std::move(b)) {}

  template<class... TT>
  constexpr decltype(auto) operator()(TT &&... tt) const & {
    return std::apply(F,
      // we have to call static_cast over *this due to the bug in cuda10+gcc9.0
      // configuration
      std::tuple_cat(
        std::tuple<const AA &...>(static_cast<const Base &>(*this)),
        std::forward_as_tuple(std::forward<TT>(tt)...)));
  }
  template<class... TT>
  constexpr decltype(auto) operator()(TT &&... tt) && {
    return std::apply(F,
      std::tuple_cat(std::tuple<AA &&...>(static_cast<Base &&>(*this)),
        std::forward_as_tuple(std::forward<TT>(tt)...)));
  }

  static partial param; // not defined; use as "f<decltype(p.param)>"
};
/// \}
} // namespace exec

/// \addtogroup execution
/// \{

/// Partially apply a function.
/// \tparam F function to call
/// \tparam AA leading arguments
/// \return a function object
/// \note A task that accepts the result will usually be a function template:
/// \code
///   void func(/*...*/);
///   template<class F>
///   void task(F f) {f(/* ... */);}
///   void client() {
///     auto p = make_partial<func>(/*...*/);
///     execute<task<decltype(p)>>(p);  // note explicit template argument
///   }\endcode
/// \deprecated Use a lambda or \c std::bind.
template<auto & F, class... AA>
[[deprecated(
  "use lambda or std::bind")]] constexpr exec::partial<F, std::decay_t<AA>...>
make_partial(AA &&... aa) {
  return {std::forward<AA>(aa)...};
}

/*!
  \link future<Return> Single\endlink or \link
  future<Return,exec::launch_type_t::index> multiple\endlink future.

  A single future can be a task argument and parameter; the task runs only
  when the value is ready.
  A multi-valued future may be passed to a task expecting a single one
  (which is then executed once with each value).

  @tparam Return The return type of the task.
  @tparam Launch FleCSI launch type: single/index.
*/
template<typename Return,
  exec::launch_type_t Launch = exec::launch_type_t::single>
struct future;

#ifdef DOXYGEN // implemented per-backend
/// Single-valued future.
template<typename Return>
struct future<Return> {
  /// Wait on the task to finish.
  void wait();
  /// Get the task's result.
  [[nodiscard]] Return get(bool silence_warnings = false);
};

/// Multi-valued future from an index launch.
template<typename Return>
struct future<Return, exec::launch_type_t::index> {
  /// Wait on all the tasks to finish.
  void wait(bool silence_warnings = false);
  /// Get the result of one of the tasks.
  /// Note that all processes must select the same \a index.
  Return get(Color index = 0, bool silence_warnings = false);
  /// Get the results of all tasks.
  /// \note This member does not exist if \a Return is \c void.
  std::vector<Return> all();
  /// Get the number of tasks.
  Color size() const;
};
#endif

namespace exec::detail {
template<class R>
struct task_param<future<R>> {
  static future<R> replace(const future<R, launch_type_t::index> &) {
    return {};
  }
};
template<class R>
struct must_convert<future<R, launch_type_t::index>> : std::true_type {};

template<class P>
struct task_param<std::vector<P>> {
  template<class A>
  static std::enable_if_t<replace_argument<P, const A &>::special,
    std::vector<P>>
  replace(const std::vector<A> & v) {
    const util::transform_view t(v, exec::replace_argument<P, const A &>);
    return {t.begin(), t.end()};
  }
};
template<class T>
struct must_convert<std::vector<T>> : must_convert<T> {};
template<class P, class A>
struct launch<std::vector<P>, std::vector<A>> {
  using type = decltype(launch<P, A>::get(std::declval<A>()));
  static type get(const std::vector<A> & v) {
    launch_combine ret{type()};
    for(auto & a : v)
      ret = ret | launch_combine(launch<P, A>::get(a));
    return ret.value();
  }
};
template<class T>
struct must_bind<std::vector<T>> : must_bind<T> {};

template<class... PP>
struct task_param<std::tuple<PP...>> {
  // Deduplicating with an alias template fails in Clang (#17042) and MSVC.
  template<class... AA>
  static std::enable_if_t<(replace_argument<PP, const AA &>::special || ...),
    std::tuple<PP...>>
  replace(const std::tuple<AA...> & t) {
    return make(t);
  }
  template<class... AA>
  static std::enable_if_t<(replace_argument<PP, const AA &>::special || ...),
    std::tuple<PP...>>
  replace(std::tuple<AA...> && t) {
    return make(std::move(t));
  }

private:
  template<class T>
  static auto make(T && t) {
    return std::apply(
      [](auto &&... xx) -> std::tuple<PP...> {
        return {exec::replace_argument<PP>(std::forward<decltype(xx)>(xx))...};
      },
      t);
  }
};
template<class... TT>
struct must_convert<std::tuple<TT...>> : std::disjunction<must_convert<TT>...> {
};
template<class... PP, class... AA>
struct launch<std::tuple<PP...>, std::tuple<AA...>> {
  static auto get(const std::tuple<AA...> & t) {
    return std::apply(
      [](auto &... xx) {
        return launch_size(static_cast<std::tuple<PP...> *>(nullptr), xx...);
      },
      t)
      .value();
  }
};
template<class... TT>
struct must_bind<std::tuple<TT...>> : std::disjunction<must_bind<TT>...> {};

template<class P>
struct launch<P, launch_domain> {
  static Index get(const launch_domain & d) {
    return d.size_;
  }
};
template<class P, class T>
struct launch<P, future<T, launch_type_t::index>> {
  static Index get(const future<T, launch_type_t::index> & f) {
    return f.size();
  }
};
} // namespace exec::detail

///\}
} // namespace flecsi

#endif
