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
namespace data { // these have semantics for exec/
struct bind_tag {}; // must be recognized as a task parameter
// Provides a send member function that accepts a function to call with a
// (subsidiary) task parameter and another function to call to transform the
// corresponding task argument (used only on the caller side).
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
template<class P, class A, class = void> // A is a reference type
struct replace_argument;
// Allow specialization as well as use of convert_tag:
template<class T>
struct must_convert
  : std::integral_constant<bool, std::is_base_of_v<data::convert_tag, T>> {};
template<class P, class A>
struct replace_argument<P,
  A,
  std::enable_if_t<!must_convert<std::decay_t<A>>::value>> {
  static A replace(A a) {
    return static_cast<A>(a);
  }
};
template<class P, class A>
struct replace_argument<P,
  A,
  decltype(void(task_param<P>::replace(std::declval<A>())))> {
  static decltype(auto) replace(A a) {
    return task_param<P>::replace(static_cast<A>(a));
  }
};

// For each parameter-type/argument pair we have either a Color (the
// size of a required index launch), std::monostate (for a required single
// launch), or std::nullptr_t (don't care).

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
  static Color get(const data::field_reference<T, L, Topo, S> & r) {
    return r.topology().colors();
  }
};
template<class P,
  class T,
  data::layout L,
  class Topo,
  typename Topo::index_space S>
struct launch<P, data::multi_reference<T, L, Topo, S>> {
  static Color get(const data::multi_reference<T, L, Topo, S> & r) {
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
        if constexpr(!std::is_same_v<T, std::monostate>)
          flog_assert(t == c.t,
            "implied launch sizes " << t << " and " << c.t << " conflict");
      }
      return *this;
    }
  }
  auto get() const {
    if constexpr(std::is_same_v<T, std::nullptr_t>)
      return std::monostate();
    else
      return t;
  }

private:
  T t;
};

template<bool M, class... PP, class... AA>
auto
launch_size(std::tuple<PP...> *, const AA &... aa) {
  return (launch_combine([] {
    // An MPI task has a known launch domain:
    if constexpr(M)
      return run::context::instance().processes();
    else
      return nullptr;
  }()) | ... |
          launch_combine(launch<std::decay_t<PP>, AA>::get(aa)))
    .get();
}
} // namespace detail
// Replaces certain task arguments before conversion to the parameter type.
template<class P, class T>
decltype(auto)
replace_argument(T && t) {
  return detail::replace_argument<std::decay_t<P>, T &&>::replace(
    std::forward<T>(t));
}

// Return the number of task invocations for the given parameter tuple and
// arguments, or std::monostate() if a single launch is appropriate.
template<TaskAttributes A, class P, class... AA>
auto
launch_size(const AA &... aa) {
  return detail::launch_size<mask_to_processor_type(A) ==
                             task_processor_type_t::mpi>(
    static_cast<P *>(nullptr), aa...);
}

enum class launch_type_t : size_t { single, index };

/// An explicit launch domain size.
struct launch_domain {
  Color size_;
};

/// \if core
/// A simple version of C++20's \c bind_front that can be an argument to a
/// task template.
/// \endif
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
/// Lambdas and \c bind objects may not in general be passed to tasks.
/// \tparam F function to call
/// \tparam AA serializable types
/// \return a function object that can be an argument to a task
/// \note The task will usually be a function template:\code
///   void func(/*...*/);
///   template<class F>
///   void task(F f) {f(/* ... */);}
///   void client() {
///     auto p = make_partial<func>(/*...*/);
///     execute<task<decltype(p)>>(p);  // note explicit template argument
///   }\endcode
template<auto & F, class... AA>
constexpr exec::partial<F, std::decay_t<AA>...>
make_partial(AA &&... aa) {
  return {std::forward<AA>(aa)...};
}

/*!
  \link future<Return> Single\endlink or \link
  future<Return,exec::launch_type_t::index> multiple\endlink future.

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
  Return get(bool silence_warnings = false);
};

/// Multi-valued future from an index launch.
template<typename Return>
struct future<Return, exec::launch_type_t::index> {
  /// Wait on all the tasks to finish.
  void wait(bool silence_warnings = false);
  /// Get the result of one of the tasks.
  Return get(Color index = 0, bool silence_warnings = false);
  /// Get the number of tasks.
  Color size() const;
};
#endif

namespace exec {
template<class R>
struct detail::task_param<future<R>> {
  static future<R> replace(const future<R, launch_type_t::index> &) {
    return {};
  }
};
template<class R>
struct detail::must_convert<future<R, launch_type_t::index>> : std::true_type {
};

template<class P>
struct detail::launch<P, launch_domain> {
  static Color get(const launch_domain & d) {
    return d.size_;
  }
};
template<class P, class T>
struct detail::launch<P, future<T, launch_type_t::index>> {
  static Color get(const future<T, launch_type_t::index> & f) {
    return f.size();
  }
};
} // namespace exec

///\}
} // namespace flecsi

#endif
