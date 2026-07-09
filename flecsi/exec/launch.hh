// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_LAUNCH_HH
#define FLECSI_EXEC_LAUNCH_HH

#include "flecsi/data/field.hh"
#include "flecsi/exec/future.hh"
#include "flecsi/exec/kernel.hh"
#include "flecsi/exec/task_attributes.hh"
#include "flecsi/util/annotation.hh"
#include "flecsi/util/function_traits.hh"

#include <cstddef>
#include <optional>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant> // monostate

namespace flecsi {
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
  std::enable_if_t<!must_convert<std::remove_cvref_t<A>>::value>> {
  static constexpr bool special = false;
  static A replace(A a) { // NB: not P
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
          launch_combine(launch<std::remove_cvref_t<PP>, AA>::get(aa)));
}

template<class, class = void>
struct has_space : std::false_type {};
template<class S>
struct has_space<S, std::void_t<typename S::execution_space>> : std::true_type {
};

template<class F>
void ignore(F); // work around GCC bug #119343

// Verify that two task variants have almost-identical signatures:
template<class, class, class, class>
struct consistent_params;
template<class... TT1, class S1, class... TT2, class S2>
struct consistent_params<std::tuple<TT1...>, S1, std::tuple<TT2...>, S2>
  : std::bool_constant<(
      (std::is_same_v<TT1, TT2> ||
        (std::is_same_v<TT1, S1> && std::is_same_v<TT2, S2>)) &&
      ...)> {};
// Accept function_traits types as common "subexpressions":
template<class F1, class S1, class F2, class S2>
using consistent_signatures = std::conjunction<
  std::is_same<typename F1::return_type, typename F2::return_type>,
  consistent_params<typename F1::arguments_type,
    S1,
    typename F2::arguments_type,
    S2>>;
template<class V, class S1, class S2>
struct consistent_variants
  : consistent_signatures<util::function_t<V::template task<S1>>,
      S1,
      util::function_t<V::template task<S2>>,
      S2> {};
} // namespace detail

// Replaces certain task arguments before conversion to the parameter type.
template<class P, class T>
decltype(auto)
replace_argument(T && t) {
  return detail::replace_argument<std::remove_cvref_t<P>, T &&>::replace(
    std::forward<T>(t));
}

namespace detail {
// Since our Legion task wrapper does not depend on argument types, even for
// an MPI task we must eagerly create parameters and objects for any
// references that require a conversion.  The following progression from task
// argument to task parameter results:
// 1. argument, with the reference introduced by any forwarding function
// 2. the result of replace_argument
// 3. the result of make_parameter (with convert applied for vectors)
// 4. mpi_params, or the result of bind_tuple
// 5. actual task parameter, without cv-qualification (with convert again)
// We derive #2 from #5 and #1.  For non-MPI tasks, we derive #3 from
// #5 and #4 from #3.  For MPI tasks, #3 is #2 unless we need a
// "temporary" to which to bind a reference, in which case it is the type of
// that temporary.  That situation can arise for an element type of a vector
// or tuple, in which case #3 is a vector/tuple of the result(s).

template<class P, class A>
struct replaced {
  using type = decltype(exec::replace_argument<P>(std::declval<A>()));
};

// An approximation of C++23's std::reference_converts_from_temporary:
template<class P, class A>
constexpr bool temporary_v =
  ((std::is_lvalue_reference_v<P> &&
     std::is_const_v<std::remove_reference_t<P>>) ||
    std::is_rvalue_reference_v<P>) &&
  !std::is_convertible_v<std::add_pointer_t<typename replaced<P, A>::type>,
    std::add_pointer_t<P>>;

template<class R, class T>
using same_ref_t = std::conditional_t<std::is_lvalue_reference_v<R>, T &, T &&>;
template<class C, class T> // similar to std::forward_like
using element_t = same_ref_t<C,
  util::maybe_const<std::is_const_v<std::remove_reference_t<C>>, T>>;

template<class P, class A, class D = std::decay_t<A>>
struct sync_storage {
  static constexpr bool temporary = temporary_v<P, A>;
  static_assert(
    !temporary || std::is_move_constructible_v<std::remove_reference_t<P>>,
    "references to non-movable types must bind directly");
  static_assert(!std::conditional_t<temporary, // avoid unneeded instantiation
                  sync_storage<std::remove_cvref_t<P>, A, D>,
                  sync_storage>::temporary,
    "MPI tasks cannot accept references that require nested conversions");
  using type = typename std::
    conditional_t<temporary, std::remove_cvref<P>, replaced<P, A>>::type;
};
template<class P, class A>
using sync_storage_t = typename sync_storage<P, A>::type;
template<class... PP, class T, class... AA>
struct sync_storage<std::tuple<PP...>, T, std::tuple<AA...>> {
  static constexpr bool temporary =
    (sync_storage<PP, element_t<T, AA>>::temporary || ...);
  using type = typename std::conditional_t<temporary,
    std::type_identity<std::tuple<sync_storage_t<PP, element_t<T, AA>>...>>,
    replaced<std::tuple<PP...>, T>>::type; // instantiated only if needed
};
template<class P, class V, class A>
struct sync_storage<std::vector<P>, V, std::vector<A>> {
  static constexpr bool temporary = sync_storage<P, element_t<V, A>>::temporary;
  using type = typename std::conditional_t<temporary,
    std::type_identity<std::vector<sync_storage_t<P, element_t<V, A>>>>,
    replaced<std::vector<P>, V>>::type;
};

template<class>
struct is_tuple : std::false_type {};
template<class... TT>
struct is_tuple<std::tuple<TT...>> : std::true_type {};
template<class>
struct is_vector : std::false_type {};
template<class T>
struct is_vector<std::vector<T>> : std::true_type {};

template<class T, class U>
T convert(U && u);
template<class... TT, class... UU>
std::tuple<TT...>
convert_tuple(std::tuple<TT...> *, UU &&... uu) {
  return {convert<TT>(std::forward<UU>(uu))...};
}

template<class T, class U>
T
convert(U && u) { // deep implicit conversions
  if constexpr(is_tuple<T>::value) // not references
    return apply(
      [](auto &&... xx) {
        return convert_tuple(
          static_cast<T *>(nullptr), std::forward<decltype(xx)>(xx)...);
      },
      std::forward<U>(u));
  else if constexpr(is_vector<T>::value) {
    const auto t = std::views::transform(u, [](auto && x) {
      return convert<typename T::value_type>(std::forward<decltype(x)>(x));
    });
    return {t.begin(), t.end()};
  }
  else
    return std::forward<U>(u);
}

template<class... FF>
auto
make_tuple(FF... ff) { // use -> decltype(auto)
  return std::tuple<decltype(std::move(ff)())...>(std::move(ff)()...);
}

template<bool M, class P>
struct param_helper {
  static_assert(M || std::is_move_constructible_v<P>,
    "only MPI tasks can accept (references to) non-movable types");
  // This is not used when M, but the assertions are:
  using type = P;
};
template<bool M>
struct protocol {
  template<class P, class = void>
  struct param_storage : param_helper<M, P> {};
  template<class P>
  using param_storage_t = typename param_storage<P>::type;
  template<class P>
  struct param_storage<const P> : param_storage<P> {};
  template<class P>
  struct param_storage<P &> : param_storage<P> {
    static_assert(M || std::is_const_v<P>,
      "only MPI tasks can accept non-const references");
  };
  template<class P>
  struct param_storage<P &&> : param_storage<P> {
    static_assert(M, "only MPI tasks can accept rvalue references");
  };
  template<class P>
  struct param_storage<P *> : param_helper<M, P *> {
    static_assert(M || std::is_const_v<P> || std::is_function_v<P>,
      "only MPI tasks can accept non-const pointers");
  };
  template<data::layout L, class T, Privileges P>
  struct param_storage<data::accessor<L, T, P>>
    : param_helper<M, data::accessor<L, T, P>> {
    static_assert(
      data::portable_v<T> ||
        (M && (privilege_count(P) <= 1 ||
                !privilege_read(get_privilege(privilege_count(P) - 1, P)))),
      "only MPI tasks can accept non-portable field accessors; "
      "they must not access ghosts");
  };
  // NB: this recursion happens regardless of must_convert.
  template<class... PP>
  struct param_storage<std::tuple<PP...>> {
    using type = std::tuple<param_storage_t<PP>...>;
  };
  template<class P>
  struct param_storage<std::vector<P>> {
    using type = std::vector<param_storage_t<P>>;
  };

  template<class P, class A>
  static decltype(auto) make_parameter(A && a) {
    if constexpr(!M)
      static_assert(std::is_copy_constructible_v<P>,
        "only MPI tasks can accept non-copyable parameters by value");
    return convert<typename std::conditional_t<M,
      sync_storage<P, A &&>,
      std::type_identity<param_storage_t<P>>>::type // always instantiated
      >(exec::replace_argument<P>(std::forward<A>(a)));
  }

  template<class... PP, class... AA>
  static auto make_parameters(std::tuple<PP...> * /* to deduce PP */,
    AA &&... aa) {
    return make_tuple([&]() -> decltype(auto) {
      return make_parameter<PP>(std::forward<AA>(aa));
    }...);
  }
};

template<class... PP, class... BB>
auto
convert_parameters(std::tuple<PP...> *, std::tuple<BB...> && bound) {
  return convert<std::tuple<std::conditional_t<std::is_convertible_v<BB &&, PP>,
    BB &&,
    std::remove_cvref_t<PP>>...>>(std::move(bound));
}
} // namespace detail

template<auto & F, TaskAttributes A>
struct launch {
  using function = util::function_t<F>;
  using Params = typename function::arguments_type;
  using Return = std::remove_cv_t<typename function::return_type>;
  static constexpr auto proc = mask_to_processor_type(A);
  static constexpr bool mpi = proc == processor::mpi;
  using protocol = detail::protocol<mpi>;

  template<class... AA>
  static auto params(AA &&... aa) {
    return protocol::make_parameters(
      static_cast<Params *>(nullptr), std::forward<AA>(aa)...);
  }
  // Return the number of point tasks for the given arguments, or
  // std::monostate() if a single launch is appropriate.
  template<class... AA>
  static auto size(const AA &... aa) {
    return detail::launch_size<mpi>(static_cast<Params *>(nullptr), aa...)
      .get();
  }

  template<class P>
  static auto call(P && params) noexcept {
    return util::annotation::rguard<util::annotation::execute_task_user>(
             util::symbol<F>()),
           apply(F,
             detail::convert_parameters(
               static_cast<Params *>(nullptr), std::forward<P>(params)));
  }
};

/// An explicit launch domain size.
struct launch_domain {
  Color size_;
};

/// Executor derivation.
template<class D>
struct executor_base {
  /// \see \c executor
  template<class C, class F>
  void for_each(C && c, F && f) const {
    d().named({}).for_each(std::forward<C>(c), std::forward<F>(f));
  }
  /// \see \c executor
  template<class R, class T, class C, class F>
  [[nodiscard]] T reduce(C && c, F && f) const {
    return d().named({}).template reduce<R, T>(
      std::forward<C>(c), std::forward<F>(f));
  }

private:
  struct no_label {};

  template<class C>
  struct forall_t {
    template<class F>
    void operator->*(F f) && {
      e.for_each(std::move(c), std::move(f));
    }
    const D & e;
    C c;
  };
  template<class C, class R, class T>
  struct reduceall_t {
    template<class F>
    [[nodiscard]] T operator->*(F f) && {
      return e.template reduce<R, T>(std::move(c), std::move(f));
    }
    const D & e;
    C c;
  };

  const D & d() const {
    return static_cast<const D &>(*this);
  }

public:
  template<class P>
  forall_t<P> flecsi_macro_forall(P && p) const {
    return {d(), std::forward<P>(p)};
  }
  template<class R, class T, class P>
  reduceall_t<P, R, T>
  flecsi_macro_reduceall(R *, T *, P && p, no_label) const {
    return {d(), std::forward<P>(p)};
  }
};

// Users need not be aware of these as different classes: they just call any
// subsequence of {threads, named} to get an object that can launch a kernel.
// In fact, each produces a class earlier in the following sequence.

/// Parallel operations given a name for debugging or profiling.
template<class S, unsigned T, unsigned B>
struct executor : executor_base<executor<S, T, B>> {
  explicit executor(std::string n) : name(std::move(n)) {}

private:
  // auto to avoid requiring Kokkos for merely choosing the execution space.
  auto range(util::id n) const {
    return exec::policy_type<typename S::execution_space,
      Kokkos::LaunchBounds<T, B>>(0, n);
  }

public:
  /// Apply a function to every element of a range.
  /// \param c sized random-access range, potentially copied
  template<class C, class F>
  void for_each(C && c, F && f) const {
    kok::parallel_for(
      name, range(c.size()), std::forward<C>(c), std::forward<F>(f));
  }
  /// Reduce the results of a function applied to every element of a range.
  /// \tparam R reduction operation type
  /// \tparam A accumulator type
  template<class R, class A, class C, class F>
  [[nodiscard]] A reduce(C && c, F && f) const {
    return kok::parallel_reduce<R, A>(
      name, range(c.size()), std::forward<C>(c), std::forward<F>(f));
  }

private:
  std::string name;
};
/// Parallel operations with thread configuration.
template<class S, unsigned T, unsigned B>
struct blocks : executor_base<blocks<S, T, B>> {
  /// Specify a name for an operation.
  /// \return \c executor
  auto named(std::string n) const {
    return executor<S, T, B>(std::move(n));
  }
};
/// A node-local context for potentially parallel operations.
template<class S>
struct agent : executor_base<agent<S>> {
  /// Specify threads and blocks for an operation.
  /// These are ignored if not supported by the execution space.
  /// \see \c Kokkos::LaunchBounds
  /// \return \c blocks
  template<unsigned T, unsigned B>
  auto threads() const {
    return blocks<S, T, B>();
  }
  /// \see \c blocks
  auto named(std::string n) const {
    return threads<0, 0>().named(std::move(n));
  }
};

/// An execution space.
struct space_base : data::bind_tag, data::convert_tag {
  /// Information about a task launch.
  struct tasks {
    Color size, ///< Number of task instances.
      index; ///< Current task instance (or point task) number.
  };
  /// Describe the tasks launched.
  const tasks & launch() const {
    return t;
  }

  void bind(Color n, Color i) {
    t = {n, i};
  }

  template<class T>
  using keep = std::conditional_t<std::is_base_of_v<space_base, T>, T, void>;

private:
  tasks t{};
};
/// Execution space operations.
template<class S>
struct space : space_base {
  // Since derived executors should be able to outlive their bases, it makes
  // sense to allow a base executor to outlive its (potentially copied) space.

  /// Get an executor for potentially parallel operations on this space.
  agent<S> executor() const {
    return {};
  }
};

/// Single-core execution space.
struct cpu : space<cpu> {
  static constexpr processor proc = processor::loc;
#ifdef KOKKOS_ENABLE_SERIAL
  using execution_space = Kokkos::Serial;
#endif // otherwise undefined
};
/// GPU execution space.
struct gpu : space<gpu> {
  static constexpr processor proc = processor::toc;
#ifdef KOKKOS_ENABLE_CUDA
  using execution_space = Kokkos::Cuda;
#elif defined(KOKKOS_ENABLE_HIP)
  using execution_space = Kokkos::HIP;
#endif // otherwise undefined
};
/// OpenMP execution space.
struct omp : space<omp> {
  static constexpr processor proc = processor::omp;
#ifdef KOKKOS_ENABLE_OPENMP
  using execution_space = Kokkos::OpenMP;
#endif // otherwise undefined
};

template<processor>
struct processor_space;
template<>
struct processor_space<processor::loc> {
  using type = cpu;
};
template<>
struct processor_space<processor::toc> {
  using type = gpu;
};
template<>
struct processor_space<processor::omp> {
  using type = omp;
};
template<>
struct processor_space<processor::mpi> : processor_space<processor::loc> {};
template<processor P>
using processor_space_t = typename processor_space<P>::type;

/// The available accelerated execution space.  Defined as \c gpu or \c omp if
/// support for one of those is available, otherwise \c cpu.
using accelerator = std::conditional_t<detail::has_space<gpu>::value,
  gpu,
  std::conditional_t<detail::has_space<omp>::value, omp, cpu>>;

// Find the (single) execution space among parameter types (or void):
template<class T>
struct processor_combine {
  using type = T;
  template<class U>
  auto operator|(const processor_combine<U> c) const { // not actually called
    if constexpr(std::is_void_v<T>)
      return c;
    else {
      static_assert(std::is_void_v<U> || std::is_same_v<T, U>,
        "execution space types conflict");
      return *this;
    }
  }
};
template<class>
struct param_space;
template<class... TT>
struct param_space<std::tuple<TT...>> {
  using type = typename decltype((
    processor_combine<void>() | ... |
    processor_combine<space_base::keep<std::remove_cvref_t<TT>>>()))::type;
};

template<class, class, class = void>
struct has_variant : std::false_type {};
template<class V, class S>
struct has_variant<V, S, decltype(detail::ignore(V::template task<S>))>
  : std::true_type {};
template<class V>
struct has_variant<V, void, decltype(detail::ignore(V::task))>
  : std::true_type {};
template<class V, class S>
constexpr bool has_variant_v = has_variant<V, S>::value;
template<class V, class S>
constexpr bool use_variant_v =
  std::conjunction_v<detail::has_space<S>, has_variant<V, S>>;

// Dynamic selection will be a compatible extension.
template<class V>
using task_variant = std::conditional_t<use_variant_v<V, gpu>,
  gpu,
  std::conditional_t<use_variant_v<V, omp>, omp, cpu>>;

template<class V, class... SS>
constexpr bool consistent_task =
  (std::disjunction_v<std::negation<has_variant<V, SS>>,
     detail::consistent_variants<V, task_variant<V>, SS>> &&
    ...);

struct on_t : data::convert_tag {};
/// Placeholder argument that corresponds to an execution-\ref space task
/// parameter.
inline constexpr on_t on;

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
///   \code
///   void func(/*...*/);
///   template<class F>
///   void task(F f) {f(/* ... */);}
///   void client(scheduler &s) {
///     auto p = make_partial<func>(/*...*/);
///     s.execute<task<decltype(p)>>(p);  // note explicit template argument
///   }
///   \endcode
///
/// \ns.
/// \deprecated Use a lambda or \c std::bind_front.
template<auto & F, class... AA>
[[deprecated("use lambda or std::bind_front")]] constexpr exec::partial<F,
  std::decay_t<AA>...>
make_partial(AA &&... aa) {
  return {std::forward<AA>(aa)...};
}

namespace exec::detail {
template<>
struct task_param<cpu> {
  static cpu replace(const on_t &) {
    return {};
  }
};
template<>
struct task_param<gpu> {
  static gpu replace(const on_t &) {
    return {};
  }
};
template<>
struct task_param<omp> {
  static omp replace(const on_t &) {
    return {};
  }
};

template<class R>
struct task_param<future<R>> {
  static future<R> replace(const future<R, launch_type_t::index> &) {
    return {};
  }
};
template<class R>
struct must_convert<future<R, launch_type_t::index>> : std::true_type {};
template<class P, class T>
struct launch<P, future<T, launch_type_t::index>> {
  static Index get(const future<T, launch_type_t::index> & f) {
    return f.size();
  }
};

template<class P>
struct task_param<std::vector<P>> {
  // Copy (breaking non-const reference parameters) only if necessary:
  template<class A,
    class = std::enable_if_t<replace_argument<P, const A &>::special>>
  static auto replace(const std::vector<A> & v) {
    return make(v);
  }
  template<class A,
    class = std::enable_if_t<replace_argument<P, A &&>::special>>
  static auto replace(std::vector<A> && v) {
    return make(std::move(v));
  }

private:
  template<class T>
  static auto make(T && v) {
    const util::transform_view<T &, decltype([](auto && x) -> decltype(auto) {
      return exec::replace_argument<P>(
        static_cast<detail::same_ref_t<T, decltype(x)>>(x));
    })>
      t(v);
    return std::vector(t.begin(), t.end());
  }
};
template<class T>
struct must_convert<std::vector<T>> : must_convert<T> {};
template<class P, class A>
struct launch<std::vector<P>, std::vector<A>> {
  using type = decltype(launch<P, A>::get(std::declval<A>()));
  static type get(const std::vector<A> & v) {
    launch_combine ret{type()};
    if constexpr(!std::is_same_v<A, bool>)
      for(auto & a : v)
        ret = ret | launch_combine(launch<P, A>::get(a));
    return ret.value();
  }
};
template<class T>
struct must_bind<std::vector<T>> : must_bind<T> {};

template<class... PP>
struct task_param<std::tuple<PP...>> {
  template<class... AA,
    class = std::enable_if_t<(
      replace_argument<std::remove_cvref_t<PP>, const AA &>::special || ...)>>
  static auto replace(const std::tuple<AA...> & t) {
    return make(t);
  }
  template<class... AA,
    class = std::enable_if_t<(
      replace_argument<std::remove_cvref_t<PP>, AA &&>::special || ...)>>
  static auto replace(std::tuple<AA...> && t) {
    return make(std::move(t));
  }

private:
  template<class T>
  static auto make(T && t) {
    return std::apply(
      [](auto &&... xx) {
        return make_tuple([&]() -> decltype(auto) {
          return exec::replace_argument<PP>(std::forward<decltype(xx)>(xx));
        }...);
      },
      std::forward<T>(t));
  }
};
template<class... TT>
struct must_convert<std::tuple<TT...>>
  : std::disjunction<must_convert<std::remove_cvref_t<TT>>...> {};
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
} // namespace exec::detail

///\}
} // namespace flecsi

#endif
