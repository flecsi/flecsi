// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_ANNOTATION_HH
#define FLECSI_UTIL_ANNOTATION_HH

#include <flecsi-config.h>

#if FLECSI_CALIPER_DETAIL != FLECSI_CALIPER_DETAIL_none
#include <caliper/Annotation.h>
#endif

#include <string>
#include <type_traits>

namespace flecsi {
namespace util {

namespace annotation {
/// \defgroup annotation Profiling
/// Markers for categorizing performance measurements.
///
/// This utility provides an interface to the
/// [Caliper](http://software.llnl.gov/Caliper/) source-code
/// annotation API.  Regions of code are marked and annotations are
/// recorded in Caliper depending on the `CALIPER_DETAIL` CMake
/// option.  Caliper's [runtime
/// configuration](http://software.llnl.gov/Caliper/configuration.html)
/// can then be used to control performance measurement and collection
/// for the annotations.
///
/// \ingroup utils
/// \{

/// used for specifying what detail of annotations to collect.
enum class detail { low, medium, high };

#if FLECSI_CALIPER_DETAIL == FLECSI_CALIPER_DETAIL_high
inline constexpr detail detail_level{detail::high};
#elif FLECSI_CALIPER_DETAIL == FLECSI_CALIPER_DETAIL_medium
inline constexpr detail detail_level{detail::medium};
#elif FLECSI_CALIPER_DETAIL == FLECSI_CALIPER_DETAIL_low
inline constexpr detail detail_level{detail::low};
#else // FLECSI_CALIPER_DETAIL == FLECSI_CALIPER_DETAIL_none
#define DISABLE_CALIPER
#endif

inline const char *
c_str(const char * s) {
  return s;
}
inline const char *
c_str(const std::string & s) {
  return s.c_str();
}

/**
 * base for annotation contexts.
 *
 * Annotation contexts correspond to Caliper context attributes and
 * are used to group code regions.
 *
 * \tparam T type used to label context by including a char[] static
 * member called name.
 */
template<class T>
struct context {
#if !defined(DISABLE_CALIPER)
  static cali::Annotation ann;
#endif
  template<detail D, class F>
  static void begin(F && f) { // f() -> std::string or const char*
    (void)f;
#ifndef DISABLE_CALIPER
    if constexpr(D <= detail_level)
      ann.begin(c_str(std::forward<F>(f)()));
#endif
  }
  // Convenience for the deprecated interfaces:
  template<detail D, class N>
  static void begin_eager(N && n) {
    begin<D>([&]() -> N && { return std::forward<N>(n); });
  }
  template<detail D>
  static void end() {
#ifndef DISABLE_CALIPER
    if constexpr(D <= detail_level)
      ann.end();
#endif
  }
};
struct execution : context<execution> {
  static constexpr char name[] = "FleCSI-Execution";
};

/**
 * base for code region annotations.
 *
 * Code region annotations are used to specify a detail level and name
 * for regions of source-code to be considered for performance analysis.
 *
 *   \tparam CTX annotation context for code region (must inherit from
 * annotation::context).
 */
template<class CTX>
struct region {
  using outer_context = CTX;
  static constexpr detail detail_level = detail::medium;
};
// These exist separately solely to support the deprecated interfaces.
template<class R, class F>
void
region_begin_defer(F && f) {
  R::outer_context::template begin<R::detail_level>(std::forward<F>(f));
}
template<class R>
void
region_begin() {
  region_begin_defer<R>([]() -> auto & { return R::name; });
}
template<class R>
void
region_begin(std::string_view n) {
  region_begin_defer<R>([&]() { return (R::name + "->") += n; });
}
template<class R>
void
region_end() {
  R::outer_context::template end<R::detail_level>();
}

template<class T>
struct execute_task : region<execution> {
  /// Set code region name for regions inheriting from execute_task with the
  /// following prefix.
  inline static const std::string name{"execute_task->" + T::tag};
};
struct execute_task_bind : execute_task<execute_task_bind> {
  inline static const std::string tag{"bind-accessors"};
  static constexpr detail detail_level = detail::high;
};
struct execute_task_prolog : execute_task<execute_task_prolog> {
  inline static const std::string tag{"prolog"};
  static constexpr detail detail_level = detail::high;
};
struct execute_bind_parameters : execute_task<execute_bind_parameters> {
  inline static const std::string tag{"bind_parameters"};
  static constexpr detail detail_level = detail::high;
};
struct execute_task_user : execute_task<execute_task_user> {
  inline static const std::string tag{"user"};
};
struct execute_task_unbind : execute_task<execute_task_unbind> {
  inline static const std::string tag{"unbind-accessors"};
  static constexpr detail detail_level = detail::high;
};
struct execute_task_copy_engine : execute_task<execute_task_copy_engine> {
  inline static const std::string tag{"copy-engine"};
  static constexpr detail detail_level = detail::high;
};
/**
 * Tag beginning of a code region with runtime name.
 *
 * This is used to mark code regions with a name at runtime in
 * contrast to using a region type.
 *
 * \tparam ctx annotation context for named code region.
 * \tparam detail severity detail level to use for code region.
 * \deprecated Use \ref guard.
 */
template<class ctx, detail severity>
[[deprecated("use annotation::guard")]] std::enable_if_t<
  std::is_base_of<context<ctx>, ctx>::value>
begin(const char * region_name) {
  ctx::template begin_eager<severity>(region_name);
}
template<class ctx, detail severity>
[[deprecated("use annotation::guard")]] std::enable_if_t<
  std::is_base_of<context<ctx>, ctx>::value>
begin(const std::string & region_name) {
  begin<ctx, severity>(region_name.c_str());
}

/**
 * Tag beginning of code region with caliper annotation.
 *
 * The region is only tagged if caliper is enabled and reg::detail_level
 * is compatible with the current annotation detail level.
 *
 * \tparam reg code region to tag (type inherits from annotation::region).
 * \deprecated Use \ref rguard.
 */
template<class reg>
[[deprecated("use annotation::rguard")]] std::enable_if_t<
  std::is_base_of<context<typename reg::outer_context>,
    typename reg::outer_context>::value>
begin() {
  region_begin<reg>();
}

/**
 * Tag beginning of an execute_task region.
 *
 * The execute_task region has multiple phases and is associated with a named
 * task.
 *
 * \tparam reg code region to tag (must inherit from
 * annotation::execute_task).
 * \param task_name name of task to
 * tag.
 * \deprecated Use \ref rguard.
 */
template<class reg>
[[deprecated("use annotation::rguard")]] std::enable_if_t<
  std::is_base_of<context<typename reg::outer_context>,
    typename reg::outer_context>::value &&
  std::is_base_of<execute_task<reg>, reg>::value>
begin(std::string_view task_name) {
  region_begin<reg>(task_name);
}

/**
 * Tag end of a named code region.
 *
 * This is used for runtime named code regions (in contrast
 * using region types).
 *
 * \tparam ctx annotation context for named code region.
 * \tparam detail severity detail level to use for code region.
 * \deprecated Use \ref guard.
 */
template<class ctx, detail severity>
[[deprecated("use annotation::guard")]] std::enable_if_t<
  std::is_base_of<context<ctx>, ctx>::value>
end() {
  ctx::template end<severity>();
}

/**
 * Tag end of code region using a region type.
 *
 * The region is only tagged if caliper is enabled and reg::detail_level
 * is compatible with the current annotation detail level.
 *
 * \tparam reg code region to tag (type inherits from annotation::region).
 * \deprecated Use \ref rguard.
 */
template<class reg>
[[deprecated("use annotation::rguard")]] std::enable_if_t<
  std::is_base_of<context<typename reg::outer_context>,
    typename reg::outer_context>::value>
end() {
  region_end<reg>();
}

/**
 * Scope guard for marking a code region.
 *
 * This type is used to mark a runtime named code region based on the
 * lifetime of the guard.
 *
 * \tparam ctx annotation context for named code region.
 * \tparam severity detail level to use for code region.
 */
template<class ctx, detail severity>
class guard
{
public:
  /// Create a guard.
  /// \param a region name as a \c std::string or `const char*`
  template<class Arg>
  guard(Arg && a) {
    ctx::template begin_eager<severity>(std::forward<Arg>(a));
  }
  ~guard() {
    ctx::template end<severity>();
  }
};

/**
 * Scope guard for marking a code region.
 *
 * This type is used to mark a code region identified with a region
 * type based on the lifetime of the guard.
 *
 * \tparam reg code region to tag (type inherits from annotation::region)
 */
template<class reg>
class rguard
{
public:
  /// Create a guard.
  /// \param a an optional task name as a \c std::string_view
  template<class... Arg>
  rguard(Arg &&... a) {
    region_begin<reg>(std::forward<Arg>(a)...);
  }
  ~rguard() {
    region_end<reg>();
  }
};
/// \}
}; // namespace annotation

/// Initialize caliper annotation objects from the context name.
#if !defined(DISABLE_CALIPER)
template<class T>
cali::Annotation annotation::context<T>::ann{T::name};
#endif

} // namespace util
} // namespace flecsi

#endif
