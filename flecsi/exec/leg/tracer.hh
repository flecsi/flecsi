#ifndef FLECSI_LEG_EXEC_TRACER_HH
#define FLECSI_LEG_EXEC_TRACER_HH

#include "flecsi/data/field.hh"
#include "flecsi/run/context.hh"

#include <legion.h>

namespace flecsi::exec {

struct trace {

  struct guard;
  using id_t = Legion::TraceID;

  inline guard make_guard();

  trace() : id_(Legion::Runtime::get_runtime()->generate_dynamic_trace_id()) {}
  [[deprecated("use default constructor")]] explicit trace(id_t id) : id_(id) {}

  void skip() {
    skip_ = true;
  }

private:
  void start() {
    if(!skip_) {
      if(tracing)
        flog_fatal("Trace already running: traces cannot be overlapping");
      // Call Legion tracing tool
      Legion::Runtime::get_runtime()->begin_trace(
        Legion::Runtime::get_context(),
        id_.value(),
        false, // logical_only = false
        false, // static_trace  = false
        NULL // std::set<RegionTreeID> *managed = NULL
      );
      tracing = true;
    }
  }

  void stop() {
    if(!skip_) {
      Legion::Runtime::get_runtime()->end_trace(
        Legion::Runtime::get_context(), id_.value());
      tracing = false;
      // invalidate the current trace ID, if the resizing can not be skipped
      if(enact_tracing_epilog())
        *this = {};
    }
    else {
      skip_ = false;
    }
  }

  friend bool is_tracing() {
    return tracing;
  }

public:
  template<class T,
    data::layout L,
    class Topo,
    typename Topo::index_space Space>
  // save the dynamic field to enact the resizing at the end of the trace
  static void save_dynamic_field(
    const data::field_reference<T, L, Topo, Space> & a) {
    // use the address of the repartitions as a key
    if(rsz_keys.insert(&a.get_elements()).second)
      resize_wrappers.push_back(
        std::make_pair([a]() { return a.get_elements().maybe_resize(); },
          [a]() { return a.get_elements().reduce_rsz_required(); }));
  }

  // return true if we had to resize during the epilog
  static bool enact_tracing_epilog() {

    for(auto & [_, reduce_rsz_required] : resize_wrappers)
      reduce_rsz_required();

    bool resized = false;
    for(auto & [enact_resizing, _] : resize_wrappers)
      if(enact_resizing())
        resized = true;

    resize_wrappers.clear();
    rsz_keys.clear();
    return resized;
  }

private:
  util::move_optional<id_t> id_;
  bool skip_ = false;
  static inline bool tracing = false;

  static inline std::vector<
    std::pair<std::function<bool()>, std::function<void()>>>
    resize_wrappers;
  static inline std::set<const void *> rsz_keys;
}; // struct trace

} // namespace flecsi::exec

#endif // FLECSI_LEG_EXEC_TRACER_HH
