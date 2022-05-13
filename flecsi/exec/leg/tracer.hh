#ifndef FLECSI_LEG_EXEC_TRACER_HH
#define FLECSI_LEG_EXEC_TRACER_HH

#include "flecsi/run/context.hh"
#include <legion.h>

namespace flecsi::exec {

struct trace {

  struct guard;
  using id_t = int;

  inline guard make_guard();

  explicit trace() : trace(g_id_++) {}
  explicit trace(id_t id) : id_(id), skip_(false) {}

  trace(trace &&) = default;

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
        id_,
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
        Legion::Runtime::get_context(), id_);
      tracing = false;
    }
    else {
      skip_ = false;
    }
  }

public:
  static bool is_tracing() {
    return tracing;
  }

private:
  static inline id_t g_id_ = 0;
  id_t id_;
  bool skip_;
  static inline bool tracing = false;

}; // struct trace

} // namespace flecsi::exec

#endif // FLECSI_LEG_EXEC_TRACER_HH
