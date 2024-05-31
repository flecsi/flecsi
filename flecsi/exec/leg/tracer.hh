#ifndef FLECSI_LEG_EXEC_TRACER_HH
#define FLECSI_LEG_EXEC_TRACER_HH

#include "flecsi/flog.hh"
#include "flecsi/util/common.hh"

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
  util::move_optional<id_t> id_;
  bool skip_ = false;
  static inline bool tracing = false;

}; // struct trace

} // namespace flecsi::exec

#endif // FLECSI_LEG_EXEC_TRACER_HH
