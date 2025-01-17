#ifndef FLECSI_LEG_EXEC_TRACER_HH
#define FLECSI_LEG_EXEC_TRACER_HH

#include "flecsi/flog.hh"
#include "flecsi/run/backend.hh"
#include "flecsi/util/common.hh"
#include "flecsi/util/types.hh" // Color

namespace flecsi::exec {

struct trace {

  struct guard;
  using id_t = Legion::TraceID;

  inline guard make_guard();

  trace()
    : dat(std::make_unique<data>(
        Legion::Runtime::get_runtime()->generate_dynamic_trace_id())) {}
  [[deprecated("use default constructor")]] explicit trace(id_t id)
    : dat(std::make_unique<data>(id)) {}

  void skip() {
    skip_ = true;
  }

private:
  struct data {
    explicit data(id_t id) : id(id) {}
    data(data &&) = delete; // for (cautious) address stability

    operator id_t() const {
      return id;
    }

    void rewind() {
      where = 0;
    }
    const run::task_count::ptr & next() {
      return where++ < tasks.size()
               ? tasks[where - 1]
               : tasks.emplace_back(std::make_shared<run::task_count>());
    }

  private:
    id_t id;
    std::vector<run::task_count::ptr> tasks; // for each launch in order
    run::task_idx where = 0;
  };

  void start() {
    if(!skip_) {
      if(tracing)
        flog_fatal("Trace already running: traces cannot be overlapping");
      tracing = dat.get();
      // Call Legion tracing tool
      Legion::Runtime::get_runtime()->begin_trace(
        Legion::Runtime::get_context(),
        *tracing,
        false, // logical_only = false
        false, // static_trace  = false
        NULL // std::set<RegionTreeID> *managed = NULL
      );
    }
  }

  void stop() {
    if(!skip_) {
      flog_assert(tracing == dat.get(), "wrong trace");
      Legion::Runtime::get_runtime()->end_trace(
        Legion::Runtime::get_context(), *tracing);
      tracing->rewind();
      tracing = nullptr;
    }
    else {
      skip_ = false;
    }
  }

public:
  static data * current() {
    return tracing;
  }

  friend bool is_tracing() {
    return tracing;
  }

private:
  std::unique_ptr<data> dat;
  bool skip_ = false;
  static inline data * tracing = nullptr;

}; // struct trace

} // namespace flecsi::exec

#endif // FLECSI_LEG_EXEC_TRACER_HH
