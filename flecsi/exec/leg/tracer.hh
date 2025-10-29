#ifndef FLECSI_LEG_EXEC_TRACER_HH
#define FLECSI_LEG_EXEC_TRACER_HH

#include "flecsi/data/field.hh"
#include "flecsi/run/backend.hh"
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
        Legion::Runtime::get_context(), *tracing);
    }
  }

  void stop() {
    if(!skip_) {
      flog_assert(tracing == dat.get(), "wrong trace");
      Legion::Runtime::get_runtime()->end_trace(
        Legion::Runtime::get_context(), *tracing);
      tracing->rewind();
      tracing = nullptr;
      // Invalidate current trace ID if resizing cannot be skipped:
      if(enact_tracing_epilog())
        *this = {};
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

public:
  template<class T,
    flecsi::data::layout L,
    class Topo,
    typename Topo::index_space Space>
  // save the dynamic field to enact the resizing at the end of the trace
  static void save_dynamic_field(
    const flecsi::data::field_reference<T, L, Topo, Space> & a) {
    // use the address of the repartitions as a key
    if(rsz_keys.insert(&a.get_elements()).second)
      resize_wrappers.push_back(
        std::make_pair([a]() { return a.get_elements().maybe_resize(); },
          [a]() { a.get_elements().reduce_rsz_required(); }));
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
  std::unique_ptr<data> dat;
  bool skip_ = false;
  static inline data * tracing = nullptr;

  static inline std::vector<
    std::pair<std::function<bool()>, std::function<void()>>>
    resize_wrappers;
  static inline std::set<const void *> rsz_keys;
}; // struct trace

} // namespace flecsi::exec

#endif // FLECSI_LEG_EXEC_TRACER_HH
