#ifndef FLECSI_MPI_EXEC_TRACER_HH
#define FLECSI_MPI_EXEC_TRACER_HH

namespace flecsi::exec {

struct trace {

  struct guard;
  using id_t = int;

  inline guard make_guard();

  trace() {}
  explicit trace(bool) {}
  explicit trace(id_t, bool = true) {}

  trace(trace &&) = default;

  void skip() {}

private:
  void start() {}
  void stop() {}
}; // struct trace

} // namespace flecsi::exec

#endif // FLECSI_MPI_EXEC_TRACER_HH