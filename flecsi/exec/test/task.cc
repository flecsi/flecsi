/*
    @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
   /@@/////  /@@          @@////@@ @@////// /@@
   /@@       /@@  @@@@@  @@    // /@@       /@@
   /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
   /@@////   /@@/@@@@@@@/@@       ////////@@/@@
   /@@       /@@/@@//// //@@    @@       /@@/@@
   /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
   //       ///  //////   //////  ////////  //

   Copyright (c) 2016, Triad National Security, LLC
   All rights reserved.
                                                                              */

#include "flecsi/execution.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;

flog::devel_tag task_tag("task");

template<task_attributes_mask_t P, exec::task_processor_type_t T>
constexpr bool
test() {
  static_assert(
    exec::mask_to_processor_type(P | leaf | inner | idempotent) == T);
  return true;
}

static_assert(test<loc, exec::task_processor_type_t::loc>());
static_assert(test<toc, exec::task_processor_type_t::toc>());

template<partition_privilege_t... PP, std::size_t... II>
constexpr void
priv(std::index_sequence<II...>) {
  constexpr auto p = privilege_pack<PP...>;
  static_assert(((get_privilege(II, p) == PP) && ...));
}
template<partition_privilege_t... PP>
constexpr bool
priv() {
  priv<PP...>(std::make_index_sequence<sizeof...(PP)>());
  return true;
}

static_assert(priv<rw>());
static_assert(priv<wo, rw>());
static_assert(priv<ro, wo, rw>());
static_assert(priv<na, ro, wo, rw>());

// ---------------
namespace hydro {

template<typename TYPE>
void
simple(TYPE arg) {
  flog::devel_guard guard(task_tag);
  flog(info) << "arg(" << arg << ")\n";
} // simple

template<class T, class F>
void
seq(const T & s, F f) {
  flog::devel_guard guard(task_tag);
  [&](auto && log) {
    bool first = true;
    for(auto & x : s) {
      if(first)
        first = false;
      else
        log << ',';
      log << f(x);
    }
    log << ")\n";
  }(flog_info("s(")); // keep temporary alive throughout
}

int
mpi(int * p) {
  *p = 1;
  return 4;
}

} // namespace hydro

flog::devel_tag color_tag("color");

namespace {
auto
drop(int n, const std::string & s) {
  return s.substr(n);
}
} // namespace

int
task_driver() {
  UNIT() {
    {
      auto & c = run::context::instance();
      flog(info) << "task depth: " << c.task_depth() << std::endl;
      ASSERT_EQ(c.task_depth(), 0);

      auto process = c.process();
      auto processes = c.processes();
      auto tpp = c.threads_per_process();

      {
        flog::devel_guard guard(color_tag);
        flog(info) << "(raw)" << std::endl
                   << "\tprocess: " << process << std::endl
                   << "\tprocesses: " << processes << std::endl
                   << "\tthreads_per_process: " << tpp << std::endl;
      }

      ASSERT_EQ(processes, 4u);
      ASSERT_LT(process, processes);
    }

    execute<hydro::simple<float>>(6.2);
    execute<hydro::simple<double>>(5.3);
    execute<hydro::simple<const float &>>(4.4);
    execute<hydro::simple<const double &>>(3.5);
    using V = std::vector<std::string>;
    const auto d = make_partial<drop>(5l);
    execute<hydro::seq<V, decltype(d.param)>>(
      V{"It's Elementary", "Dear, Dear Data"}, d);

    int x = 0;
    ASSERT_EQ((execute<hydro::mpi, mpi>(&x).get(0)), 4);
    ASSERT_EQ(x, 1); // NB: MPI calls are synchronous
  };
} // task_driver

flecsi::unit::driver<task_driver> driver;
