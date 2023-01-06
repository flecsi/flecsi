#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;

flog::devel_tag task_tag("task");

using reduction_type = std::uint64_t;

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

int
index_task(exec::launch_domain) {
  UNIT("TASK") {
    flog(info) << "program: " << program() << std::endl;
    flog(info) << "processes: " << processes() << std::endl;
    flog(info) << "process: " << process() << std::endl;
    // flog(info) << "threads per process: " << threads_per_process() <<
    // std::endl; flog(info) << "threads: " << threads() << std::endl;
    // flog(info)
    // << "colors: " << colors() << std::endl; flog(info) << "color: " <<
    // color()
    // << std::endl;

    ASSERT_LT(process(), processes());
    ASSERT_GE(process(), 0u);
    // ASSERT_LT(color(), domain.size());
    // ASSERT_GE(color(), 0u);
    // ASSERT_EQ(colors(), domain.size());
  };
}
} // namespace

void
init_array(field<reduction_type>::accessor<wo> v) {
  int i = 0;
  for(auto & vv : v.span()) {
    vv = color() + i++;
  }
}
void
init(field<reduction_type>::accessor<wo> v) {
  for(auto & vv : v.span()) {
    vv = 0;
  }
}
int
check(field<reduction_type>::accessor<ro> v, const int np) {
  UNIT("TASK") {
    for(std::size_t i = 0; i < v.span().size(); ++i) {
      reduction_type n = np - 1 + i;
      reduction_type t = n * (n + 1) - (i - 1) * i;
      EXPECT_EQ(v[i], t);
    }
  };
}
void
reduction(field<reduction_type>::accessor<ro> v,
  field<reduction_type>::reduction<flecsi::exec::fold::sum> r) {
  assert(v.span().size() == r.span().size());
  for(std::size_t i = 0; i < v.span().size(); ++i) {
    r[i](v[i]);
  }
} // reduce_task

using arr = topo::array<void>;
arr::slot arr_s;
const field<reduction_type>::definition<arr> arr_f;

topo::global::slot gl_arr_s;
const field<reduction_type>::definition<topo::global> gl_arr_f;

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

    EXPECT_EQ(test<index_task>(exec::launch_domain{
                processes() + 4 * (FLECSI_BACKEND != FLECSI_BACKEND_mpi)}),
      0);

    // Test reduction
    auto np = processes();
    const int vpp = 5;
    // Array of initial values per color
    arr_s.allocate(arr::coloring(np, vpp));
    auto arr_vals = arr_f(arr_s);
    flecsi::execute<init_array>(arr_vals);
    // Reduction
    gl_arr_s.allocate(vpp);
    auto vals = gl_arr_f(gl_arr_s);
    // Init reduction array to 0
    flecsi::execute<init>(vals);
    flecsi::execute<reduction>(arr_vals, vals);
    flecsi::execute<reduction>(arr_vals, vals);
    EXPECT_EQ(test<check>(vals, np), 0);

    // Tracing utility test
    // Auto id + ignore first iteration
    exec::trace t1;
    t1.skip();
    for(int i = 0; i < 5; ++i) {
      auto g = t1.make_guard();
      execute<hydro::simple<float>>(6.2);
    }
    // User id + ignore first iteration
    exec::trace t2(10);
    t2.skip();
    for(int i = 0; i < 5; ++i) {
      auto g = t2.make_guard();
      execute<hydro::simple<float>>(6.2);
    }
    // User id + trace first iteration
    exec::trace t3(11);
    for(int i = 0; i < 5; ++i) {
      auto g = t3.make_guard();
      execute<hydro::simple<float>>(6.2);
    }
  };
} // task_driver

util::unit::driver<task_driver> driver;
