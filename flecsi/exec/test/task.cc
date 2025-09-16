#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;

using reduction_type = std::uint64_t;

template<task_attributes_mask_t P, exec::processor T>
constexpr bool
test() {
  static_assert(exec::mask_to_processor_type(P | leaf | inner) == T);
  return true;
}

static_assert(test<loc, exec::processor::loc>());
static_assert(test<toc, exec::processor::toc>());

template<privilege... PP, std::size_t... II>
constexpr void
priv(std::index_sequence<II...>) {
  constexpr auto p = privilege_pack<PP...>;
  static_assert(((get_privilege(II, p) == PP) && ...));
}
template<privilege... PP>
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
simple(TYPE arg) noexcept {
  flog(info) << "arg(" << arg << ")\n";
} // simple

struct move {
  template<class S>
  static void task(S, const std::unique_ptr<int> &) noexcept = delete;
};
template<>
void
move::task(exec::cpu c, const std::unique_ptr<int> &) noexcept {
  std::cerr << "moveTask: " << c.launch().index << '/' << c.launch().size
            << '\n';
}

template<class T, class F>
void
seq(const T & s, F f) noexcept {
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

void
mpi(int * p) {
  *p = 1;
}

} // namespace hydro

namespace {
void
vb(const std::vector<bool> &) noexcept {}

int
index_task(const flecsi::runtime * r, exec::launch_domain) noexcept {
  UNIT("TASK") {
    flog(info) << "processes: " << r->processes() << std::endl;
    flog(info) << "process: " << r->process() << std::endl;
    // flog(info)
    // << "colors: " << colors() << std::endl; flog(info) << "color: " <<
    // color()
    // << std::endl;

    EXPECT_LT(r->process(), r->processes());
    EXPECT_GE(r->process(), 0u);
    // EXPECT_LT(color(), domain.size());
    // EXPECT_GE(color(), 0u);
    // EXPECT_EQ(colors(), domain.size());
  };
}
} // namespace

void
init_array(exec::cpu s,
  std::vector<field<reduction_type>::accessor<wo>> v) noexcept {
  flog_assert(v.size() == 1, "wrong accessor count");
  int i = 0;
  for(auto & vv : v.front().span()) {
    vv = s.launch().index + i++;
  }
}
void
init(field<reduction_type>::accessor<wo> v) noexcept {
  for(auto & vv : v.span()) {
    vv = 0;
  }
}
int
check(field<reduction_type>::accessor<ro> v, const int np) noexcept {
  UNIT("TASK") {
    for(std::size_t i = 0; i < v.span().size(); ++i) {
      reduction_type n = np - 1 + i;
      reduction_type t = n * (n + 1) - (i - 1) * i;
      EXPECT_EQ(v[i], t);
    }
  };
}
void
reduction(std::tuple<field<reduction_type>::accessor<ro>,
  field<reduction_type>::reduction<flecsi::exec::fold::sum>> t) noexcept {
  auto & [v, r] = t;
  assert(v.span().size() == r.span().size());
  for(std::size_t i = 0; i < v.span().size(); ++i) {
    r[i](v[i]);
  }
} // reduce_task

using arr = topo::array<void>;
const field<reduction_type>::definition<arr> arr_f;

const field<reduction_type>::definition<topo::global> gl_arr_f;
const field<int, data::particle>::definition<topo::global> gpart;

void
gpinit(field<int, data::particle>::mutator<wo> m) noexcept {
  m.insert(17);
}
int
gpuse(field<int, data::particle>::accessor<ro> a,
  exec::launch_domain) noexcept {
  return std::accumulate(a.begin(), a.end(), 0);
}

namespace user_types {

const field<int>::definition<arr> int_field;
const field<double>::definition<arr> double_field;

constexpr int test_int = 317;
constexpr double test_double = 8.64;

struct type_with_accessors : data::params_tag {
  field<int>::accessor<wo> fa1;
  field<double>::accessor<wo> fa2;

  type_with_accessors(decltype(fa1) fa1, decltype(fa2) fa2)
    : fa1(fa1), fa2(fa2) {}

  auto flecsi_params() {
    return std::tie(fa1, fa2);
  }

  void update_members(int test_int, double test_double) const {
    fa1[0] = test_int;
    fa2[0] = test_double;
  }
};

void
set_field_values(type_with_accessors instance) noexcept {
  instance.update_members(test_int, test_double);
}

int
get_field_values(type_with_accessors instance) noexcept {
  UNIT() {
    ASSERT_EQ(instance.fa1[0], test_int);
    ASSERT_EQ(instance.fa2[0], test_double);
  };
}

struct type_with_references : data::arg_tag {
  using topo_t = arr::topology;
  topo_t & topo1;
  topo_t & topo2;

  type_with_references(topo_t & topo1, topo_t & topo2)
    : topo1(topo1), topo2(topo2) {}

  auto flecsi_arg() const {
    return std::tuple(
      user_types::int_field(topo1), user_types::double_field(topo2));
  }
};

void
set_values(std::tuple<field<int>::accessor<wo>, field<double>::accessor<wo>>
    aa) noexcept {
  auto & [fa1, fa2] = aa;
  fa1[0] = test_int;
  fa2[0] = test_double;
}

int
get_values(std::tuple<field<int>::accessor<ro>, field<double>::accessor<ro>>
    aa) noexcept {
  UNIT() {
    auto & [fa1, fa2] = aa;
    EXPECT_EQ(fa1[0], test_int);
    EXPECT_EQ(fa2[0], test_double);
  };
}

} // namespace user_types

int
task_driver(scheduler & s) {
  UNIT() {
    {
      auto & c = run::context::instance();
      flog(info) << "task depth: " << c.task_depth() << std::endl;
      EXPECT_EQ(c.task_depth(), 0);

      auto process = c.process();
      auto processes = c.processes();

      EXPECT_EQ(processes, 4u);
      EXPECT_LT(process, processes);
    }

    s.execute<hydro::simple<float>>(6.2);
    s.execute<hydro::simple<double>>(5.3);
    s.execute<hydro::simple<const float &>>(4.4);
    s.execute<hydro::simple<const double &>>(3.5);
    using V = std::vector<std::string>;
    const auto d = [n = 5](const std::string & s) { return s.substr(n); };
    s.execute<hydro::seq<V, decltype(d)>>(
      V{"It's Elementary", "Dear, Dear Data"}, d);

    int x = 0;
    execute<hydro::mpi, mpi>(&x);
    EXPECT_EQ(x, 1); // NB: MPI calls are synchronous

    s.execute<vb>(std::vector<bool>(1));

    constexpr bool add_four = (FLECSI_BACKEND != FLECSI_BACKEND_mpi) &&
                              (FLECSI_BACKEND != FLECSI_BACKEND_hpx);
    EXPECT_EQ(s.test<index_task>(&s.runtime(),
                exec::launch_domain{s.runtime().processes() + 4 * add_four}),
      0);

    // Test reduction
    auto np = s.runtime().processes();
    const int vpp = 5;
    // Array of initial values per color
    arr::topology arr_s(s, arr::coloring(np, vpp));
    auto arr_vals = arr_f(arr_s);
    s.execute<init_array>(exec::on, std::vector{arr_vals});
    // Reduction
    topo::global::topology gl_arr_s(s, vpp);
    auto vals = gl_arr_f(gl_arr_s);
    // Init reduction array to 0
    s.execute<init>(vals);
    for(int i = 0; i < 2; ++i)
      s.execute<reduction>(std::tuple(arr_vals, vals));
    EXPECT_EQ(s.test<check>(vals, np), 0);
    s.execute<gpinit>(gpart(gl_arr_s));
    EXPECT_EQ((s.reduce<gpuse, exec::fold::sum>(
                 gpart(gl_arr_s), exec::launch_domain{np}))
                .get(),
      17 * np);

    exec::trace t0, t1 = std::move(t0);
    t1.skip();
    for(int i = 0; i < 5; ++i) {
      auto g = t1.make_guard();
      s.execute<hydro::simple<float>>(6.2);
    }

    const float obj = 8.9;
    s.execute<hydro::simple<const float *>>(&obj);
    s.execute<hydro::move>(exec::on, std::make_unique<int>());

    // params test
    s.execute<user_types::set_field_values>(std::tuple(
      user_types::int_field(arr_s), user_types::double_field(arr_s)));
    EXPECT_EQ(s.test<user_types::get_field_values>(std::tuple(
                user_types::int_field(arr_s), user_types::double_field(arr_s))),
      0);

    // args test
    user_types::type_with_references type_with_references_instance{
      arr_s, arr_s};
    s.execute<user_types::set_values>(type_with_references_instance);
    EXPECT_EQ(s.test<user_types::get_values>(type_with_references_instance), 0);

    // params and args combined
    s.execute<user_types::set_field_values>(type_with_references_instance);
    EXPECT_EQ(
      s.test<user_types::get_field_values>(type_with_references_instance), 0);
  };
} // task_driver

util::unit::driver<task_driver> driver;
