#include "flecsi/util/demangle.hh"
#include "flecsi/util/unit.hh"
#include <flecsi/data.hh>
#include <flecsi/execution.hh>

#include <atomic>

using namespace flecsi;
using namespace flecsi::data;

struct Noisy {
  ~Noisy() {
    ++count;
  }
  std::size_t i = value();
  static std::size_t value() {
    return color() + 1;
  }
  static inline std::atomic<std::size_t> count;
};

using double_field = field<double, single>;
const double_field::definition<topo::index> pressure_field;
using intN = field<int, ragged>;
const intN::definition<topo::index> verts_field, ghost_field;
using double_at = field<double, sparse>;
const double_at::definition<topo::index> vfrac_field;

struct trivial_array : topo::specialization<topo::user, trivial_array> {};

using short_part = field<short, particle>;
const short_part::definition<trivial_array> particles;
const intN::definition<trivial_array> arag;

constexpr std::size_t column = 42;

void
allocate(exec::cpu s, topo::resize::Field::accessor<wo> a) {
  a = s.launch().index + 1;
}
void
irows(exec::cpu s, intN::mutator<wo> r) {
  r[0].resize(s.launch().index + 1);
}
int
drows(exec::cpu s, double_at::mutator<wo> mm) {
  UNIT("TASK") {
    const auto me = s.launch().index;
    const auto && m = mm[0];
    for(std::size_t c = 0; c <= me; ++c)
      m.try_emplace(column + c, me + c);
    for(const auto && p : m)
      EXPECT_EQ(p.first - column, p.second - me);
    // Exercise std::map-like interface:
    {
      const auto [i, nu] = m.insert({column, 0});
      EXPECT_FALSE(nu);
      EXPECT_EQ((*i).first, column);
      EXPECT_EQ((*i).second, me);
    }
    {
      const auto [i, nu] = m.try_emplace(0, 1);
      EXPECT_TRUE(nu);
      EXPECT_EQ((*i).first, 0u);
      EXPECT_EQ((*i).second, 1u);
    }
    {
      const auto [i, nu] = m.insert_or_assign(0, 2);
      EXPECT_FALSE(nu);
      EXPECT_EQ((*i).first, 0u);
      EXPECT_EQ((*i).second, 2u);
    }
    EXPECT_EQ(m.count(0), 1u);
    EXPECT_EQ(m.erase(0), 1u);
    EXPECT_EQ(m.count(0), 0u);
    EXPECT_EQ(m.erase(0), 0u);
    EXPECT_EQ((*m.find(column)).second, me);
    EXPECT_EQ(m.lower_bound(column), m.begin());
    EXPECT_EQ(m.upper_bound((*--m.end()).first), m.end());
  };
}

using noisy = field<Noisy, single>;
const noisy::definition<topo::index> noisy_field;

const field<int *>::definition<topo::index> ptr_field;

void
assign(exec::cpu s,
  double_field::accessor<wo> p,
  intN::accessor<rw> r,
  double_at::accessor<rw> sp) {
  const auto i = s.launch().index;
  flog(info) << "assign on " << i << std::endl;
  p = i;
  static_assert(std::is_same_v<decltype(r.get_offsets().span()),
    util::span<const std::size_t>>);
  r[0].back() = 1;
  ++sp[0](column + i);
} // assign

std::size_t
reset(noisy::accessor<wo>) { // must be an MPI task for correct total
  return Noisy::count;
}
void
use_ptr(field<int *>::accessor<wo>) {} // must be an MPI task

// The unnamed mutator still allocates according to the growth policy.
void
ragged_start(intN::accessor<ro> v, intN::mutator<wo>, buffers::Start mv) {
  assert(mv.span().size() == 2u);
  bool sent = false;
  (void)buffers::ragged(mv[0], true)(v, 0, sent);
}

int
ragged_xfer(intN::accessor<ro> v, intN::mutator<rw> g, buffers::Transfer mv) {
  buffers::ragged::read(g, mv[1], util::iota_view(0, 1));
  bool sent = false;
  (void)buffers::ragged{mv[0]}(v, 0, sent);
  return sent;
}

int
check(exec::cpu es,
  double_field::accessor<ro> p,
  intN::accessor<ro> r,
  intN::accessor<ro> g,
  double_at::accessor<ro> sp,
  noisy::accessor<ro> n) {
  UNIT("TASK") {
    const auto me = es.launch().index;
    flog(info) << "check on " << me << std::endl;
    ASSERT_EQ(p, me);
    EXPECT_GE(r.get_base().span().size(), r.span().size() * 2);
    ASSERT_EQ(r.size(), 1u);
    const auto s = r[0];
    static_assert(std::is_same_v<decltype(s), const util::span<const int>>);
    ASSERT_EQ(s.size(), me + 1);
    EXPECT_EQ(s.back(), 1);
    const auto sg = g[0];
    ASSERT_EQ(sg.size(), me ? me : es.launch().size);
    EXPECT_EQ(sg.back(), 1);
    ASSERT_EQ(sp.size(), 1u);
    const auto sr = sp[0];
    EXPECT_EQ(sr(column + me), 2 * me + 1);
    EXPECT_EQ(n.get().i, Noisy::value());
  };
} // print

int
part(exec::cpu s, short_part::mutator<wo> a) {
  UNIT("TASK") {
    const short pi[] = {3, 0, 1, 4, 0, 0, 0, 1, 0, 0, 5, 0};
    short sum = 0, chk = 0;
    unsigned digits = 0;
    for(auto x : pi) {
      a.insert(x);
      sum += x;
      if(x)
        ++digits;
    }
    EXPECT_EQ(a.size(), a.capacity());
    for(auto i = a.begin(), e = a.end(); i != e;)
      if(*i)
        ++i;
      else
        i = a.erase(i); // pi doesn't have 0s in it!
    EXPECT_EQ(a.size(), digits);
    for(auto & x : a)
      chk += x;
    EXPECT_EQ(sum, chk);
    for(auto i = a.end(), b = a.begin(); i != b;)
      sum -= *--i;
    EXPECT_FALSE(sum);

    a.clear();
    EXPECT_EQ(a.size(), 0u);
    const auto i1 = a.insert(1), i2 = a.insert(2), i3 = a.insert(3);
    EXPECT_EQ(a.size(), 3u);
    a.erase(i1);
    EXPECT_EQ(a.size(), 2u);
    a.erase(i2); // goes on the free list after i1
    EXPECT_EQ(a.size(), 1u);
    const auto i4 = a.insert(4);
    EXPECT_EQ(a.size(), 2u);
    EXPECT_EQ(*i3, 3);
    EXPECT_EQ(*i4, 4);
    EXPECT_EQ(i3, a.get_iterator_from_pointer(&*i3));
    *a.begin() = s.launch().index;
  };
}

// The MPI backend doesn't support non-trivial color mappings:
constexpr int process_fraction = 2 - (FLECSI_BACKEND == FLECSI_BACKEND_mpi ||
                                       FLECSI_BACKEND == FLECSI_BACKEND_hpx);

int
use_map(exec::cpu s,
  data::multi<short_part::accessor<ro>> ma,
  data::multi<intN::mutator<wo>> mm) {
  UNIT() {
    const auto p = processes(), nc = std::max(p / process_fraction, {1}),
               c = s.launch().index;
    EXPECT_EQ(s.launch().size, nc);
    const auto ac = ma.components();
    EXPECT_EQ(ac.size(), p / nc + (c < p % nc));
    for(auto [c, a] : ac)
      EXPECT_EQ(*a.begin(), c);
    const auto mc = mm.components();
    EXPECT_EQ(mc.size(), ac.size());
    for(auto [c, m] : mc)
      m[0].resize(c + 1, c);
  };
}

int
check_map(exec::cpu s, intN::accessor<ro> a) {
  UNIT() {
    const auto c = s.launch().index;
    ASSERT_EQ(a[0].size(), c + 1);
    EXPECT_EQ(a[0].back(), c);
  };
}

int
index_driver() {
  UNIT() {
    const auto np = processes();
    {
      region r({}, {});
      EXPECT_FALSE((r.ghost<privilege_pack<wo, wo, na>>(0)));
      EXPECT_TRUE((r.ghost<privilege_pack<ro, ro, ro>>(0)));
      EXPECT_FALSE((r.ghost<privilege_pack<ro, ro, ro>>(0)));
      EXPECT_FALSE((r.ghost<privilege_pack<ro, rw, rw>>(0)));
      EXPECT_FALSE((r.ghost<privilege_pack<ro, rw, ro>>(0)));
      EXPECT_FALSE((r.ghost<privilege_pack<ro, ro, na>>(0)));
      EXPECT_TRUE((r.ghost<privilege_pack<ro, rw, rw>>(0)));
    }

    Noisy::count = 0;
    constexpr static auto alloc = [](auto f) {
      auto & p = f.get_elements();
      p.growth = {0, 0, 0.25, 0.5, 1};
      execute<allocate>(exec::on, p.sizes());
      p.resize();
    };
    const auto pressure = pressure_field(process_topology);
    const auto verts = verts_field(process_topology),
               ghost = ghost_field(process_topology);
    const auto vfrac = vfrac_field(process_topology);
    const auto noise = noisy_field(process_topology);
    alloc(verts);
    alloc(vfrac);
    ghost.get_elements().growth = {processes() + 1};
    {
      // Use the mutator twice to record/replay the trace and to
      // make the new size visible to check below.
      exec::trace t;
      for(int i = 0; i < 2; i++)
        t.make_guard(), execute<irows>(exec::on, verts);
    }
    EXPECT_EQ(test<drows>(exec::on, vfrac), 0);
    execute<assign>(exec::on, pressure, verts, vfrac);
    execute<reset>(noise);
    EXPECT_EQ(
      (reduce<reset, exec::fold::sum, flecsi::mpi>(noise).get()), processes());
    execute<use_ptr, flecsi::mpi>(ptr_field(process_topology));

    // Rotate the ragged field by one color:
    buffers::core([] {
      const auto p = processes();
      buffers::coloring ret(p);
      Color i = 0;
      for(auto & g : ret)
        g.push_back(++i % p);
      return ret;
    }())
      .xfer<ragged_start, ragged_xfer>(verts, ghost);

    EXPECT_EQ(test<check>(exec::on, pressure, verts, ghost, vfrac, noise), 0);

    // Duplicate work to support the MPI backend:
    trivial_array::slot a;
    a.allocate(trivial_array::coloring(processes(), 12));
    EXPECT_EQ(test<part>(exec::on, particles(a)), 0);
    execute<allocate>(exec::on, arag(a).get_elements().sizes());
    arag(a).get_elements().resize();

    auto lm = launch::make(
      a, launch::robin(a.colors(), std::max(np / process_fraction, {1})));
    EXPECT_EQ(test<use_map>(exec::on, particles(lm), arag(lm)), 0);
    EXPECT_EQ(test<check_map>(exec::on, arag(a)), 0);
  };
} // index

util::unit::driver<index_driver> driver;
