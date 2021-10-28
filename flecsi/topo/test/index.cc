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

#include "flecsi/util/demangle.hh"
#include "flecsi/util/unit.hh"
#include <flecsi/data.hh>
#include <flecsi/execution.hh>

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
  static inline std::size_t count;
};

using double_field = field<double, single>;
const double_field::definition<topo::index> pressure_field;
using intN = field<int, ragged>;
const intN::definition<topo::index> verts_field, ghost_field;
using double_at = field<double, sparse>;
const double_at::definition<topo::index> vfrac_field;

constexpr std::size_t column = 42;

void
allocate(topo::resize::Field::accessor<wo> a) {
  a = color() + 1;
}
void
irows(intN::mutator<wo> r) {
  r[0].resize(color() + 1);
}
int
drows(double_at::mutator<wo> s) {
  UNIT {
    const auto me = color();
    const auto && m = s[0];
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

void
assign(double_field::accessor<wo> p,
  intN::accessor<rw> r,
  double_at::accessor<rw> sp) {
  const auto i = color();
  flog(info) << "assign on " << i << std::endl;
  p = i;
  static_assert(std::is_same_v<decltype(r.get_offsets().span()),
    util::span<const std::size_t>>);
  r[0].back() = 1;
  ++sp[0](column + i);
} // assign

std::size_t reset(noisy::accessor<wo>) { // must be an MPI task
  return Noisy::count;
}

// The unnamed mutator still allocates according to the growth policy.
void
ragged_start(intN::accessor<ro> v, intN::mutator<wo>, buffers::Start mv) {
  assert(mv.span().size() == 2u);
  buffers::ragged::truncate(mv[0])(v, 0);
}

int
ragged_xfer(intN::accessor<ro> v, intN::mutator<rw> g, buffers::Transfer mv) {
  buffers::ragged::read(g, mv[1], [](std::size_t) { return 0; });
  return !buffers::ragged{mv[0]}(v, 0);
}

int
check(double_field::accessor<ro> p,
  intN::accessor<ro> r,
  intN::accessor<ro> g,
  double_at::accessor<ro> sp,
  noisy::accessor<ro> n) {
  UNIT {
    const auto me = color();
    flog(info) << "check on " << me << std::endl;
    ASSERT_EQ(p, me);
    EXPECT_GE(r.get_base().span().size(), r.span().size() * 2);
    ASSERT_EQ(r.size(), 1u);
    const auto s = r[0];
    static_assert(std::is_same_v<decltype(s), const util::span<const int>>);
    ASSERT_EQ(s.size(), me + 1);
    EXPECT_EQ(s.back(), 1);
    const auto sg = g[0];
    ASSERT_EQ(sg.size(), me ? me : colors());
    EXPECT_EQ(sg.back(), 1);
    ASSERT_EQ(sp.size(), 1u);
    const auto sr = sp[0];
    EXPECT_EQ(sr(column + me), 2 * me + 1);
    EXPECT_EQ(n.get().i, Noisy::value());
  };
} // print

int
index_driver() {
  UNIT {
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
    for(const auto f : {verts_field.fid, vfrac_field.fid}) {
      auto & p = process_topology->ragged.get_partition<topo::elements>(f);
      p.growth = {0, 0, 0.25, 0.5, 1};
      execute<allocate>(p.sizes());
    }
    process_topology->ragged.get_partition<topo::elements>(ghost_field.fid)
      .growth = {processes() + 1};
    const auto pressure = pressure_field(process_topology);
    const auto verts = verts_field(process_topology),
               ghost = ghost_field(process_topology);
    const auto vfrac = vfrac_field(process_topology);
    const auto noise = noisy_field(process_topology);
    execute<irows>(verts);
    execute<irows>(verts); // to make new size visible
    execute<drows>(vfrac);
    execute<assign>(pressure, verts, vfrac);
    execute<reset>(noise);
    EXPECT_EQ(
      (reduce<reset, exec::fold::sum, flecsi::mpi>(noise).get()), processes());

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

    EXPECT_EQ(test<check>(pressure, verts, ghost, vfrac, noise), 0);
  };
} // index

flecsi::unit::driver<index_driver> driver;
