#include "flecsi/topo/unstructured/test/fixed.hh"
#include "flecsi/config.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;

const field<int>::definition<fixed_mesh, fixed_mesh::cells> pressure;
const field<double>::definition<fixed_mesh, fixed_mesh::vertices> density;

void
init_pressure(fixed_mesh::accessor<ro, ro, ro> m,
  field<int>::accessor<wo, wo, wo> p) noexcept {
  for(auto c : m.cells()) {
    static_assert(std::is_same_v<decltype(c), topo::id<fixed_mesh::cells>>);
    p[+c] = -1;
  }
}

void
update_pressure(exec::accelerator s,
  fixed_mesh::accessor<ro, ro, ro> m,
  field<int>::accessor<rw, rw, rw> p) noexcept {
  int clr = s.launch().index;
  s.executor().forall(c, m.cells()) {
    p[c] = clr;
  };
}

void
check_pressure(data::multi<fixed_mesh::accessor<ro, ro, ro>> mm,
  data::multi<field<int>::accessor<ro, ro, ro>> pp) noexcept {
  const auto pc = pp.components();
  auto i = pc.begin();
  for(auto [clr, m] : mm.components()) {
    auto [clr2, p] = *i++;
    flog_assert(clr == clr2, "color mismatch");
    for(auto c : m.cells()) {
      unsigned int v = p[c];
      flog_assert(v == clr, "invalid pressure");
    }
  }
}

void
init_density(fixed_mesh::accessor<ro, ro, ro> m,
  field<double>::accessor<wo, wo, wo> d) noexcept {
  for(auto c : m.vertices()) {
    d[c] = -1;
  }
}

void
update_density(exec::accelerator s,
  fixed_mesh::accessor<ro, ro, ro> m,
  field<double>::accessor<rw, rw, rw> d) noexcept {
  auto clr = s.launch().index;
  s.executor().forall(v, m.vertices()) {
    d[v] = clr;
  };
}

void
check_density(exec::cpu s,
  fixed_mesh::accessor<ro, ro, ro> m,
  field<double>::accessor<ro, ro, ro> d) noexcept {
  auto clr = s.launch().index;
  for(auto c : m.vertices()) {
    unsigned int v = d[c];
    flog_assert(v == clr, "invalid pressure");
  }
}

int
verify_mesh(exec::cpu s,
  fixed_mesh::accessor<ro, ro, ro> m,
  field<util::gid>::accessor<ro, ro, ro> cids,
  field<util::gid>::accessor<ro, ro, ro> vids) noexcept {
  UNIT("TASK") {
    auto & out = UNIT_CAPTURE();

    for(auto c : m.cells()) {
      out << "cell(" << cids[c] << ',' << c << "):";
      for(auto v : m.vertices(c)) {
        out << ' ' << vids[v];
      }
      out << '\n';
    }

    auto print_ids = [&](std::string key, auto & f, auto && s) {
      out << key;
      for(auto i : s) {
        out << ' ' << f[i];
      }
      out << '\n';
    };

    print_ids("cells(owned):", cids, m.cells<fixed_mesh::owned>());
    print_ids("cells(shared):", cids, m.cells<fixed_mesh::shared>());
    print_ids("cells(ghost):", cids, m.cells<fixed_mesh::ghost>());
    out << '\n';

    for(auto v : m.vertices()) {
      out << "vertex(" << vids[v] << ',' << v << "):";
      for(auto c : m.cells(v)) {
        out << ' ' << cids[c];
      }
      out << '\n';
    }

    print_ids("vertex(owned):", vids, m.vertices<fixed_mesh::owned>());
    print_ids("vertex(shared):", vids, m.vertices<fixed_mesh::shared>());
    print_ids("vertex(ghost):", vids, m.vertices<fixed_mesh::ghost>());

    EXPECT_TRUE(UNIT_EQUAL_BLESSED(
      "fixed_" + std::to_string(s.launch().index) + ".blessed"));
  };
}

static data::launch::Claims
rotate(Color n) {
  data::launch::Claims ret(n);
  for(Color i = 0; i < n; ++i)
    ret[(i + n -
          (FLECSI_BACKEND != FLECSI_BACKEND_mpi &&
            FLECSI_BACKEND != FLECSI_BACKEND_hpx)) %
        n]
      .push_back(i);
  return ret;
}

int
fixed_driver(scheduler & s) {
  UNIT() {
    fixed_mesh::init fields;
    fixed_mesh::ptr m;
    auto & mesh = s.allocate(m,
      fixed_mesh::mpi_coloring(s, s.runtime(), "simple-4x4.fixed", 4, fields),
      fields);

    EXPECT_EQ(s.test<verify_mesh>(
                exec::on, mesh, fixed_mesh::cid(mesh), fixed_mesh::vid(mesh)),
      0);

    s.execute<init_pressure>(mesh, pressure(mesh));
    s.execute<update_pressure>(exec::on, mesh, pressure(mesh));
    auto lm = data::launch::make(s, mesh, rotate(mesh.colors()));
    s.execute<check_pressure>(lm, pressure(lm));

    s.execute<init_density>(mesh, density(mesh));
    s.execute<update_density>(exec::on, mesh, density(mesh));
    s.execute<check_density>(exec::on, mesh, density(mesh));
  };
}

util::unit::driver<fixed_driver> driver;
