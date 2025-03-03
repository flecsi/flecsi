#include "flecsi/topo/unstructured/test/fixed.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;

const field<int>::definition<fixed_mesh, fixed_mesh::cells> pressure;
const field<double>::definition<fixed_mesh, fixed_mesh::vertices> density;

void
init_pressure(fixed_mesh::accessor<ro, ro, ro> m,
  field<int>::accessor<wo, wo, wo> p) {
  for(auto c : m.cells()) {
    static_assert(std::is_same_v<decltype(c), topo::id<fixed_mesh::cells>>);
    p[c] = -1;
  }
}

void
update_pressure(fixed_mesh::accessor<ro, ro, ro> m,
  field<int>::accessor<rw, rw, rw> p) {
  int clr = color();
  forall(c, m.cells(), "pressure_c") { p[c] = clr; };
}

void
check_pressure(data::multi<fixed_mesh::accessor<ro, ro, ro>> mm,
  data::multi<field<int>::accessor<ro, ro, ro>> pp) {
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
  field<double>::accessor<wo, wo, wo> d) {
  for(auto c : m.vertices()) {
    d[c] = -1;
  }
}

void
update_density(fixed_mesh::accessor<ro, ro, ro> m,
  field<double>::accessor<rw, rw, rw> d) {
  auto clr = color();
  forall(v, m.vertices(), "density_c") { d[v] = clr; };
}

void
check_density(fixed_mesh::accessor<ro, ro, ro> m,
  field<double>::accessor<ro, ro, ro> d) {
  auto clr = color();
  for(auto c : m.vertices()) {
    unsigned int v = d[c];
    flog_assert(v == clr, "invalid pressure");
  }
}

int
verify_mesh(fixed_mesh::accessor<ro, ro, ro> m,
  field<util::gid>::accessor<ro, ro, ro> cids,
  field<util::gid>::accessor<ro, ro, ro> vids) {
  UNIT("TASK") {
    auto & out = UNIT_CAPTURE();

    for(auto c : m.cells()) {
      out << "cell(" << cids[c] << "," << c << "):";
      for(auto v : m.vertices(c)) {
        out << " " << vids[v];
      }
      out << "\n";
    }
    out << "\n";

    for(auto v : m.vertices()) {
      out << "vertex(" << vids[v] << "," << v << "):";
      for(auto c : m.cells(v)) {
        out << " " << cids[c];
      }
      out << "\n";
    }
    out << "\n";
    EXPECT_TRUE(
      UNIT_EQUAL_BLESSED("fixed_" + std::to_string(color()) + ".blessed"));
  };
}

static data::launch::Claims
rotate(Color n) {
  data::launch::Claims ret(n);
  for(Color i = 0; i < n; ++i)
    ret[(i + n - (FLECSI_BACKEND != FLECSI_BACKEND_mpi)) % n].push_back(i);
  return ret;
}

int
fixed_driver() {
  UNIT() {
    fixed_mesh::slot mesh;
    fixed_mesh::init fields;
    mesh.allocate(
      fixed_mesh::mpi_coloring("simple-4x4.fixed", 4, fields), fields);

    EXPECT_EQ(
      test<verify_mesh>(mesh, fixed_mesh::cid(mesh), fixed_mesh::vid(mesh)), 0);

    execute<init_pressure>(mesh, pressure(mesh));
    execute<update_pressure, default_accelerator>(mesh, pressure(mesh));
    auto lm = data::launch::make(mesh, rotate(mesh.colors()));
    execute<check_pressure>(lm, pressure(lm));

    execute<init_density>(mesh, density(mesh));
    execute<update_density, default_accelerator>(mesh, density(mesh));
    execute<check_density>(mesh, density(mesh));

    std::swap(mesh, mesh);
  };
}

util::unit::driver<fixed_driver> driver;
