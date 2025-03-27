#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/run/context.hh"
#include "flecsi/topo/unstructured/test/fixed.hh"
#include "flecsi/util/geometry/point.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;

struct spec_setopo_t : topo::specialization<topo::set, spec_setopo_t> {
  static constexpr unsigned int dimension = 2;

  using point_t = util::point<double, dimension>;

  using mesh_type = fixed_mesh;

  static coloring color(mesh_type::slot * ptr) {

    coloring c;

    c.ptr = ptr;
    c.counts = std::vector<std::size_t>(processes(), 100);
    return c;
  }
};

struct Particle {
  double pressure;
  std::size_t cgid;
};

using accessorm = spec_setopo_t::mesh_type::accessor<ro, ro, ro>;

using particle_field = field<Particle, data::particle>;
const particle_field::definition<spec_setopo_t> particles;

void
init_fields(accessorm m,
  field<util::gid>::accessor<ro, ro, ro> cids,
  particle_field::mutator<wo> particle_t_m) noexcept {
  for(auto c : m.cells()) {
    Particle p_t;
    p_t.cgid = cids[c];
    p_t.pressure = 1.0;
    particle_t_m.insert(p_t);
  }
}

void
print_test(particle_field::accessor<ro> particle_t) noexcept {
  std::stringstream ss;

  for(auto & p : particle_t) {
    ss << "pressure=" << p.pressure;
    ss << std::endl;
  }
  flog(info) << ss.rdbuf() << std::endl;
  std::cout << "before num_particle=" << particle_t.size() << ", "
            << "particle_cap=" << particle_t.capacity() << std::endl;
}

void
insert_test(accessorm m,
  field<util::gid>::accessor<ro, ro, ro> cids,
  particle_field::mutator<rw> particle_t_m) {
  forall(c, m.cells(), "insert_test") {
    if(cids[c] == 20) {
      Particle p_t{1.0, cids[c]};
      particle_t_m.insert(p_t);
    }
  };
}

void
update_test(particle_field::accessor<rw> particle_t) noexcept {

  for(auto & p : particle_t) {
    p.pressure = 2.0;
  }
  std::stringstream ss;
  for(auto & p : particle_t) {
    ss << "after update pressure=" << p.pressure;
    ss << std::endl;
  }

  flog(info) << ss.rdbuf() << std::endl;
}

int
set_driver(scheduler & s) {

  UNIT() {
    using mesh_type = spec_setopo_t::mesh_type;

    mesh_type::slot mesh_underlying;
    mesh_type::init fields;
    spec_setopo_t::slot spec_setopo;

    mesh_underlying.allocate(
      s, mesh_type::mpi_coloring(s, "simple-4x4.fixed", 4, fields), fields);
    spec_setopo.allocate(s, spec_setopo_t::mpi_coloring(s, &mesh_underlying));

    auto particle_t = particles(spec_setopo);
    s.execute<init_fields>(
      mesh_underlying, mesh_type::cid(mesh_underlying), particle_t);

    s.execute<print_test>(particle_t);
    execute<insert_test, default_accelerator>(
      mesh_underlying, mesh_type::cid(mesh_underlying), particle_t);
    s.execute<update_test>(particle_t);
  };

} // set_driver

util::unit::driver<set_driver> driver1;
