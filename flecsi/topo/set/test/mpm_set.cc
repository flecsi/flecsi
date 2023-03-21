#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/run/context.hh"
#include "flecsi/topo/unstructured/test/unstructured.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;

unstructured::slot mesh_underlying;
unstructured::cslot mesh_coloring;

struct spec_setopo_t : topo::specialization<topo::set, spec_setopo_t> {
  static constexpr unsigned int dimension = 2;

  using point_t = util::point<double, dimension>;

  typedef unstructured mesh_type;

  static coloring color(unstructured::slot * ptr) {

    coloring c;

    c.ptr = ptr;
    c.counts = std::vector<std::size_t>(processes(), 100);
    return c;
  }
};

spec_setopo_t::slot spec_setopo;
spec_setopo_t::cslot set_coloring;

struct Particle {
  double pressure;
  std::size_t cgid;
};

using accessorm = unstructured::accessor<ro, ro, ro>;

using particle_field = field<Particle, data::particle>;
const particle_field::definition<spec_setopo_t> particles;

void
init_fields(accessorm m,
  field<util::gid>::accessor<ro, ro, ro> cids,
  particle_field::mutator<wo> particle_t_m) {
  for(auto c : m.cells()) {
    Particle p_t;
    p_t.cgid = cids[c];
    p_t.pressure = 1.0;
    particle_t_m.insert(p_t);
  }
}

void
print_test(particle_field::accessor<ro> particle_t) {
  std::stringstream ss;

  for(auto & p : particle_t) {
    ss << "pressure=" << p.pressure;
    ss << std::endl;
  }
  flog(info) << ss.str() << std::endl;
  std::cout << "before num_particle=" << particle_t.size()
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
update_test(particle_field::accessor<rw> particle_t) {

  for(auto & p : particle_t) {
    p.pressure = 2.0;
  }
  std::stringstream ss;
  for(auto & p : particle_t) {
    ss << "after update pressure=" << p.pressure;
    ss << std::endl;
  }

  flog(info) << ss.str() << std::endl;
}

int
set_driver() {

  UNIT() {
    unstructured::init fields;
    mesh_coloring.allocate("simple2d-16x16.msh", fields);
    mesh_underlying.allocate(mesh_coloring.get(), fields);
    set_coloring.allocate(&mesh_underlying);
    spec_setopo.allocate(set_coloring.get());

    auto const & cids = mesh_underlying->forward_map<unstructured::cells>();
    auto particle_t = particles(spec_setopo);
    execute<init_fields>(mesh_underlying, cids(mesh_underlying), particle_t);

    execute<print_test>(particle_t);
    execute<insert_test, default_accelerator>(
      mesh_underlying, cids(mesh_underlying), particle_t);
    execute<update_test>(particle_t);
  };

} // set_driver

util::unit::driver<set_driver> driver1;
