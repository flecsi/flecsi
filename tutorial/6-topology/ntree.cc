#include "control.hh"

using namespace flecsi;

const field<double>::definition<sph_ntree_t> density, pressure, energy,
  d_energy, velocity, acceleration, sound_speed;
const field<bool>::definition<sph_ntree_t> is_wall;

void
density_task(sph_ntree_t::accessor<ro, ro> t,
  field<double>::accessor<wo, ro> rho) {
  sph::density(t, rho.span());
}

void
eos_task(sph_ntree_t::accessor<ro, ro> t,
  field<double>::accessor<wo, na> p,
  field<double>::accessor<ro, na> rho,
  field<double>::accessor<ro, na> u) {
  sph::eos(t, u.span(), rho.span(), p.span());
}

void
acceleration_task(sph_ntree_t::accessor<ro, ro> t,
  field<double>::accessor<wo, na> dvdt,
  field<double>::accessor<ro, ro> v,
  field<double>::accessor<ro, ro> rho,
  field<double>::accessor<ro, ro> p,
  field<double>::accessor<ro, ro> u) {
  sph::acceleration(t, v.span(), rho.span(), p.span(), u.span(), dvdt.span());
}

void
dudt_task(sph_ntree_t::accessor<ro, ro> t,
  field<double>::accessor<wo, na> dudt,
  field<double>::accessor<ro, ro> v,
  field<double>::accessor<ro, ro> rho,
  field<double>::accessor<ro, ro> p,
  field<double>::accessor<ro, ro> u,
  field<bool>::accessor<ro, ro> is_w) {
  sph::dudt(
    t, v.span(), rho.span(), p.span(), u.span(), is_w.span(), dudt.span());
}

// init_sodtube_task
void
init_sodtube_task(sph_ntree_t::accessor<ro, na> t,
  field<double>::accessor<wo, na> rho,
  field<double>::accessor<wo, na> p,
  field<double>::accessor<wo, na> v,
  field<double>::accessor<wo, na> u,
  field<bool>::accessor<wo, na> is_w) {
  sph::init_physics(t, v.span(), rho.span(), p.span(), u.span(), is_w.span());
} // init_sodtube_task

void
advance_task(sph_ntree_t::accessor<rw, wo> t,
  field<double>::accessor<rw, wo> v,
  field<double>::accessor<ro, ro> dvdt,
  field<double>::accessor<rw, wo> u,
  field<double>::accessor<ro, ro> dudt,
  field<bool>::accessor<ro, wo> is_w) {
  sph::advance(t, v.span(), dvdt.span(), u.span(), dudt.span(), is_w.span());
}

void
initialize_action(sph::control_policy & cp) {

  {
    std::vector<sph_ntree_t::ent_t> ents;
    const int nents = sph::n_entities.value();
    sph_ntree_t::mpi_coloring coloring(nents, ents);
    cp.sph_ntree.allocate(coloring, ents);
  }

  cp.max_iterations = sph::n_iterations.value();
  auto rho = density(cp.sph_ntree);
  auto p = pressure(cp.sph_ntree);
  auto v = velocity(cp.sph_ntree);
  auto u = energy(cp.sph_ntree);
  auto is_w = is_wall(cp.sph_ntree);
  flecsi::execute<init_sodtube_task>(cp.sph_ntree, rho, p, v, u, is_w);
}

void
output_task(sph_ntree_t::accessor<ro, ro> t,
  field<double>::accessor<ro, ro> rho,
  field<double>::accessor<ro, ro> p,
  field<double>::accessor<ro, ro> v,
  field<double>::accessor<ro, ro> u,
  field<double>::accessor<ro, ro> dudt,
  field<double>::accessor<ro, ro> dvdt,
  int nfile) {
  std::ofstream file("output_sodtube_" + std::to_string(nfile) + '_' +
                     std::to_string(process()) + ".dat");
  file << "x rho p v u dudt dvdt\n";
  for(auto e : t.entities()) {
    file << t.e_i[e].coordinates[0] << ' ' << rho[e] << ' ' << p[e] << ' '
         << v[e] << ' ' << u[e] << ' ' << dudt[e] << ' ' << dvdt[e] << '\n';
  }
#if defined(FLECSI_ENABLE_GRAPHVIZ)
  // This allows to draw a representation of the N-Tree using graphviz.
  t.graphviz_draw(nfile);
#endif
}

void
merge_output(int mi) {
  for(int i = 0; i < mi; i += 100) {
    std::ostringstream filename;
    filename << "output_sodtube_" << std::setfill('0') << std::setw(4) << i
             << ".dat";
    std::ofstream file(std::move(filename).str());
    for(unsigned int c = 0; c < processes(); ++c) {
      std::string fname = "output_sodtube_" + std::to_string(i) + '_' +
                          std::to_string(c) + ".dat";
      file << std::ifstream(fname).rdbuf();
      std::remove(fname.c_str());
    }
  }
}

void
iterate_action(sph::control_policy & cp) {
  auto rho = density(cp.sph_ntree);
  auto p = pressure(cp.sph_ntree);
  auto u = energy(cp.sph_ntree);
  auto v = velocity(cp.sph_ntree);
  auto dvdt = acceleration(cp.sph_ntree);
  auto dudt = d_energy(cp.sph_ntree);
  auto is_w = is_wall(cp.sph_ntree);

  flog(info) << "Iteration: " << cp.step << '\n';
  flecsi::execute<density_task>(cp.sph_ntree, rho);
  flecsi::execute<eos_task>(cp.sph_ntree, p, rho, u);
  flecsi::execute<acceleration_task>(cp.sph_ntree, dvdt, v, rho, p, u);
  flecsi::execute<dudt_task>(cp.sph_ntree, dudt, v, rho, p, u, is_w);
  flecsi::execute<advance_task>(cp.sph_ntree, v, dvdt, u, dudt, is_w);
  sph_ntree_t::sph_reset(cp.sph_ntree);
}

void
output_action(sph::control_policy & cp) {
  if(cp.step % 100 == 0) {
    auto rho = density(cp.sph_ntree);
    auto p = pressure(cp.sph_ntree);
    auto u = energy(cp.sph_ntree);
    auto v = velocity(cp.sph_ntree);
    auto dvdt = acceleration(cp.sph_ntree);
    auto dudt = d_energy(cp.sph_ntree);

    flecsi::execute<output_task>(
      cp.sph_ntree, rho, p, v, u, dudt, dvdt, cp.step);
  }
}

void
finalize_action(sph::control_policy & cp) {
  if(process() == 0)
    merge_output(cp.max_iterations);
}

sph::control::action<initialize_action, sph::cp::initialize> init;
sph::control::action<iterate_action, sph::cp::iterate> it;
sph::control::action<output_action, sph::cp::output> out;
sph::control::action<finalize_action, sph::cp::finalize> fin;

int
main(int argc, char ** argv) {
  flecsi::getopt()(argc, argv);
  const flecsi::run::dependencies_guard dg;
  const flecsi::runtime run;
  flecsi::flog::add_output_stream("clog", std::clog, true);
  return run.control<sph::control>();
}
