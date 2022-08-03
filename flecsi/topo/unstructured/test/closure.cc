#include "simple_definition.hh"
#include "ugm_definition.hh"

#include "flecsi/execution.hh"
#include "flecsi/flog.hh"
#include "flecsi/topo/unstructured/coloring_utils.hh"
#include "flecsi/topo/unstructured/types.hh"
#include "flecsi/util/parmetis.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;

struct closure_policy {

  using primary = topo::unstructured_impl::primary_independent<0, 2, 0, 1>;
  // using primary = topo::unstructured_impl::primary_independent<0, 2, 0, 2>;

  using auxiliary =
    std::tuple<topo::unstructured_impl::auxiliary_independent<1, 0, 2>>;

  static constexpr size_t auxiliary_colorings =
    std::tuple_size<auxiliary>::value;
}; // coloring_policy

#if 0
struct staging_area {
  std::vector<std::size_t> raw;
  std::map<std::size_t, std::vector<std::size_t>> primaries;
}; // struct staging_area

staging_area staging;
#endif

void
compute_closure() {
#if 1
  const Color colors = 4;
  topo::unstructured_impl::simple_definition sd("simple2d-8x8.msh");
  // const Color colors = 6;
  // topo::unstructured_impl::simple_definition sd("simple2d-16x16.msh");
#else
  const Color colors = 6;
  topo::unstructured_impl::ugm_definition sd("bunny.ugm");
#endif

  auto [naive, ge, c2v, v2c, c2c] = topo::unstructured_impl::make_dcrs(sd, 1);
  auto raw = util::parmetis::color(naive, colors);

#if 0
    {
      std::stringstream ss;
      ss << "v2c:" << std::endl;
      for(auto v : v2c) {
        ss << "vertex " << v.first << ": ";
        for(auto c : v.second) {
          ss << c << " ";
        } // for
        ss << std::endl;
      } // for
      flog(error) << ss.str() << std::endl;
    } // scope
#endif

#if 0
    {
      std::stringstream ss;
      ss << "raw:" << std::endl;
      for(auto r : raw) {
        ss << r << " ";
      }
      ss << std::endl;
      flog(warn) << ss.str();
    } // scope
#endif

#if 0
    auto primaries = topo::unstructured_impl::distribute(naive, colors, raw);

    {
      std::stringstream ss;
      Color color=0;
      for(auto p : primaries) {
        ss << "color " << color++ << ":" << std::endl;
        for(auto i : p) {
          ss << i << " ";
        }
        ss << std::endl;
      }
      ss << std::endl;
      flog_devel(warn) << ss.str();
    } // scope

    for(auto p : primaries) {
      topo::unstructured_impl::closure<closure_policy>(p);
    }
#endif

  auto [primaries, p2m, m2p] =
    topo::unstructured_impl::migrate(naive, colors, raw, c2v, v2c, c2c);

#if 0
    flog(info) << "PRIMARIES" << std::endl;
    for(auto co : primaries) {
      std::stringstream ss;
      ss << "Color: " << co.first << ": ";
      for(auto c : co.second) {
        ss << c << " ";
      } // for
      flog(warn) << ss.str() << std::endl;
    } // for

    flog(info) << "V2C CONNECTIVITIES" << std::endl;
    for(auto const & v : v2c) {
      std::stringstream ss;
      ss << "vertex " << v.first << ": ";
      for(auto const & c : v.second) {
        ss << c << " ";
      } // for
      flog(warn) << ss.str() << std::endl;
    } // for

    flog(info) << "C2C CONNECTIVITIES" << std::endl;
    for(auto const & c : c2c) {
      std::stringstream ss;
      ss << "cell " << c.first << ": ";
      for(auto const & cc : c.second) {
        ss << cc << " ";
      } // for
      flog(warn) << ss.str() << std::endl;
    } // for

    flog(info) << "CELL DEFINITIONS" << std::endl;
    std::size_t lid{0};
    for(auto const & c : c2v) {
      std::stringstream ss;
      ss << "cell " << p2m[lid++] << ": ";
      for(auto const & v : c) {
        ss << v << " ";
      } // for
      flog(warn) << ss.str() << std::endl;
    } // for
#endif

#if 1
  auto colorings = topo::unstructured_impl::closure<closure_policy>(
    sd, colors, raw, primaries, c2v, v2c, c2c, m2p, p2m);
#endif

#if 0
    flog(info) << "V2C CONNECTIVITIES" << std::endl;
    for(auto const & v : v2c) {
      std::stringstream ss;
      ss << "vertex " << v.first << ": ";
      for(auto const & c : v.second) {
        ss << c << " ";
      } // for
      flog(warn) << ss.str() << std::endl;
    } // for

    flog(info) << "C2C CONNECTIVITIES" << std::endl;
    for(auto const & c : c2c) {
      std::stringstream ss;
      ss << "cell " << c.first << ": ";
      for(auto const & cc : c.second) {
        ss << cc << " ";
      } // for
      flog(warn) << ss.str() << std::endl;
    } // for

    flog(info) << "CELL DEFINITIONS" << std::endl;
    std::size_t lid{0};
    for(auto const & c : c2v) {
      std::stringstream ss;
      ss << "cell " << p2m[lid++] << ": ";
      for(auto const & v : c) {
        ss << v << " ";
      } // for
      flog(warn) << ss.str() << std::endl;
    } // for
#endif
}

int
closure_driver() {
  UNIT { execute<compute_closure, mpi>(); };
}

flecsi::unit::driver<closure_driver> driver;
