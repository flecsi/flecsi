#ifndef TUTORIAL_6_TOPOLOGY_CONTROL_HH
#define TUTORIAL_6_TOPOLOGY_CONTROL_HH

#include "ntree_sph.hh"

namespace sph {

// Program options with default values
inline flecsi::program_option<std::size_t> n_entities("#entities",
  "ents,e",
  "The number of entities.",
  {{flecsi::option_default, 100}});
inline flecsi::program_option<std::size_t> n_iterations("#iterations",
  "iters,i",
  "The number of iterations.",
  {{flecsi::option_default, 10}});

enum class cp { initialize, iterate, output, finalize };

inline const char *
operator*(cp control_point) {
  switch(control_point) {
    case cp::initialize:
      return "initialize";
    case cp::iterate:
      return "iterate";
    case cp::output:
      return "output";
    case cp::finalize:
      return "finalize";
  }
  flog_fatal("invalid control point");
}

struct control_policy : flecsi::run::control_base {

  sph_ntree_t::slot sph_ntree;
  std::size_t max_iterations = 100;
  std::size_t step = 0;
  std::size_t intv = 100;

  using control_points_enum = cp;

  static bool cycle_control(control_policy & policy) {
    return policy.step++ < policy.max_iterations;
  }
  using main_cycle =
    cycle<cycle_control, point<cp::iterate>, point<cp::output>>;

  using control_points = list<point<cp::initialize>, main_cycle>;
};

using control = flecsi::run::control<control_policy>;
} // namespace sph

#endif
