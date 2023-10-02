#ifndef FLECSI_RUN_TEST_PACKAGE_C_HH
#define FLECSI_RUN_TEST_PACKAGE_C_HH

#include "cycle.hh"
#include "package_a.hh"
#include "package_b.hh"

#include "flecsi/flog.hh"

namespace cycle {
namespace common {
namespace package_c {

inline void
advance(control_policy &) {
  flog(info) << "C advance" << std::endl;
} // advance

inline control::action<advance, cp::advance> advance_action;

inline const auto dep_bc = package_b::advance_action.add(advance_action);
inline const auto dep_ca = advance_action.add(ns1::package_a::advance_action);

inline void
analyze(control_policy &) {
  flog(info) << "C analyze" << std::endl;
} // analyze

inline control::action<analyze, cp::analyze> analyze_action;
inline const auto dep_a = analyze_action.add(ns1::package_a::analyze_action);

} // namespace package_c
} // namespace common
} // namespace cycle

#endif
