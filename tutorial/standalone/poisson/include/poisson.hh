/*----------------------------------------------------------------------------*
  Copyright (c) 2020 Triad National Security, LLC
  All rights reserved
 *----------------------------------------------------------------------------*/

#ifndef POISSON_POISSON_HH
#define POISSON_POISSON_HH

#include <flecsi/execution.hh>
#include <flecsi/util/annotation.hh>

using namespace flecsi::util;

struct main_region : annotation::region<annotation::execution> {
  inline static const std::string name{"main"};
};
struct user_execution : annotation::context<user_execution> {
  static constexpr char name[] = "User-Execution";
};
struct problem_region : annotation::region<user_execution> {
  inline static const std::string name{"problem"};
};
struct solve_region : annotation::region<user_execution> {
  inline static const std::string name{"solve"};
};
struct analyze_region : annotation::region<user_execution> {
  inline static const std::string name{"analyze"};
};

#endif
