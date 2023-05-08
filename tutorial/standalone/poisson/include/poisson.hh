#ifndef POISSON_POISSON_HH
#define POISSON_POISSON_HH

#include <flecsi/execution.hh>
#include <flecsi/util/annotation.hh>

struct main_region
  : flecsi::util::annotation::region<flecsi::util::annotation::execution> {
  inline static const std::string name{"main"};
};
struct user_execution : flecsi::util::annotation::context<user_execution> {
  static constexpr char name[] = "User-Execution";
};
struct problem_region : flecsi::util::annotation::region<user_execution> {
  inline static const std::string name{"problem"};
};
struct solve_region : flecsi::util::annotation::region<user_execution> {
  inline static const std::string name{"solve"};
};
struct analyze_region : flecsi::util::annotation::region<user_execution> {
  inline static const std::string name{"analyze"};
};

#endif
