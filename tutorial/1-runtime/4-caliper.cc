#include <chrono>
#include <thread>

#include <flecsi/execution.hh>
#include <flecsi/run/control.hh>
#include <flecsi/util/annotation.hh>

using namespace flecsi;
using namespace flecsi::util;

struct user_execution : annotation::context<user_execution> {
  static constexpr char name[] = "User-Execution";
};

struct main_region : annotation::region<annotation::execution> {
  inline static const std::string name{"main"};
};

struct sleeper_region : annotation::region<user_execution> {
  inline static const std::string name{"sleeper"};
};

struct sleeper_subtask : annotation::region<user_execution> {
  inline static const std::string name{"subtask"};
  static constexpr annotation::detail detail_level = annotation::detail::high;
};

void
sleeper() {
  annotation::rguard<sleeper_region> guard;

  annotation::rguard<sleeper_subtask>(),
    std::this_thread::sleep_for(std::chrono::milliseconds(400));

  std::this_thread::sleep_for(std::chrono::milliseconds(300));
}

int
top_level_action() {

  sleeper();

  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  return 0;
}

// main
int
main(int argc, char * argv[]) {
  annotation::rguard<main_region> main_guard;

  run::arguments args(argc, argv);
  const run::dependencies_guard dg(args.dep);
  const runtime run(args.cfg);
  return (annotation::guard<annotation::execution, annotation::detail::low>(
            "top-level-task"),
    run.main<run::call>(args.act, top_level_action));
} // main
