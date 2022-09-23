#include <chrono>
#include <thread>

#include <caliper/RegionProfile.h>

#include "flecsi/util/annotation.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;

std::size_t
check_custom(double val) {
  return val * 100;
}

struct test_context : util::annotation::context<test_context> {
  static constexpr char name[] = "test-context";
};

void
wait() {
  std::this_thread::sleep_for(std::chrono::milliseconds(40));
}

int
annotation_driver() {
  UNIT() {
    namespace ann = flecsi::util::annotation;

    auto & c = run::context::instance();
    auto rank = c.process();
    auto size = c.processes();

    cali::RegionProfile rp;
    rp.start();

    { // test custom annotation context/region
      {
        ann::guard<test_context, ann::detail::low> g("custom");
        std::this_thread::sleep_for(std::chrono::milliseconds(10 * (rank + 1)));
      }

      auto times = std::get<0>(rp.exclusive_region_times(test_context::name));

      auto custom_time = times.find("custom");
      ASSERT_NE(custom_time, times.end());
      auto combined_time =
        reduce<check_custom, exec::fold::sum, mpi>(custom_time->second).get();
      EXPECT_GE(combined_time, size * (size + 1) / 2);
    }

    execute<wait, mpi>();

    auto times = std::get<0>(rp.exclusive_region_times(ann::execution::name));
    auto wait_time =
      times.find(ann::execute_task_user::name + "->" + util::symbol<wait>());

    if constexpr(ann::execute_task_user::detail_level <=
                 ann::detail_level) { // test user execution
      ASSERT_NE(wait_time, times.end());
      auto elapsed = wait_time->second;
      EXPECT_GE(elapsed, 0.04);
    }
    else {
      // check user execution is not included
      ASSERT_EQ(wait_time, times.end());
    }

    { // test annotation in high detail level
      {
        ann::guard<test_context, ann::detail::high> g("high");
        std::this_thread::sleep_for(std::chrono::milliseconds(10 * (rank + 1)));
      }
      auto times = std::get<0>(rp.exclusive_region_times(test_context::name));

      auto custom_time = times.find("high");
      if constexpr(ann::detail_level == ann::detail::high) {
        ASSERT_NE(custom_time, times.end());
        auto combined_time =
          reduce<check_custom, exec::fold::sum, mpi>(custom_time->second).get();
        EXPECT_GE(combined_time, size * (size + 1) / 2);
      }
      else { // test timer is not included
        ASSERT_EQ(custom_time, times.end());
      }
    }
  };
} // annotation_driver

unit::driver<annotation_driver> driver;
