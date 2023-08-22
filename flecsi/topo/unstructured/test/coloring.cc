#include "simple_definition.hh"

#include "flecsi/execution.hh"
#include "flecsi/flog.hh"
#include "flecsi/topo/unstructured/coloring_utils.hh"
#include "flecsi/topo/unstructured/interface.hh"
#include "flecsi/util/parmetis.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;
using namespace flecsi::topo::unstructured_impl;

int
parmetis_coloring() {
  UNIT("TASK") {
    using util::mpi::test;
    simple_definition sd("simple2d-16x16.msh");
    ASSERT_EQ(sd.dimension(), 2lu);
    ASSERT_EQ(sd.num_entities(0), 289lu);
    ASSERT_EQ(sd.num_entities(2), 256lu);

    // Coloring with 5 colors with MPI_COMM_WORLD
    {
      const Color colors = 5;

      coloring_utils cu(
        &sd, {colors, {2 /*id*/, 0 /*idx*/}, 1, {0, 1}, {{1, 2}}}, {});
      auto const naive = cu.color_primaries(1, util::parmetis::color);

      {
        std::stringstream ss;
        ss << "raw: " << std::endl;
        for(auto r : cu.primary_raw()) {
          ss << r << " ";
        }
        ss << std::endl;
        flog_devel(info) << ss.rdbuf();
      } // scope

      cu.migrate_primaries();
      auto const & cnns = cu.primary_connectivity_state();

      std::vector<size_t> distribution = {52, 103, 154, 205, 256};
      ASSERT_EQ(
        util::offsets(util::equal_map(sd.num_entities(2), colors)).ends(),
        distribution);

      UNIT_CAPTURE() << flog::container(naive.offsets.ends()) << '\n'
                     << flog::container(naive.values) << '\n'
                     << flog::container(cnns.m2p) << '\n';

      EXPECT_TRUE(
        UNIT_EQUAL_BLESSED(("coloring_5." + std::to_string(processes()) + "." +
                            std::to_string(process()) + ".blessed")));
    } // scope

    // Coloring with 5 colors with custom communicator with 2 processes
    {
      const Color colors = 5;

      const auto c2 = util::mpi::comm::split(
        MPI_COMM_WORLD, process() < 2 ? 0 : MPI_UNDEFINED);

      if(c2) {
        auto processes = util::mpi::size(c2.c);

        coloring_utils cu(
          &sd, {colors, {2 /*id*/, 0 /*idx*/}, 1, {0, 1}, {{1, 2}}}, {}, c2.c);
        cu.color_primaries(1, util::parmetis::color);

        {
          std::stringstream ss;
          ss << "raw: " << std::endl;
          for(auto r : cu.primary_raw()) {
            ss << r << " ";
          }
          ss << std::endl;
          flog_devel(info) << ss.rdbuf();
        } // scope

        cu.migrate_primaries();
        auto const & cnns = cu.primary_connectivity_state();

        UNIT_CAPTURE() << cu.ours().front() << '\n'
                       << flog::container(cu.primaries()) << '\n'
                       << flog::container(cnns.m2p) << '\n';
        EXPECT_TRUE(
          UNIT_EQUAL_BLESSED(("coloring_5." + std::to_string(processes) + '.' +
                              std::to_string(process()) + ".blessed")));
      } // if
    } // scope

    // Coloring with 1 color with 5 processes
    {
      const Color colors = 1;

      coloring_utils cu(
        &sd, {colors, {2 /*id*/, 0 /*idx*/}, 1, {0, 1}, {{1, 2}}}, {});
      cu.color_primaries(1, util::parmetis::color);
      auto const naive = cu.color_primaries(1, util::parmetis::color);

      {
        std::stringstream ss;
        ss << "raw: " << std::endl;
        for(auto r : cu.primary_raw()) {
          ss << r << " ";
        }
        ss << std::endl;
        flog_devel(info) << ss.str();
      } // scope

      cu.migrate_primaries();
      auto const & cnns = cu.primary_connectivity_state();

      // Verify single color processes leave distribution untouched.
      std::vector<size_t> distribution = {256};
      ASSERT_EQ(
        util::offsets(util::equal_map(sd.num_entities(2), colors)).ends(),
        distribution);
      UNIT_CAPTURE() << flog::container(naive.offsets.ends()) << '\n'
                     << flog::container(naive.values) << '\n'
                     << flog::container(cnns.m2p) << '\n';
      EXPECT_TRUE(
        UNIT_EQUAL_BLESSED(("coloring_1." + std::to_string(processes()) + "." +
                            std::to_string(process()) + ".blessed")));
    } // scope
  }; // UNIT
} // parmetis_coloring

int
coloring_driver() {
  UNIT() { ASSERT_EQ((test<parmetis_coloring, mpi>()), 0); };
} // simple2d_8x8

util::unit::driver<coloring_driver> driver;
