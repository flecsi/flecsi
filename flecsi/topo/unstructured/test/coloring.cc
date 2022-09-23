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

    const Color colors = 5;

    // Coloring with 5 colors with MPI_COMM_WORLD
    {
      coloring_utils cu(
        &sd, {colors, {2 /*id*/, 0 /*idx*/}, 1, {0, 1}, {{1, 2}}}, {});
      cu.create_graph(2);
      cu.color_primaries(1, util::parmetis::color);

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
      auto const & naive = cu.get_naive();

      std::vector<size_t> distribution = {52, 103, 154, 205, 256};
      ASSERT_EQ(
        util::offsets(util::equal_map(sd.num_entities(2), colors)).ends(),
        distribution);

      EXPECT_EQ(cu.primaries().at(process()), cnns.p2m);
      UNIT_CAPTURE() << flog::container(naive.offsets.ends()) << '\n'
                     << flog::container(naive.indices) << '\n'
                     << flog::container(cnns.p2m) << '\n'
                     << flog::container(cnns.m2p) << '\n';
      EXPECT_TRUE(UNIT_EQUAL_BLESSED(
        ("coloring_5." + std::to_string(process()) + ".blessed").c_str()));
    } // scope

    // Coloring with 5 colors with custom communicator with 2 processes
    {
      MPI_Comm group_comm;
      test(MPI_Comm_split(
        MPI_COMM_WORLD, process() < 2 ? 0 : MPI_UNDEFINED, 0, &group_comm));

      if(process() < 2) {
        coloring_utils cu(&sd,
          {colors, {2 /*id*/, 0 /*idx*/}, 1, {0, 1}, {{1, 2}}},
          {},
          group_comm);
        cu.create_graph(2);
        cu.color_primaries(1, util::parmetis::color);

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

        UNIT_CAPTURE() << flog::container(cu.primaries()) << '\n'
                       << flog::container(cnns.p2m) << '\n'
                       << flog::container(cnns.m2p) << '\n';
        EXPECT_TRUE(UNIT_EQUAL_BLESSED(
          ("coloring_2." + std::to_string(process()) + ".blessed").c_str()));

        test(MPI_Comm_free(&group_comm));
      } // if
    } // scope
  };
} // parmetis_coloring

int
coloring_driver() {
  UNIT() { ASSERT_EQ((test<parmetis_coloring, mpi>()), 0); };
} // simple2d_8x8

flecsi::unit::driver<coloring_driver> driver;
