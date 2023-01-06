#include <vector>

#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/io.hh"
#include "flecsi/run/context.hh"
#include "flecsi/topo/narray/test/narray.hh"
#include "flecsi/util/constant.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;
using namespace flecsi::data;

using mesh1d = mesh<1>;
using ax = mesh1d::axis;

mesh1d::slot m;
mesh1d::cslot mc;

const field<double>::definition<mesh1d> m_field_1, m_field_2;
const field<int>::definition<mesh1d> m_field_i;
const field<std::size_t>::definition<mesh1d> m_field_s;
const field<int, ragged>::definition<mesh1d> m_field_r1, m_field_r2;

void
allocate(topo::resize::Field::accessor<wo> a) {
  a = 64;
}

void
init(mesh1d::accessor<ro> m,
  field<double>::accessor<wo, na> mf1,
  field<double>::accessor<wo, na> mf2,
  field<int>::accessor<wo, na> mfi,
  field<std::size_t>::accessor<wo, na> mfs,
  field<int, ragged>::mutator<wo, na> mfr1,
  field<int, ragged>::mutator<wo, na> mfr2) {
  for(auto i : m.range<ax::x_axis>()) {
    double val = 100. * color() + (int)i;
    mf1[i] = val;
    mf2[i] = val + 1000.;
    mfi[i] = val + 2000;
    mfs[i] = val + 3000;
    int r1size = (i & 1) + 2;
    mfr1[i].resize(r1size);
    for(int j = 0; j < r1size; ++j) {
      mfr1[i][j] = val + 4000 + 20 * j;
    }
    int r2size = r1size ^ 1;
    mfr2[i].resize(r2size);
    for(int j = 0; j < r2size; ++j) {
      mfr2[i][j] = val + 5000 + 20 * j;
    }
  } // for i
} // init

void
clear(mesh1d::accessor<ro> m,
  field<double>::accessor<wo, na> mf1,
  field<double>::accessor<wo, na> mf2,
  field<int>::accessor<wo, na> mfi,
  field<std::size_t>::accessor<wo, na> mfs,
  // TODO:  in attach mode, this fails if I change "accessor" to "mutator"
  //        - why?
  field<int, ragged>::accessor<rw, na> mfr1,
  field<int, ragged>::accessor<rw, na> mfr2) {
  for(auto i : m.range<ax::x_axis>()) {
    mf1[i] = 0.;
    mf2[i] = 0.;
    mfi[i] = 0;
    mfs[i] = 0;
    int r1size = mfr1[i].size();
    for(int j = 0; j < r1size; ++j) {
      mfr1[i][j] = 0;
    }
    int r2size = mfr2[i].size();
    for(int j = 0; j < r2size; ++j) {
      mfr2[i][j] = 0;
    }
  } // for i
} // clear

int
check(mesh1d::accessor<ro> m,
  field<double>::accessor<ro, na> mf1,
  field<double>::accessor<ro, na> mf2,
  field<int>::accessor<ro, na> mfi,
  field<std::size_t>::accessor<ro, na> mfs,
  field<int, ragged>::accessor<ro, na> mfr1,
  field<int, ragged>::accessor<ro, na> mfr2) {
  UNIT("TASK") {
    for(auto i : m.range<ax::x_axis>()) {
      double val = 100. * color() + (int)i;
      ASSERT_EQ(mf1[i], val);
      ASSERT_EQ(mf2[i], val + 1000.);
      ASSERT_EQ(mfi[i], val + 2000);
      ASSERT_EQ(mfs[i], val + 3000);
      int r1size = (i & 1) + 2;
      ASSERT_EQ(mfr1[i].size(), r1size);
      for(int j = 0; j < r1size; ++j) {
        ASSERT_EQ(mfr1[i][j], val + 4000 + 20 * j);
      }
      int r2size = r1size ^ 1;
      ASSERT_EQ(mfr2[i].size(), r2size);
      for(int j = 0; j < r2size; ++j) {
        ASSERT_EQ(mfr2[i][j], val + 5000 + 20 * j);
      }
    } // for i
  };
} // check

namespace {
int
setup() {
  mesh1d::gcoord indices{64};
  Color colors{4};
  mesh1d::index_definition idef;
  idef.axes = topo::narray_utils::make_axes(colors, indices);
  mc.allocate(idef);
  m.allocate(mc.get());
  run::context::instance().add_topology(m);
  return 0;
}
} // namespace

template<bool Attach>
int
restart_driver() {
  UNIT() {
    auto & mr = m_field_r1(m).get_ragged();
    mr.growth = {0, 0, 0.25, 0.5, 1};
    execute<allocate>(mr.sizes());
    // TODO:  figure out how to resize to the right size on a restart
    mr.resize();
    auto & mr2 = m_field_r2(m).get_ragged();
    mr2.growth = {0, 0, 0.25, 0.5, 1};
    execute<allocate>(mr2.sizes());
    mr2.resize();

    auto mf1 = m_field_1(m);
    auto mf2 = m_field_2(m);
    auto mfi = m_field_i(m);
    auto mfs = m_field_s(m);
    auto mfr1 = m_field_r1(m);
    auto mfr2 = m_field_r2(m);

    execute<init>(m, mf1, mf2, mfi, mfs, mfr1, mfr2);

    // Legion backend doesn't support N-to-M yet - use 1 rank/file
    // MPI backend supports N-to-M restarts - use 2 ranks/file
    io::io_interface iif(FLECSI_BACKEND == FLECSI_BACKEND_legion ? 1 : 2);
    auto filename =
      std::string{"hdf5_restart"} + (Attach ? "_w" : "_wo") + ".dat";
    iif.checkpoint_all_fields(filename, Attach);

    execute<clear>(m, mf1, mf2, mfi, mfs, mfr1, mfr2);
    iif.recover_all_fields(filename, Attach);

    EXPECT_EQ(test<check>(m, mf1, mf2, mfi, mfs, mfr1, mfr2), 0);
  };

  return 0;
}

util::unit::initialization<setup> init_mesh;
// for MPI:  run test once, since attach flag is ignored.
// for Legion:  run test twice, once with and once without attach.
util::unit::driver<restart_driver<true>> driver_w;
#if defined(FLECSI_ENABLE_LEGION)
util::unit::driver<restart_driver<false>> driver_wo;
#endif
