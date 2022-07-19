// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#include "narray.hh"

#include "flecsi/execution.hh"
#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"

#include <set>

using namespace flecsi;

// 1D Mesh
mesh1d::slot m1;
mesh1d::cslot coloring1;
const field<std::size_t>::definition<mesh1d, mesh1d::index_space::entities> f1;

void
set_field_1d(mesh1d::accessor<ro> m, field<std::size_t>::accessor<wo, na> ca) {
  auto c = m.mdspan<mesh1d::index_space::entities>(ca);
  auto clr = color();
  forall(i, m.extents<mesh1d::axis::x_axis>(), "set_field_1d") { c[i] = clr; };
}

void
print_field_1d(mesh1d::accessor<ro> m,
  field<std::size_t>::accessor<ro, ro> ca) {
  auto c = m.mdspan<mesh1d::index_space::entities>(ca);
  std::stringstream ss;
  for(auto i : m.extents<mesh1d::axis::x_axis, mesh1d::range::all>()) {
    ss << c[i] << " ";
  } // for
  ss << std::endl;
  flog(warn) << ss.str() << std::endl;
}

int
check_1d(mesh1d::accessor<ro> m) {
  UNIT {
    using r = mesh1d::range;
    using ax = mesh1d::axis;

    // check extents
    std::set<util::id> logical[4] = {{2, 3, 4}, {1, 2}, {1, 2}, {1, 2}};
    std::set<util::id> extended[4] = {
      {0, 1, 2, 3, 4}, {1, 2}, {1, 2}, {1, 2, 3, 4}};
    std::set<util::id> all[4] = {
      {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3, 4}};
    std::set<util::id> boundary_low[4] = {{0, 1}, {}, {}, {}};
    std::set<util::id> boundary_high[4] = {{}, {}, {}, {3, 4}};
    std::set<util::id> ghost_low[4] = {{}, {0}, {0}, {0}};
    std::set<util::id> ghost_high[4] = {{5}, {3}, {3}, {}};

    const int rank = process();
    const auto s = [](auto && r) {
      return std::set<util::id>(r.begin(), r.end());
    };

    EXPECT_EQ(s(m.extents<ax::x_axis>()), logical[rank]);
    EXPECT_EQ(s(m.extents<ax::x_axis, r::extended>()), extended[rank]);
    EXPECT_EQ(s(m.extents<ax::x_axis, r::all>()), all[rank]);
    EXPECT_EQ(s(m.extents<ax::x_axis, r::boundary_low>()), boundary_low[rank]);
    EXPECT_EQ(
      s(m.extents<ax::x_axis, r::boundary_high>()), boundary_high[rank]);
    EXPECT_EQ(s(m.extents<ax::x_axis, r::ghost_low>()), ghost_low[rank]);
    EXPECT_EQ(s(m.extents<ax::x_axis, r::ghost_high>()), ghost_high[rank]);

    // check sizes
    std::size_t xsizes_ex[4][8] = {{3, 5, 6, 2, 0, 0, 1, 9},
      {2, 2, 4, 0, 0, 1, 1, 9},
      {2, 2, 4, 0, 0, 1, 1, 9},
      {2, 4, 5, 0, 2, 1, 0, 9}};

    std::size_t xsizes[] = {m.size<ax::x_axis>(),
      m.size<ax::x_axis, r::extended>(),
      m.size<ax::x_axis, r::all>(),
      m.size<ax::x_axis, r::boundary_low>(),
      m.size<ax::x_axis, r::boundary_high>(),
      m.size<ax::x_axis, r::ghost_low>(),
      m.size<ax::x_axis, r::ghost_high>(),
      m.size<ax::x_axis, r::global>()};

    for(int i = 0; i < 8; i++) {
      EXPECT_EQ(xsizes[i], xsizes_ex[process()][i]);
    }

    // check offsets
    std::size_t xoffsets_ex[4][8] = {{2, 0, 0, 0, 5, 0, 5, 0},
      {1, 1, 0, 1, 3, 0, 3, 2},
      {1, 1, 0, 1, 3, 0, 3, 4},
      {1, 1, 0, 1, 3, 0, 3, 6}};

    std::size_t xoffsets[] = {m.offset<ax::x_axis>(),
      m.offset<ax::x_axis, r::extended>(),
      m.offset<ax::x_axis, r::all>(),
      m.offset<ax::x_axis, r::boundary_low>(),
      m.offset<ax::x_axis, r::boundary_high>(),
      m.offset<ax::x_axis, r::ghost_low>(),
      m.offset<ax::x_axis, r::ghost_high>(),
      m.offset<ax::x_axis, r::global>()};

    for(int i = 0; i < 8; i++) {
      EXPECT_EQ(xoffsets[i], xoffsets_ex[process()][i]);
    }
  };
} // check_1d

// 2D Mesh
mesh2d::slot m2;
mesh2d::cslot coloring2;
const field<std::size_t>::definition<mesh2d, mesh2d::index_space::entities> f2;

void
set_field_2d(mesh2d::accessor<ro> m, field<std::size_t>::accessor<wo, na> ca) {
  auto c = m.mdspan<mesh2d::index_space::entities>(ca);
  auto x_ex = m.extents<mesh2d::axis::x_axis>();
  auto clr = color();
  forall(j, m.extents<mesh2d::axis::y_axis>(), "set_field_2d") {
    for(auto i : x_ex)
      c[j][i] = clr;
  };
}

void
print_field_2d(mesh2d::accessor<ro> m,
  field<std::size_t>::accessor<ro, ro> ca) {
  auto c = m.mdspan<mesh2d::index_space::entities>(ca);
  std::stringstream ss;
  for(int j = m.size<mesh2d::axis::y_axis, mesh2d::range::all>() - 1; j >= 0;
      --j) {
    for(auto i : m.extents<mesh2d::axis::x_axis, mesh2d::range::all>()) {
      ss << c[j][i] << " ";
    } // for
    ss << std::endl;
  } // for
  ss << std::endl;
  flog(warn) << ss.str() << std::endl;
}

int
check_2d(mesh2d::accessor<ro> m) {
  UNIT {
    using r = mesh2d::range;
    using ax = mesh2d::axis;

    // check extents
    std::set<util::id> xlogical[4] = {
      {2, 3, 4, 5}, {1, 2, 3, 4}, {2, 3, 4, 5}, {1, 2, 3, 4}};
    std::set<util::id> xextended[4] = {{0, 1, 2, 3, 4, 5},
      {1, 2, 3, 4, 5, 6},
      {0, 1, 2, 3, 4, 5},
      {1, 2, 3, 4, 5, 6}};
    std::set<util::id> xall[4] = {{0, 1, 2, 3, 4, 5, 6},
      {0, 1, 2, 3, 4, 5, 6},
      {0, 1, 2, 3, 4, 5, 6},
      {0, 1, 2, 3, 4, 5, 6}};
    std::set<util::id> xboundary_low[4] = {{0, 1}, {}, {0, 1}, {}};
    std::set<util::id> xboundary_high[4] = {{}, {5, 6}, {}, {5, 6}};
    std::set<util::id> xghost_low[4] = {{}, {0}, {}, {0}};
    std::set<util::id> xghost_high[4] = {{6}, {}, {6}, {}};

    std::set<util::id> ylogical[4] = {
      {1, 2, 3, 4}, {1, 2, 3, 4}, {2, 3, 4, 5}, {2, 3, 4, 5}};
    std::set<util::id> yextended[4] = {
      {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    std::set<util::id> yall[4] = {{0, 1, 2, 3, 4, 5, 6},
      {0, 1, 2, 3, 4, 5, 6},
      {0, 1, 2, 3, 4, 5, 6},
      {0, 1, 2, 3, 4, 5, 6}};
    std::set<util::id> yboundary_low[4] = {{0}, {0}, {}, {}};
    std::set<util::id> yboundary_high[4] = {{}, {}, {6}, {6}};
    std::set<util::id> yghost_low[4] = {{}, {}, {0, 1}, {0, 1}};
    std::set<util::id> yghost_high[4] = {{5, 6}, {5, 6}, {}, {}};

    const int rank = process();
    const auto s = [](auto && r) {
      return std::set<util::id>(r.begin(), r.end());
    };

    EXPECT_EQ(s(m.extents<ax::x_axis>()), xlogical[rank]);
    EXPECT_EQ(s(m.extents<ax::x_axis, r::extended>()), xextended[rank]);
    EXPECT_EQ(s(m.extents<ax::x_axis, r::all>()), xall[rank]);
    EXPECT_EQ(s(m.extents<ax::x_axis, r::boundary_low>()), xboundary_low[rank]);
    EXPECT_EQ(
      s(m.extents<ax::x_axis, r::boundary_high>()), xboundary_high[rank]);
    EXPECT_EQ(s(m.extents<ax::x_axis, r::ghost_low>()), xghost_low[rank]);
    EXPECT_EQ(s(m.extents<ax::x_axis, r::ghost_high>()), xghost_high[rank]);

    EXPECT_EQ(s(m.extents<ax::y_axis>()), ylogical[rank]);
    EXPECT_EQ(s(m.extents<ax::y_axis, r::extended>()), yextended[rank]);
    EXPECT_EQ(s(m.extents<ax::y_axis, r::all>()), yall[rank]);
    EXPECT_EQ(s(m.extents<ax::y_axis, r::boundary_low>()), yboundary_low[rank]);
    EXPECT_EQ(
      s(m.extents<ax::y_axis, r::boundary_high>()), yboundary_high[rank]);
    EXPECT_EQ(s(m.extents<ax::y_axis, r::ghost_low>()), yghost_low[rank]);
    EXPECT_EQ(s(m.extents<ax::y_axis, r::ghost_high>()), yghost_high[rank]);

    // check sizes
    std::size_t xsizes_ex[4][8] = {{4, 6, 7, 2, 0, 0, 1, 8},
      {4, 6, 7, 0, 2, 1, 0, 8},
      {4, 6, 7, 2, 0, 0, 1, 8},
      {4, 6, 7, 0, 2, 1, 0, 8}};

    std::size_t xsizes[] = {m.size<ax::x_axis>(),
      m.size<ax::x_axis, r::extended>(),
      m.size<ax::x_axis, r::all>(),
      m.size<ax::x_axis, r::boundary_low>(),
      m.size<ax::x_axis, r::boundary_high>(),
      m.size<ax::x_axis, r::ghost_low>(),
      m.size<ax::x_axis, r::ghost_high>(),
      m.size<ax::x_axis, r::global>()};

    for(int i = 0; i < 8; i++) {
      EXPECT_EQ(xsizes[i], xsizes_ex[process()][i]);
    };

    std::size_t ysizes_ex[4][8] = {{4, 5, 7, 1, 0, 0, 2, 8},
      {4, 5, 7, 1, 0, 0, 2, 8},
      {4, 5, 7, 0, 1, 2, 0, 8},
      {4, 5, 7, 0, 1, 2, 0, 8}};

    std::size_t ysizes[] = {m.size<ax::y_axis>(),
      m.size<ax::y_axis, r::extended>(),
      m.size<ax::y_axis, r::all>(),
      m.size<ax::y_axis, r::boundary_low>(),
      m.size<ax::y_axis, r::boundary_high>(),
      m.size<ax::y_axis, r::ghost_low>(),
      m.size<ax::y_axis, r::ghost_high>(),
      m.size<ax::y_axis, r::global>()};

    for(int i = 0; i < 8; i++) {
      EXPECT_EQ(ysizes[i], ysizes_ex[process()][i]);
    }

    // check offsets
    std::size_t xoffsets_ex[4][8] = {{2, 0, 0, 0, 6, 0, 6, 0},
      {1, 1, 0, 1, 5, 0, 5, 3},
      {2, 0, 0, 0, 6, 0, 6, 0},
      {1, 1, 0, 1, 5, 0, 5, 3}};

    std::size_t xoffsets[] = {m.offset<ax::x_axis>(),
      m.offset<ax::x_axis, r::extended>(),
      m.offset<ax::x_axis, r::all>(),
      m.offset<ax::x_axis, r::boundary_low>(),
      m.offset<ax::x_axis, r::boundary_high>(),
      m.offset<ax::x_axis, r::ghost_low>(),
      m.offset<ax::x_axis, r::ghost_high>(),
      m.offset<ax::x_axis, r::global>()};

    for(int i = 0; i < 8; i++) {
      EXPECT_EQ(xoffsets[i], xoffsets_ex[process()][i]);
    }

    std::size_t yoffsets_ex[4][8] = {{1, 0, 0, 0, 5, 0, 5, 0},
      {1, 0, 0, 0, 5, 0, 5, 0},
      {2, 2, 0, 2, 6, 0, 6, 2},
      {2, 2, 0, 2, 6, 0, 6, 2}};

    std::size_t yoffsets[] = {m.offset<ax::y_axis>(),
      m.offset<ax::y_axis, r::extended>(),
      m.offset<ax::y_axis, r::all>(),
      m.offset<ax::y_axis, r::boundary_low>(),
      m.offset<ax::y_axis, r::boundary_high>(),
      m.offset<ax::y_axis, r::ghost_low>(),
      m.offset<ax::y_axis, r::ghost_high>(),
      m.offset<ax::y_axis, r::global>()};

    for(int i = 0; i < 8; i++) {
      EXPECT_EQ(yoffsets[i], yoffsets_ex[process()][i]);
    }
  };
} // check_2d

// 3D Mesh
mesh3d::slot m3;
mesh3d::cslot coloring3;
const field<std::size_t>::definition<mesh3d, mesh3d::index_space::entities> f3;

void
set_field_3d(mesh3d::accessor<ro> m, field<std::size_t>::accessor<wo, na> ca) {
  auto c = m.mdspan<mesh3d::index_space::entities>(ca);
  auto x_ex = m.extents<mesh3d::axis::x_axis>();
  auto y_ex = m.extents<mesh3d::axis::y_axis>();
  auto clr = color();
  forall(k, m.extents<mesh3d::axis::z_axis>(), "set_field_3d") {
    for(auto j : y_ex)
      for(auto i : x_ex)
        c[k][j][i] = clr;
  };
}

void
print_field_3d(mesh3d::accessor<ro> m,
  field<std::size_t>::accessor<ro, ro> ca) {
  auto c = m.mdspan<mesh3d::index_space::entities>(ca);
  std::stringstream ss;
  for(int k = m.size<mesh3d::axis::z_axis, mesh3d::range::all>() - 1; k >= 0;
      --k) {
    for(int j = m.size<mesh3d::axis::y_axis, mesh3d::range::all>() - 1; j >= 0;
        --j) {
      for(auto i : m.extents<mesh3d::axis::x_axis, mesh3d::range::all>()) {
        ss << c[k][j][i] << " ";
      } // for
      ss << std::endl;
    } // for
    ss << std::endl;
  } // for
  ss << std::endl;
  flog(warn) << ss.str() << std::endl;
}

int
check_3d(mesh3d::accessor<ro> m) {
  UNIT {
    using r = mesh3d::range;
    using ax = mesh3d::axis;

    std::set<util::id> xlogical[4] = {{1, 2}, {2}, {1, 2}, {2}};
    std::set<util::id> xextended[4] = {{0, 1, 2}, {2, 3}, {0, 1, 2}, {2, 3}};
    std::set<util::id> xall[4] = {
      {0, 1, 2, 3, 4}, {0, 1, 2, 3}, {0, 1, 2, 3, 4}, {0, 1, 2, 3}};
    std::set<util::id> xboundary_low[4] = {{0}, {}, {0}, {}};
    std::set<util::id> xboundary_high[4] = {{}, {3}, {}, {3}};
    std::set<util::id> xghost_low[4] = {{}, {0, 1}, {}, {0, 1}};
    std::set<util::id> xghost_high[4] = {{3, 4}, {}, {3, 4}, {}};

    std::set<util::id> ylogical[4] = {
      {2, 3, 4}, {2, 3, 4}, {2, 3, 4}, {2, 3, 4}};
    std::set<util::id> yextended[4] = {{0, 1, 2, 3, 4, 5, 6},
      {0, 1, 2, 3, 4, 5, 6},
      {0, 1, 2, 3, 4, 5, 6},
      {0, 1, 2, 3, 4, 5, 6}};
    std::set<util::id> yall[4] = {{0, 1, 2, 3, 4, 5, 6},
      {0, 1, 2, 3, 4, 5, 6},
      {0, 1, 2, 3, 4, 5, 6},
      {0, 1, 2, 3, 4, 5, 6}};
    std::set<util::id> yboundary_low[4] = {{0, 1}, {0, 1}, {0, 1}, {0, 1}};
    std::set<util::id> yboundary_high[4] = {{5, 6}, {5, 6}, {5, 6}, {5, 6}};
    std::set<util::id> yghost_low[4] = {{}, {}, {}, {}};
    std::set<util::id> yghost_high[4] = {{}, {}, {}, {}};

    std::set<util::id> zlogical[4] = {{1, 2}, {1, 2}, {1, 2}, {1, 2}};
    std::set<util::id> zextended[4] = {
      {0, 1, 2}, {0, 1, 2}, {1, 2, 3}, {1, 2, 3}};
    std::set<util::id> zall[4] = {
      {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}};
    std::set<util::id> zboundary_low[4] = {{0}, {0}, {}, {}};
    std::set<util::id> zboundary_high[4] = {{}, {}, {3}, {3}};
    std::set<util::id> zghost_low[4] = {{}, {}, {0}, {0}};
    std::set<util::id> zghost_high[4] = {{3}, {3}, {}, {}};

    const int rank = process();
    const auto s = [](auto && r) {
      return std::set<util::id>(r.begin(), r.end());
    };

    EXPECT_EQ(s(m.extents<ax::x_axis>()), xlogical[rank]);
    EXPECT_EQ(s(m.extents<ax::x_axis, r::extended>()), xextended[rank]);
    EXPECT_EQ(s(m.extents<ax::x_axis, r::all>()), xall[rank]);
    EXPECT_EQ(s(m.extents<ax::x_axis, r::boundary_low>()), xboundary_low[rank]);
    EXPECT_EQ(
      s(m.extents<ax::x_axis, r::boundary_high>()), xboundary_high[rank]);
    EXPECT_EQ(s(m.extents<ax::x_axis, r::ghost_low>()), xghost_low[rank]);
    EXPECT_EQ(s(m.extents<ax::x_axis, r::ghost_high>()), xghost_high[rank]);

    EXPECT_EQ(s(m.extents<ax::y_axis>()), ylogical[rank]);
    EXPECT_EQ(s(m.extents<ax::y_axis, r::extended>()), yextended[rank]);
    EXPECT_EQ(s(m.extents<ax::y_axis, r::all>()), yall[rank]);
    EXPECT_EQ(s(m.extents<ax::y_axis, r::boundary_low>()), yboundary_low[rank]);
    EXPECT_EQ(
      s(m.extents<ax::y_axis, r::boundary_high>()), yboundary_high[rank]);
    EXPECT_EQ(s(m.extents<ax::y_axis, r::ghost_low>()), yghost_low[rank]);
    EXPECT_EQ(s(m.extents<ax::y_axis, r::ghost_high>()), yghost_high[rank]);

    EXPECT_EQ(s(m.extents<ax::z_axis>()), zlogical[rank]);
    EXPECT_EQ(s(m.extents<ax::z_axis, r::extended>()), zextended[rank]);
    EXPECT_EQ(s(m.extents<ax::z_axis, r::all>()), zall[rank]);
    EXPECT_EQ(s(m.extents<ax::z_axis, r::boundary_low>()), zboundary_low[rank]);
    EXPECT_EQ(
      s(m.extents<ax::z_axis, r::boundary_high>()), zboundary_high[rank]);
    EXPECT_EQ(s(m.extents<ax::z_axis, r::ghost_low>()), zghost_low[rank]);
    EXPECT_EQ(s(m.extents<ax::z_axis, r::ghost_high>()), zghost_high[rank]);

    // check sizes
    std::size_t xsizes_ex[4][8] = {{2, 3, 5, 1, 0, 0, 2, 3},
      {1, 2, 4, 0, 1, 2, 0, 3},
      {2, 3, 5, 1, 0, 0, 2, 3},
      {1, 2, 4, 0, 1, 2, 0, 3}};

    std::size_t xsizes[] = {m.size<ax::x_axis>(),
      m.size<ax::x_axis, r::extended>(),
      m.size<ax::x_axis, r::all>(),
      m.size<ax::x_axis, r::boundary_low>(),
      m.size<ax::x_axis, r::boundary_high>(),
      m.size<ax::x_axis, r::ghost_low>(),
      m.size<ax::x_axis, r::ghost_high>(),
      m.size<ax::x_axis, r::global>()};

    for(int i = 0; i < 8; i++) {
      EXPECT_EQ(xsizes[i], xsizes_ex[process()][i]);
    }

    std::size_t ysizes_ex[4][8] = {{3, 7, 7, 2, 2, 0, 0, 3},
      {3, 7, 7, 2, 2, 0, 0, 3},
      {3, 7, 7, 2, 2, 0, 0, 3},
      {3, 7, 7, 2, 2, 0, 0, 3}};

    std::size_t ysizes[] = {m.size<ax::y_axis>(),
      m.size<ax::y_axis, r::extended>(),
      m.size<ax::y_axis, r::all>(),
      m.size<ax::y_axis, r::boundary_low>(),
      m.size<ax::y_axis, r::boundary_high>(),
      m.size<ax::y_axis, r::ghost_low>(),
      m.size<ax::y_axis, r::ghost_high>(),
      m.size<ax::y_axis, r::global>()};

    for(int i = 0; i < 8; i++) {
      EXPECT_EQ(ysizes[i], ysizes_ex[process()][i]);
    }

    std::size_t zsizes_ex[4][8] = {{2, 3, 4, 1, 0, 0, 1, 4},
      {2, 3, 4, 1, 0, 0, 1, 4},
      {2, 3, 4, 0, 1, 1, 0, 4},
      {2, 3, 4, 0, 1, 1, 0, 4}};

    std::size_t zsizes[] = {m.size<ax::z_axis>(),
      m.size<ax::z_axis, r::extended>(),
      m.size<ax::z_axis, r::all>(),
      m.size<ax::z_axis, r::boundary_low>(),
      m.size<ax::z_axis, r::boundary_high>(),
      m.size<ax::z_axis, r::ghost_low>(),
      m.size<ax::z_axis, r::ghost_high>(),
      m.size<ax::z_axis, r::global>()};

    for(int i = 0; i < 8; i++) {
      EXPECT_EQ(zsizes[i], zsizes_ex[process()][i]);
    }

    // check offsets
    std::size_t xoffsets_ex[4][8] = {{1, 0, 0, 0, 3, 0, 3, 0},
      {2, 2, 0, 2, 3, 0, 3, 0},
      {1, 0, 0, 0, 3, 0, 3, 0},
      {2, 2, 0, 2, 3, 0, 3, 0}};

    std::size_t xoffsets[] = {m.offset<ax::x_axis>(),
      m.offset<ax::x_axis, r::extended>(),
      m.offset<ax::x_axis, r::all>(),
      m.offset<ax::x_axis, r::boundary_low>(),
      m.offset<ax::x_axis, r::boundary_high>(),
      m.offset<ax::x_axis, r::ghost_low>(),
      m.offset<ax::x_axis, r::ghost_high>(),
      m.offset<ax::x_axis, r::global>()};

    for(int i = 0; i < 8; i++) {
      EXPECT_EQ(xoffsets[i], xoffsets_ex[process()][i]);
    }

    std::size_t yoffsets_ex[4][8] = {{2, 0, 0, 0, 5, 0, 5, 0},
      {2, 0, 0, 0, 5, 0, 5, 0},
      {2, 0, 0, 0, 5, 0, 5, 0},
      {2, 0, 0, 0, 5, 0, 5, 0}};

    std::size_t yoffsets[] = {m.offset<ax::y_axis>(),
      m.offset<ax::y_axis, r::extended>(),
      m.offset<ax::y_axis, r::all>(),
      m.offset<ax::y_axis, r::boundary_low>(),
      m.offset<ax::y_axis, r::boundary_high>(),
      m.offset<ax::y_axis, r::ghost_low>(),
      m.offset<ax::y_axis, r::ghost_high>(),
      m.offset<ax::y_axis, r::global>()};
    for(int i = 0; i < 8; i++) {
      EXPECT_EQ(yoffsets[i], yoffsets_ex[process()][i]);
    }

    std::size_t zoffsets_ex[4][8] = {{1, 0, 0, 0, 3, 0, 3, 0},
      {1, 0, 0, 0, 3, 0, 3, 0},
      {1, 1, 0, 1, 3, 0, 3, 1},
      {1, 1, 0, 1, 3, 0, 3, 1}};

    std::size_t zoffsets[] = {m.offset<ax::z_axis>(),
      m.offset<ax::z_axis, r::extended>(),
      m.offset<ax::z_axis, r::all>(),
      m.offset<ax::z_axis, r::boundary_low>(),
      m.offset<ax::z_axis, r::boundary_high>(),
      m.offset<ax::z_axis, r::ghost_low>(),
      m.offset<ax::z_axis, r::ghost_high>(),
      m.offset<ax::z_axis, r::global>()};

    for(int i = 0; i < 8; i++) {
      EXPECT_EQ(zoffsets[i], zoffsets_ex[process()][i]);
    }
  };
} // check_3d

int
narray_driver() {
  UNIT {
    {
      using topo::narray_utils::factor;
      using V = std::vector<std::size_t>;
      EXPECT_EQ(factor(2 * 5 * 11 * 13 * 29), (V{29, 13, 11, 5, 2}));
      EXPECT_EQ(factor(2 * 2 * 23 * 23), (V{23, 23, 2, 2}));
    }

    {
      // 1D Mesh
      mesh1d::coord indices{9};
      auto colors = topo::narray_utils::distribute(processes(), indices);
      flog(warn) << log::container{colors} << std::endl;

      mesh1d::coord hdepths{1};
      mesh1d::coord bdepths{2};
      std::vector<bool> periodic{false};
      std::vector<mesh1d::coloring_definition> index_definitions = {
        {colors, indices, hdepths, bdepths, periodic, true}};

      coloring1.allocate(index_definitions);
      m1.allocate(coloring1.get());
      execute<set_field_1d, default_accelerator>(m1, f1(m1));
      execute<print_field_1d>(m1, f1(m1));
      EXPECT_EQ(test<check_1d>(m1), 0);
    } // scope

    {
      // 2D Mesh
      mesh2d::coord indices{8, 8};
      auto colors = topo::narray_utils::distribute(processes(), indices);
      flog(warn) << log::container{colors} << std::endl;

      mesh2d::coord hdepths{1, 2};
      mesh2d::coord bdepths{2, 1};
      std::vector<bool> periodic{false, false};
      std::vector<mesh2d::coloring_definition> index_definitions = {
        {colors, indices, hdepths, bdepths, periodic, true}};
      coloring2.allocate(index_definitions);
      m2.allocate(coloring2.get());
      execute<set_field_2d, default_accelerator>(m2, f2(m2));
      execute<print_field_2d>(m2, f2(m2));
      EXPECT_EQ(test<check_2d>(m2), 0);
    } // scope

    {
      // 3D Mesh
      mesh3d::coord indices{3, 3, 4};
      auto colors = topo::narray_utils::distribute(processes(), indices);
      flog(warn) << log::container{colors} << std::endl;

      mesh3d::coord hdepths{2, 1, 1};
      mesh3d::coord bdepths{1, 2, 1};
      std::vector<bool> periodic{false, false, false};
      std::vector<mesh3d::coloring_definition> index_definitions = {
        {colors, indices, hdepths, bdepths, periodic, true}};
      coloring3.allocate(index_definitions);
      m3.allocate(coloring3.get());
      execute<set_field_3d>(m3, f3(m3));
      execute<print_field_3d>(m3, f3(m3));
      EXPECT_EQ(test<check_3d>(m3), 0);
    } // scope
  };
} // coloring_driver

flecsi::unit::driver<narray_driver> driver;
