
/*
    @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
   /@@/////  /@@          @@////@@ @@////// /@@
   /@@       /@@  @@@@@  @@    // /@@       /@@
   /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
   /@@////   /@@/@@@@@@@/@@       ////////@@/@@
   /@@       /@@/@@//// //@@    @@       /@@/@@
   /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
   //       ///  //////   //////  ////////  //

   Copyright (c) 2016, Triad National Security, LLC
   All rights reserved.
                                                                              */
#pragma once

#include "flecsi/util/unit.hh"

#include <cmath>
#include <set>

using mesh1d = mesh<1>;
using mesh2d = mesh<2>;
using mesh3d = mesh<3>;
using mesh4d = mesh<4>;

template<std::size_t D, typename F>
void
field_helper(typename mesh<D>::template accessor<ro> m,
  field<std::size_t>::accessor<wo, na> ca,
  F && fvalue) {
  auto c = m.template mdspan<mesh<D>::index_space::entities>(ca);

  // auto x=  m.template range<mesh1d::axis::x_axis>();
  if constexpr(D == 1) {
    forall(i, m.template range<mesh1d::axis::x_axis>(), "field_helper 1d") {
      fvalue(c[i]);
    }; // for
  }
  else if constexpr(D == 2) {
    auto x = m.template range<mesh2d::axis::x_axis>();
    forall(j, m.template range<mesh2d::axis::y_axis>(), "field_helper 2d") {
      for(auto i : x) {
        fvalue(c[j][i]);
      } // for
    }; // for
  }
  else {
    auto x = m.template range<mesh3d::axis::x_axis>();
    auto y = m.template range<mesh3d::axis::y_axis>();
    forall(k, m.template range<mesh3d::axis::z_axis>(), "field_helper 3d") {
      for(auto j : y) {
        for(auto i : x) {
          fvalue(c[k][j][i]);
        } // for
      } // for
    }; // for
  }
} // field_helper

template<std::size_t D>
void
init_field(typename mesh<D>::template accessor<ro> m,
  field<std::size_t>::accessor<wo, na> ca) {
  return field_helper<D>(m, ca, [c = color()](auto & x) { x = c; });
} // init_field

template<std::size_t D>
void
update_field(typename mesh<D>::template accessor<ro> m,
  field<std::size_t>::accessor<wo, na> ca) {
  return field_helper<D>(
    m, ca, [c = color()](auto & x) { x = std::pow(10, c); });
} // update_field

template<std::size_t D>
void
print_field(typename mesh<D>::template accessor<ro> m,
  field<std::size_t>::accessor<ro, ro> ca) {
  auto c = m.template mdspan<mesh<D>::index_space::entities>(ca);
  std::stringstream ss;
  if constexpr(D == 1) {
    for(auto i :
      m.template range<mesh1d::axis::x_axis, mesh1d::domain::all>()) {
      ss << c[i] << "   ";
    } // for
    ss << std::endl;
    flog(warn) << ss.str() << std::endl;
  }
  else if constexpr(D == 2) {
    for(int j = m.template size<mesh2d::axis::y_axis, mesh2d::domain::all>();
        j--;) {
      for(auto i :
        m.template range<mesh2d::axis::x_axis, mesh2d::domain::all>()) {
        ss << c[j][i] << "   ";
      } // for
      ss << std::endl;
    } // for
    flog(warn) << ss.str() << std::endl;
  }
  else {
    for(int k = m.template size<mesh3d::axis::z_axis, mesh3d::domain::all>();
        k--;) {
      for(int j = m.template size<mesh3d::axis::y_axis, mesh3d::domain::all>();
          j--;) {
        for(auto i :
          m.template range<mesh3d::axis::x_axis, mesh3d::domain::all>()) {
          ss << c[k][j][i] << "   ";
        } // for
        ss << std::endl;
      } // for
      ss << std::endl;
    } // for
    ss << std::endl;
    flog(warn) << ss.str() << std::endl;
  }

} // print_field

template<std::size_t D>
int
check_mesh_field(typename mesh<D>::template accessor<ro> m,
  field<std::size_t>::accessor<ro, ro> ca) {
  if constexpr(D == 1) {
    UNIT {
      using r = mesh1d::domain;
      using ax = mesh1d::axis;

      // check range
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

      EXPECT_EQ(s(m.template range<ax::x_axis>()), logical[rank]);
      EXPECT_EQ(s(m.template range<ax::x_axis, r::extended>()), extended[rank]);
      EXPECT_EQ(s(m.template range<ax::x_axis, r::all>()), all[rank]);
      EXPECT_EQ(
        s(m.template range<ax::x_axis, r::boundary_low>()), boundary_low[rank]);
      EXPECT_EQ(s(m.template range<ax::x_axis, r::boundary_high>()),
        boundary_high[rank]);
      EXPECT_EQ(
        s(m.template range<ax::x_axis, r::ghost_low>()), ghost_low[rank]);
      EXPECT_EQ(
        s(m.template range<ax::x_axis, r::ghost_high>()), ghost_high[rank]);

      // check sizes
      std::size_t xsizes_ex[4][8] = {{3, 5, 6, 2, 0, 0, 1, 9},
        {2, 2, 4, 0, 0, 1, 1, 9},
        {2, 2, 4, 0, 0, 1, 1, 9},
        {2, 4, 5, 0, 2, 1, 0, 9}};

      std::size_t xsizes[] = {m.template size<ax::x_axis>(),
        m.template size<ax::x_axis, r::extended>(),
        m.template size<ax::x_axis, r::all>(),
        m.template size<ax::x_axis, r::boundary_low>(),
        m.template size<ax::x_axis, r::boundary_high>(),
        m.template size<ax::x_axis, r::ghost_low>(),
        m.template size<ax::x_axis, r::ghost_high>(),
        m.template size<ax::x_axis, r::global>()};

      for(int i = 0; i < 8; i++) {
        EXPECT_EQ(xsizes[i], xsizes_ex[rank][i]);
      }

      // check offsets
      std::size_t xoffsets_ex[4][8] = {{2, 0, 0, 0, 5, 0, 5, 0},
        {1, 1, 0, 1, 3, 0, 3, 2},
        {1, 1, 0, 1, 3, 0, 3, 4},
        {1, 1, 0, 1, 3, 0, 3, 6}};

      std::size_t xoffsets[] = {m.template offset<ax::x_axis>(),
        m.template offset<ax::x_axis, r::extended>(),
        m.template offset<ax::x_axis, r::all>(),
        m.template offset<ax::x_axis, r::boundary_low>(),
        m.template offset<ax::x_axis, r::boundary_high>(),
        m.template offset<ax::x_axis, r::ghost_low>(),
        m.template offset<ax::x_axis, r::ghost_high>(),
        m.template offset<ax::x_axis, r::global>()};

      for(int i = 0; i < 8; i++) {
        EXPECT_EQ(xoffsets[i], xoffsets_ex[rank][i]);
      }

      // check field values on the ghost layers
      int ngb_ranks[4][2] = {{-1, 1}, {0, 2}, {1, 3}, {2, -1}};
      auto c = m.template mdspan<mesh1d::index_space::entities>(ca);

      if(ngb_ranks[rank][0] != -1) {
        for(auto i : ghost_low[rank]) {
          EXPECT_EQ(c[i], std::pow(10, ngb_ranks[rank][0]));
        }
      }

      if(ngb_ranks[rank][1] != -1) {
        for(auto i : ghost_high[rank]) {
          EXPECT_EQ(c[i], std::pow(10, ngb_ranks[rank][1]));
        }
      }
    };
  } // d=1
  else if constexpr(D == 2) {
    UNIT {
      using r = mesh2d::domain;
      using ax = mesh2d::axis;

      // check range
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

      EXPECT_EQ(s(m.template range<ax::x_axis>()), xlogical[rank]);
      EXPECT_EQ(
        s(m.template range<ax::x_axis, r::extended>()), xextended[rank]);
      EXPECT_EQ(s(m.template range<ax::x_axis, r::all>()), xall[rank]);
      EXPECT_EQ(s(m.template range<ax::x_axis, r::boundary_low>()),
        xboundary_low[rank]);
      EXPECT_EQ(s(m.template range<ax::x_axis, r::boundary_high>()),
        xboundary_high[rank]);
      EXPECT_EQ(
        s(m.template range<ax::x_axis, r::ghost_low>()), xghost_low[rank]);
      EXPECT_EQ(
        s(m.template range<ax::x_axis, r::ghost_high>()), xghost_high[rank]);

      EXPECT_EQ(s(m.template range<ax::y_axis>()), ylogical[rank]);
      EXPECT_EQ(
        s(m.template range<ax::y_axis, r::extended>()), yextended[rank]);
      EXPECT_EQ(s(m.template range<ax::y_axis, r::all>()), yall[rank]);
      EXPECT_EQ(s(m.template range<ax::y_axis, r::boundary_low>()),
        yboundary_low[rank]);
      EXPECT_EQ(s(m.template range<ax::y_axis, r::boundary_high>()),
        yboundary_high[rank]);
      EXPECT_EQ(
        s(m.template range<ax::y_axis, r::ghost_low>()), yghost_low[rank]);
      EXPECT_EQ(
        s(m.template range<ax::y_axis, r::ghost_high>()), yghost_high[rank]);

      // check sizes
      std::size_t xsizes_ex[4][8] = {{4, 6, 7, 2, 0, 0, 1, 8},
        {4, 6, 7, 0, 2, 1, 0, 8},
        {4, 6, 7, 2, 0, 0, 1, 8},
        {4, 6, 7, 0, 2, 1, 0, 8}};

      std::size_t xsizes[] = {m.template size<ax::x_axis>(),
        m.template size<ax::x_axis, r::extended>(),
        m.template size<ax::x_axis, r::all>(),
        m.template size<ax::x_axis, r::boundary_low>(),
        m.template size<ax::x_axis, r::boundary_high>(),
        m.template size<ax::x_axis, r::ghost_low>(),
        m.template size<ax::x_axis, r::ghost_high>(),
        m.template size<ax::x_axis, r::global>()};

      for(int i = 0; i < 8; i++) {
        EXPECT_EQ(xsizes[i], xsizes_ex[process()][i]);
      };

      std::size_t ysizes_ex[4][8] = {{4, 5, 7, 1, 0, 0, 2, 8},
        {4, 5, 7, 1, 0, 0, 2, 8},
        {4, 5, 7, 0, 1, 2, 0, 8},
        {4, 5, 7, 0, 1, 2, 0, 8}};

      std::size_t ysizes[] = {m.template size<ax::y_axis>(),
        m.template size<ax::y_axis, r::extended>(),
        m.template size<ax::y_axis, r::all>(),
        m.template size<ax::y_axis, r::boundary_low>(),
        m.template size<ax::y_axis, r::boundary_high>(),
        m.template size<ax::y_axis, r::ghost_low>(),
        m.template size<ax::y_axis, r::ghost_high>(),
        m.template size<ax::y_axis, r::global>()};

      for(int i = 0; i < 8; i++) {
        EXPECT_EQ(ysizes[i], ysizes_ex[process()][i]);
      }

      // check offsets
      std::size_t xoffsets_ex[4][8] = {{2, 0, 0, 0, 6, 0, 6, 0},
        {1, 1, 0, 1, 5, 0, 5, 3},
        {2, 0, 0, 0, 6, 0, 6, 0},
        {1, 1, 0, 1, 5, 0, 5, 3}};

      std::size_t xoffsets[] = {m.template offset<ax::x_axis>(),
        m.template offset<ax::x_axis, r::extended>(),
        m.template offset<ax::x_axis, r::all>(),
        m.template offset<ax::x_axis, r::boundary_low>(),
        m.template offset<ax::x_axis, r::boundary_high>(),
        m.template offset<ax::x_axis, r::ghost_low>(),
        m.template offset<ax::x_axis, r::ghost_high>(),
        m.template offset<ax::x_axis, r::global>()};

      for(int i = 0; i < 8; i++) {
        EXPECT_EQ(xoffsets[i], xoffsets_ex[process()][i]);
      }

      std::size_t yoffsets_ex[4][8] = {{1, 0, 0, 0, 5, 0, 5, 0},
        {1, 0, 0, 0, 5, 0, 5, 0},
        {2, 2, 0, 2, 6, 0, 6, 2},
        {2, 2, 0, 2, 6, 0, 6, 2}};

      std::size_t yoffsets[] = {m.template offset<ax::y_axis>(),
        m.template offset<ax::y_axis, r::extended>(),
        m.template offset<ax::y_axis, r::all>(),
        m.template offset<ax::y_axis, r::boundary_low>(),
        m.template offset<ax::y_axis, r::boundary_high>(),
        m.template offset<ax::y_axis, r::ghost_low>(),
        m.template offset<ax::y_axis, r::ghost_high>(),
        m.template offset<ax::y_axis, r::global>()};

      for(int i = 0; i < 8; i++) {
        EXPECT_EQ(yoffsets[i], yoffsets_ex[process()][i]);
      }

      // Check field values on the ghost layers
      // Layout used to identify neighbor ranks
      // for computing the correct field values
      // on the ghost layers.
      //  2  6  7  8
      //  1  3  4  5
      //  0  0  1  2
      //     0  1  2
      int ngb_ranks[4][9] = {{-1, -1, -1, -1, -1, 1, -1, 2, 3},
        {-1, -1, -1, 0, -1, -1, 2, 3},
        {-1, 0, 1, -1, -1, 3, -1, -1, -1},
        {0, 1, -1, 2, -1, -1, -1, -1, -1}};

      auto c = m.template mdspan<mesh2d::index_space::entities>(ca);

      auto chk =
        [&c](std::set<util::id> & ybnd, std::set<util::id> & xbnd, int r) {
          bool iseq = true;
          for(auto j : ybnd) {
            for(auto i : xbnd) {
              iseq = iseq && (c[j][i] == std::pow(10, r));
            }
          }
          return iseq;
        };

      EXPECT_EQ(
        chk(ylogical[rank], xghost_low[rank], ngb_ranks[rank][3]), true);
      EXPECT_EQ(
        chk(ylogical[rank], xghost_high[rank], ngb_ranks[rank][5]), true);
      EXPECT_EQ(
        chk(yghost_low[rank], xlogical[rank], ngb_ranks[rank][1]), true);
      EXPECT_EQ(
        chk(yghost_high[rank], xlogical[rank], ngb_ranks[rank][7]), true);

      EXPECT_EQ(
        chk(yghost_low[rank], xghost_low[rank], ngb_ranks[rank][0]), true);
      EXPECT_EQ(
        chk(yghost_low[rank], xghost_high[rank], ngb_ranks[rank][2]), true);
      EXPECT_EQ(
        chk(yghost_high[rank], xghost_low[rank], ngb_ranks[rank][6]), true);
      EXPECT_EQ(
        chk(yghost_high[rank], xghost_high[rank], ngb_ranks[rank][8]), true);
    };
  } // d=2
  else {
    UNIT {
      using r = mesh3d::domain;
      using ax = mesh3d::axis;

      std::set<util::id> xlogical[4] = {{1, 2}, {1, 2}, {1, 2}, {1, 2}};
      std::set<util::id> xextended[4] = {
        {0, 1, 2}, {1, 2, 3}, {0, 1, 2}, {1, 2, 3}};
      std::set<util::id> xall[4] = {
        {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}};
      std::set<util::id> xboundary_low[4] = {{0}, {}, {0}, {}};
      std::set<util::id> xboundary_high[4] = {{}, {3}, {}, {3}};
      std::set<util::id> xghost_low[4] = {{}, {0}, {}, {0}};
      std::set<util::id> xghost_high[4] = {{3}, {}, {3}, {}};

      std::set<util::id> ylogical[4] = {{1, 2}, {1, 2}, {1, 2}, {1, 2}};
      std::set<util::id> yextended[4] = {
        {0, 1, 2}, {0, 1, 2}, {1, 2, 3}, {1, 2, 3}};
      std::set<util::id> yall[4] = {
        {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}};
      std::set<util::id> yboundary_low[4] = {{0}, {0}, {}, {}};
      std::set<util::id> yboundary_high[4] = {{}, {}, {3}, {3}};
      std::set<util::id> yghost_low[4] = {{}, {}, {0}, {0}};
      std::set<util::id> yghost_high[4] = {{3}, {3}, {}, {}};

      std::set<util::id> zlogical[4] = {
        {1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}};
      std::set<util::id> zextended[4] = {{0, 1, 2, 3, 4, 5},
        {0, 1, 2, 3, 4, 5},
        {0, 1, 2, 3, 4, 5},
        {0, 1, 2, 3, 4, 5}};
      std::set<util::id> zall[4] = {{0, 1, 2, 3, 4, 5},
        {0, 1, 2, 3, 4, 5},
        {0, 1, 2, 3, 4, 5},
        {0, 1, 2, 3, 4, 5}};
      std::set<util::id> zboundary_low[4] = {{0}, {0}, {0}, {0}};
      std::set<util::id> zboundary_high[4] = {{5}, {5}, {5}, {5}};
      std::set<util::id> zghost_low[4] = {{}, {}, {}, {}};
      std::set<util::id> zghost_high[4] = {{}, {}, {}, {}};

      const int rank = process();

      const auto s = [](auto && r) {
        return std::set<util::id>(r.begin(), r.end());
      };

      EXPECT_EQ(s(m.template range<ax::x_axis>()), xlogical[rank]);
      EXPECT_EQ(
        s(m.template range<ax::x_axis, r::extended>()), xextended[rank]);
      EXPECT_EQ(s(m.template range<ax::x_axis, r::all>()), xall[rank]);
      EXPECT_EQ(s(m.template range<ax::x_axis, r::boundary_low>()),
        xboundary_low[rank]);
      EXPECT_EQ(s(m.template range<ax::x_axis, r::boundary_high>()),
        xboundary_high[rank]);
      EXPECT_EQ(
        s(m.template range<ax::x_axis, r::ghost_low>()), xghost_low[rank]);
      EXPECT_EQ(
        s(m.template range<ax::x_axis, r::ghost_high>()), xghost_high[rank]);

      EXPECT_EQ(s(m.template range<ax::y_axis>()), ylogical[rank]);
      EXPECT_EQ(
        s(m.template range<ax::y_axis, r::extended>()), yextended[rank]);
      EXPECT_EQ(s(m.template range<ax::y_axis, r::all>()), yall[rank]);
      EXPECT_EQ(s(m.template range<ax::y_axis, r::boundary_low>()),
        yboundary_low[rank]);
      EXPECT_EQ(s(m.template range<ax::y_axis, r::boundary_high>()),
        yboundary_high[rank]);
      EXPECT_EQ(
        s(m.template range<ax::y_axis, r::ghost_low>()), yghost_low[rank]);
      EXPECT_EQ(
        s(m.template range<ax::y_axis, r::ghost_high>()), yghost_high[rank]);

      EXPECT_EQ(s(m.template range<ax::z_axis>()), zlogical[rank]);
      EXPECT_EQ(
        s(m.template range<ax::z_axis, r::extended>()), zextended[rank]);
      EXPECT_EQ(s(m.template range<ax::z_axis, r::all>()), zall[rank]);
      EXPECT_EQ(s(m.template range<ax::z_axis, r::boundary_low>()),
        zboundary_low[rank]);
      EXPECT_EQ(s(m.template range<ax::z_axis, r::boundary_high>()),
        zboundary_high[rank]);
      EXPECT_EQ(
        s(m.template range<ax::z_axis, r::ghost_low>()), zghost_low[rank]);
      EXPECT_EQ(
        s(m.template range<ax::z_axis, r::ghost_high>()), zghost_high[rank]);

      // check sizes
      std::size_t xsizes_ex[4][8] = {{2, 3, 4, 1, 0, 0, 1, 4},
        {2, 3, 4, 0, 1, 1, 0, 4},
        {2, 3, 4, 1, 0, 0, 1, 4},
        {2, 3, 4, 0, 1, 1, 0, 4}};

      std::size_t xsizes[] = {m.template size<ax::x_axis>(),
        m.template size<ax::x_axis, r::extended>(),
        m.template size<ax::x_axis, r::all>(),
        m.template size<ax::x_axis, r::boundary_low>(),
        m.template size<ax::x_axis, r::boundary_high>(),
        m.template size<ax::x_axis, r::ghost_low>(),
        m.template size<ax::x_axis, r::ghost_high>(),
        m.template size<ax::x_axis, r::global>()};

      for(int i = 0; i < 8; i++) {
        EXPECT_EQ(xsizes[i], xsizes_ex[process()][i]);
      }

      std::size_t ysizes_ex[4][8] = {{2, 3, 4, 1, 0, 0, 1, 4},
        {2, 3, 4, 1, 0, 0, 1, 4},
        {2, 3, 4, 0, 1, 1, 0, 4},
        {2, 3, 4, 0, 1, 1, 0, 4}};

      std::size_t ysizes[] = {m.template size<ax::y_axis>(),
        m.template size<ax::y_axis, r::extended>(),
        m.template size<ax::y_axis, r::all>(),
        m.template size<ax::y_axis, r::boundary_low>(),
        m.template size<ax::y_axis, r::boundary_high>(),
        m.template size<ax::y_axis, r::ghost_low>(),
        m.template size<ax::y_axis, r::ghost_high>(),
        m.template size<ax::y_axis, r::global>()};

      for(int i = 0; i < 8; i++) {
        EXPECT_EQ(ysizes[i], ysizes_ex[process()][i]);
      }

      std::size_t zsizes_ex[4][8] = {{4, 6, 6, 1, 1, 0, 0, 4},
        {4, 6, 6, 1, 1, 0, 0, 4},
        {4, 6, 6, 1, 1, 0, 0, 4},
        {4, 6, 6, 1, 1, 0, 0, 4}};

      std::size_t zsizes[] = {m.template size<ax::z_axis>(),
        m.template size<ax::z_axis, r::extended>(),
        m.template size<ax::z_axis, r::all>(),
        m.template size<ax::z_axis, r::boundary_low>(),
        m.template size<ax::z_axis, r::boundary_high>(),
        m.template size<ax::z_axis, r::ghost_low>(),
        m.template size<ax::z_axis, r::ghost_high>(),
        m.template size<ax::z_axis, r::global>()};

      for(int i = 0; i < 8; i++) {
        EXPECT_EQ(zsizes[i], zsizes_ex[process()][i]);
      }

      // check offsets
      std::size_t xoffsets_ex[4][8] = {{1, 0, 0, 0, 3, 0, 3, 0},
        {1, 1, 0, 1, 3, 0, 3, 1},
        {1, 0, 0, 0, 3, 0, 3, 0},
        {1, 1, 0, 1, 3, 0, 3, 1}};

      std::size_t xoffsets[] = {m.template offset<ax::x_axis>(),
        m.template offset<ax::x_axis, r::extended>(),
        m.template offset<ax::x_axis, r::all>(),
        m.template offset<ax::x_axis, r::boundary_low>(),
        m.template offset<ax::x_axis, r::boundary_high>(),
        m.template offset<ax::x_axis, r::ghost_low>(),
        m.template offset<ax::x_axis, r::ghost_high>(),
        m.template offset<ax::x_axis, r::global>()};

      for(int i = 0; i < 8; i++) {
        EXPECT_EQ(xoffsets[i], xoffsets_ex[process()][i]);
      }

      std::size_t yoffsets_ex[4][8] = {{1, 0, 0, 0, 3, 0, 3, 0},
        {1, 0, 0, 0, 3, 0, 3, 0},
        {1, 1, 0, 1, 3, 0, 3, 1},
        {1, 1, 0, 1, 3, 0, 3, 1}};

      std::size_t yoffsets[] = {m.template offset<ax::y_axis>(),
        m.template offset<ax::y_axis, r::extended>(),
        m.template offset<ax::y_axis, r::all>(),
        m.template offset<ax::y_axis, r::boundary_low>(),
        m.template offset<ax::y_axis, r::boundary_high>(),
        m.template offset<ax::y_axis, r::ghost_low>(),
        m.template offset<ax::y_axis, r::ghost_high>(),
        m.template offset<ax::y_axis, r::global>()};
      for(int i = 0; i < 8; i++) {
        EXPECT_EQ(yoffsets[i], yoffsets_ex[process()][i]);
      }

      std::size_t zoffsets_ex[4][8] = {{1, 0, 0, 0, 5, 0, 5, 0},
        {1, 0, 0, 0, 5, 0, 5, 0},
        {1, 0, 0, 0, 5, 0, 5, 0},
        {1, 0, 0, 0, 5, 0, 5, 0}};

      std::size_t zoffsets[] = {m.template offset<ax::z_axis>(),
        m.template offset<ax::z_axis, r::extended>(),
        m.template offset<ax::z_axis, r::all>(),
        m.template offset<ax::z_axis, r::boundary_low>(),
        m.template offset<ax::z_axis, r::boundary_high>(),
        m.template offset<ax::z_axis, r::ghost_low>(),
        m.template offset<ax::z_axis, r::ghost_high>(),
        m.template offset<ax::z_axis, r::global>()};

      for(int i = 0; i < 8; i++) {
        EXPECT_EQ(zoffsets[i], zoffsets_ex[process()][i]);
      }

      // Check field values on the ghost layers
      // For this particular mesh, the partition
      // is 2x2 over the x-y plane, so, only the edge-connected
      // cells (diagonals) are the corners, so same neighbor rank
      // datastructure as 2d case can be used here.
      //  2  6  7  8
      //  1  3  4  5
      //  0  0  1  2
      //     0  1  2
      int ngb_ranks[4][9] = {{-1, -1, -1, -1, -1, 1, -1, 2, 3},
        {-1, -1, -1, 0, -1, -1, 2, 3, -1},
        {-1, 0, 1, -1, -1, 3, -1, -1, -1},
        {0, 1, -1, 2, -1, -1, -1, -1, -1}};

      auto c = m.template mdspan<mesh3d::index_space::entities>(ca);

      auto chk = [&c](std::set<util::id> & zbnd,
                   std::set<util::id> & ybnd,
                   std::set<util::id> & xbnd,
                   int r) {
        bool iseq = true;
        for(auto k : zbnd) {
          for(auto j : ybnd) {
            for(auto i : xbnd) {
              iseq = iseq && (c[k][j][i] == std::pow(10, r));
            }
          }
        }
        return iseq;
      };

      EXPECT_EQ(
        chk(
          zlogical[rank], ylogical[rank], xghost_low[rank], ngb_ranks[rank][3]),
        true);
      EXPECT_EQ(chk(zlogical[rank],
                  ylogical[rank],
                  xghost_high[rank],
                  ngb_ranks[rank][5]),
        true);
      EXPECT_EQ(
        chk(
          zlogical[rank], yghost_low[rank], xlogical[rank], ngb_ranks[rank][1]),
        true);
      EXPECT_EQ(chk(zlogical[rank],
                  yghost_high[rank],
                  xlogical[rank],
                  ngb_ranks[rank][7]),
        true);

      EXPECT_EQ(chk(zlogical[rank],
                  yghost_low[rank],
                  xghost_low[rank],
                  ngb_ranks[rank][0]),
        true);
      EXPECT_EQ(chk(zlogical[rank],
                  yghost_low[rank],
                  xghost_high[rank],
                  ngb_ranks[rank][2]),
        true);
      EXPECT_EQ(chk(zlogical[rank],
                  yghost_high[rank],
                  xghost_low[rank],
                  ngb_ranks[rank][6]),
        true);
      EXPECT_EQ(chk(zlogical[rank],
                  yghost_high[rank],
                  xghost_high[rank],
                  ngb_ranks[rank][8]),
        true);
    };
  } // d=3
} // check_mesh_field

int
check_4dmesh(mesh<4>::accessor<ro> m) {
  UNIT {
    using r = mesh4d::domain;
    using ax = mesh4d::axis;

    std::set<util::id> logical[2] = {{1, 2}, {1, 2}};
    std::set<util::id> extended[2] = {{0, 1, 2}, {1, 2, 3}};
    std::set<util::id> all[2] = {{0, 1, 2, 3}, {0, 1, 2, 3}};
    std::set<util::id> ghost_low[2] = {{}, {0}};
    std::set<util::id> ghost_high[2] = {{3}, {}};
    std::set<util::id> boundary_low[2] = {{0}, {}};
    std::set<util::id> boundary_high[2] = {{}, {3}};

    const int rank = process();
    const int nparts = 2; //#num colors on each axis

    const auto indices = [&](auto && offset) {
      auto rem = offset;
      std::vector<int> idx(4);
      for(auto dim = 0; dim < 4; ++dim) {
        idx[dim] = rem % nparts;
        rem = (rem - idx[dim]) / nparts;
      }
      return idx;
    };

    const auto s = [](auto && r) {
      return std::set<util::id>(r.begin(), r.end());
    };

    auto idx = indices(rank);

    EXPECT_EQ(s(m.range<ax::x_axis>()), logical[idx[0]]);
    EXPECT_EQ(s(m.range<ax::x_axis, r::extended>()), extended[idx[0]]);
    EXPECT_EQ(s(m.range<ax::x_axis, r::all>()), all[idx[0]]);
    EXPECT_EQ(s(m.range<ax::x_axis, r::ghost_low>()), ghost_low[idx[0]]);
    EXPECT_EQ(s(m.range<ax::x_axis, r::ghost_high>()), ghost_high[idx[0]]);
    EXPECT_EQ(s(m.range<ax::x_axis, r::boundary_low>()), boundary_low[idx[0]]);
    EXPECT_EQ(
      s(m.range<ax::x_axis, r::boundary_high>()), boundary_high[idx[0]]);

    EXPECT_EQ(s(m.range<ax::y_axis>()), logical[idx[1]]);
    EXPECT_EQ(s(m.range<ax::y_axis, r::extended>()), extended[idx[1]]);
    EXPECT_EQ(s(m.range<ax::y_axis, r::all>()), all[idx[1]]);
    EXPECT_EQ(s(m.range<ax::y_axis, r::ghost_low>()), ghost_low[idx[1]]);
    EXPECT_EQ(s(m.range<ax::y_axis, r::ghost_high>()), ghost_high[idx[1]]);
    EXPECT_EQ(s(m.range<ax::y_axis, r::boundary_low>()), boundary_low[idx[1]]);
    EXPECT_EQ(
      s(m.range<ax::y_axis, r::boundary_high>()), boundary_high[idx[1]]);

    EXPECT_EQ(s(m.range<ax::z_axis>()), logical[idx[2]]);
    EXPECT_EQ(s(m.range<ax::z_axis, r::extended>()), extended[idx[2]]);
    EXPECT_EQ(s(m.range<ax::z_axis, r::all>()), all[idx[2]]);
    EXPECT_EQ(s(m.range<ax::z_axis, r::ghost_low>()), ghost_low[idx[2]]);
    EXPECT_EQ(s(m.range<ax::z_axis, r::ghost_high>()), ghost_high[idx[2]]);
    EXPECT_EQ(s(m.range<ax::z_axis, r::boundary_low>()), boundary_low[idx[2]]);
    EXPECT_EQ(
      s(m.range<ax::z_axis, r::boundary_high>()), boundary_high[idx[2]]);

    EXPECT_EQ(s(m.range<ax::t_axis>()), logical[idx[3]]);
    EXPECT_EQ(s(m.range<ax::t_axis, r::extended>()), extended[idx[3]]);
    EXPECT_EQ(s(m.range<ax::t_axis, r::all>()), all[idx[3]]);
    EXPECT_EQ(s(m.range<ax::t_axis, r::ghost_low>()), ghost_low[idx[3]]);
    EXPECT_EQ(s(m.range<ax::t_axis, r::ghost_high>()), ghost_high[idx[3]]);
    EXPECT_EQ(s(m.range<ax::t_axis, r::boundary_low>()), boundary_low[idx[3]]);
    EXPECT_EQ(
      s(m.range<ax::t_axis, r::boundary_high>()), boundary_high[idx[3]]);
  };
} // check_4dmesh
