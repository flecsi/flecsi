
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

template<std::size_t D>
void
init_field(typename mesh<D>::template accessor<ro> m,
  field<std::size_t>::accessor<wo, na> ca) {
  auto c = m.template mdspan<mesh<D>::index_space::entities>(ca);

  if constexpr(D == 1) {
    for(auto i : m.template extents<mesh1d::axis::x_axis>()) {
      c[i] = color();
    } // for
  }
  else if constexpr(D == 2) {
    for(auto j : m.template extents<mesh2d::axis::y_axis>()) {
      for(auto i : m.template extents<mesh2d::axis::x_axis>()) {
        c[j][i] = color();
      } // for
    } // for
  }
  else {
    for(auto k : m.template extents<mesh3d::axis::z_axis>()) {
      for(auto j : m.template extents<mesh3d::axis::y_axis>()) {
        for(auto i : m.template extents<mesh3d::axis::x_axis>()) {
          c[k][j][i] = color();
        } // for
      } // for
    } // for
  }
} // init_field

template<std::size_t D>
void
print_field(typename mesh<D>::template accessor<ro> m,
  field<std::size_t>::accessor<ro, ro> ca) {
  auto c = m.template mdspan<mesh<D>::index_space::entities>(ca);
  std::stringstream ss;
  if constexpr(D == 1) {
    for(auto i :
      m.template extents<mesh1d::axis::x_axis, mesh1d::range::all>()) {
      ss << c[i] << "   ";
    } // for
    ss << std::endl;
    flog(warn) << ss.str() << std::endl;
  }
  else if constexpr(D == 2) {
    for(int j = m.template size<mesh2d::axis::y_axis, mesh2d::range::all>() - 1;
        j >= 0;
        --j) {
      for(auto i :
        m.template extents<mesh2d::axis::x_axis, mesh2d::range::all>()) {
        ss << c[j][i] << "   ";
      } // for
      ss << std::endl;
    } // for
    flog(warn) << ss.str() << std::endl;
  }
  else {
    for(int k = m.template size<mesh3d::axis::z_axis, mesh3d::range::all>() - 1;
        k >= 0;
        --k) {
      for(int j =
            m.template size<mesh3d::axis::y_axis, mesh3d::range::all>() - 1;
          j >= 0;
          --j) {
        for(auto i :
          m.template extents<mesh3d::axis::x_axis, mesh3d::range::all>()) {
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
void
update_field(typename mesh<D>::template accessor<ro> m,
  field<std::size_t>::accessor<wo, na> ca) {
  auto r = color();
  auto c = m.template mdspan<mesh<D>::index_space::entities>(ca);

  if constexpr(D == 1) {
    for(auto i : m.template extents<mesh1d::axis::x_axis>()) {
      c[i] = std::pow(10, r);
    } // for
  }
  else if constexpr(D == 2) {
    for(auto j : m.template extents<mesh2d::axis::y_axis>()) {
      for(auto i : m.template extents<mesh2d::axis::x_axis>()) {
        c[j][i] = std::pow(10, r);
      } // for
    } // for
  }
  else {
    for(auto k : m.template extents<mesh3d::axis::z_axis>()) {
      for(auto j : m.template extents<mesh3d::axis::y_axis>()) {
        for(auto i : m.template extents<mesh3d::axis::x_axis>()) {
          c[k][j][i] = std::pow(10, r);
        } // for
      } // for
    } // for
  }
} // update_field

template<std::size_t D>
int
check_mesh_field(typename mesh<D>::template accessor<ro> m,
  field<std::size_t>::accessor<ro, ro> ca) {
  if constexpr(D == 1) {
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

      EXPECT_EQ(s(m.template extents<ax::x_axis>()), logical[rank]);
      EXPECT_EQ(
        s(m.template extents<ax::x_axis, r::extended>()), extended[rank]);
      EXPECT_EQ(s(m.template extents<ax::x_axis, r::all>()), all[rank]);
      EXPECT_EQ(s(m.template extents<ax::x_axis, r::boundary_low>()),
        boundary_low[rank]);
      EXPECT_EQ(s(m.template extents<ax::x_axis, r::boundary_high>()),
        boundary_high[rank]);
      EXPECT_EQ(
        s(m.template extents<ax::x_axis, r::ghost_low>()), ghost_low[rank]);
      EXPECT_EQ(
        s(m.template extents<ax::x_axis, r::ghost_high>()), ghost_high[rank]);

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
        for(auto i = ghost_low[rank].begin(); i != ghost_low[rank].end(); ++i) {
          EXPECT_EQ(c[*i], std::pow(10, ngb_ranks[rank][0]));
        }
      }

      if(ngb_ranks[rank][1] != -1) {
        for(auto i = ghost_high[rank].begin(); i != ghost_high[rank].end();
            ++i) {
          EXPECT_EQ(c[*i], std::pow(10, ngb_ranks[rank][1]));
        }
      }
    };
  } // d=1
  else if constexpr(D == 2) {
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

      EXPECT_EQ(s(m.template extents<ax::x_axis>()), xlogical[rank]);
      EXPECT_EQ(
        s(m.template extents<ax::x_axis, r::extended>()), xextended[rank]);
      EXPECT_EQ(s(m.template extents<ax::x_axis, r::all>()), xall[rank]);
      EXPECT_EQ(s(m.template extents<ax::x_axis, r::boundary_low>()),
        xboundary_low[rank]);
      EXPECT_EQ(s(m.template extents<ax::x_axis, r::boundary_high>()),
        xboundary_high[rank]);
      EXPECT_EQ(
        s(m.template extents<ax::x_axis, r::ghost_low>()), xghost_low[rank]);
      EXPECT_EQ(
        s(m.template extents<ax::x_axis, r::ghost_high>()), xghost_high[rank]);

      EXPECT_EQ(s(m.template extents<ax::y_axis>()), ylogical[rank]);
      EXPECT_EQ(
        s(m.template extents<ax::y_axis, r::extended>()), yextended[rank]);
      EXPECT_EQ(s(m.template extents<ax::y_axis, r::all>()), yall[rank]);
      EXPECT_EQ(s(m.template extents<ax::y_axis, r::boundary_low>()),
        yboundary_low[rank]);
      EXPECT_EQ(s(m.template extents<ax::y_axis, r::boundary_high>()),
        yboundary_high[rank]);
      EXPECT_EQ(
        s(m.template extents<ax::y_axis, r::ghost_low>()), yghost_low[rank]);
      EXPECT_EQ(
        s(m.template extents<ax::y_axis, r::ghost_high>()), yghost_high[rank]);

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

      // left
      for(auto j = ylogical[rank].begin(); j != ylogical[rank].end(); ++j) {
        for(auto i = xghost_low[rank].begin(); i != xghost_low[rank].end();
            ++i) {
          EXPECT_EQ(c[*j][*i], std::pow(10, ngb_ranks[rank][3]));
        }
      }

      // right
      for(auto j = ylogical[rank].begin(); j != ylogical[rank].end(); ++j) {
        for(auto i = xghost_high[rank].begin(); i != xghost_high[rank].end();
            ++i) {
          EXPECT_EQ(c[*j][*i], std::pow(10, ngb_ranks[rank][5]));
        }
      }

      // down
      for(auto i = xlogical[rank].begin(); i != xlogical[rank].end(); ++i) {
        for(auto j = yghost_low[rank].begin(); j != yghost_low[rank].end();
            ++j) {
          EXPECT_EQ(c[*j][*i], std::pow(10, ngb_ranks[rank][1]));
        }
      }

      // up
      for(auto i = xlogical[rank].begin(); i != xlogical[rank].end(); ++i) {
        for(auto j = yghost_high[rank].begin(); j != yghost_high[rank].end();
            ++j) {
          EXPECT_EQ(c[*j][*i], std::pow(10, ngb_ranks[rank][7]));
        }
      }

      // corners
      for(auto j = yghost_low[rank].begin(); j != yghost_low[rank].end(); ++j) {
        for(auto i = xghost_low[rank].begin(); i != xghost_low[rank].end();
            ++i) {
          EXPECT_EQ(c[*j][*i], std::pow(10, ngb_ranks[rank][0]));
        }
      }

      for(auto j = yghost_low[rank].begin(); j != yghost_low[rank].end(); ++j) {
        for(auto i = xghost_high[rank].begin(); i != xghost_high[rank].end();
            ++i) {
          EXPECT_EQ(c[*j][*i], std::pow(10, ngb_ranks[rank][2]));
        }
      }

      for(auto j = yghost_high[rank].begin(); j != yghost_high[rank].end();
          ++j) {
        for(auto i = xghost_low[rank].begin(); i != xghost_low[rank].end();
            ++i) {
          EXPECT_EQ(c[*j][*i], std::pow(10, ngb_ranks[rank][6]));
        }
      }

      for(auto j = yghost_high[rank].begin(); j != yghost_high[rank].end();
          ++j) {
        for(auto i = xghost_high[rank].begin(); i != xghost_high[rank].end();
            ++i) {
          EXPECT_EQ(c[*j][*i], std::pow(10, ngb_ranks[rank][8]));
        }
      }
    };
  } // d=2
  else {
    UNIT {
      using r = mesh3d::range;
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

      EXPECT_EQ(s(m.template extents<ax::x_axis>()), xlogical[rank]);
      EXPECT_EQ(
        s(m.template extents<ax::x_axis, r::extended>()), xextended[rank]);
      EXPECT_EQ(s(m.template extents<ax::x_axis, r::all>()), xall[rank]);
      EXPECT_EQ(s(m.template extents<ax::x_axis, r::boundary_low>()),
        xboundary_low[rank]);
      EXPECT_EQ(s(m.template extents<ax::x_axis, r::boundary_high>()),
        xboundary_high[rank]);
      EXPECT_EQ(
        s(m.template extents<ax::x_axis, r::ghost_low>()), xghost_low[rank]);
      EXPECT_EQ(
        s(m.template extents<ax::x_axis, r::ghost_high>()), xghost_high[rank]);

      EXPECT_EQ(s(m.template extents<ax::y_axis>()), ylogical[rank]);
      EXPECT_EQ(
        s(m.template extents<ax::y_axis, r::extended>()), yextended[rank]);
      EXPECT_EQ(s(m.template extents<ax::y_axis, r::all>()), yall[rank]);
      EXPECT_EQ(s(m.template extents<ax::y_axis, r::boundary_low>()),
        yboundary_low[rank]);
      EXPECT_EQ(s(m.template extents<ax::y_axis, r::boundary_high>()),
        yboundary_high[rank]);
      EXPECT_EQ(
        s(m.template extents<ax::y_axis, r::ghost_low>()), yghost_low[rank]);
      EXPECT_EQ(
        s(m.template extents<ax::y_axis, r::ghost_high>()), yghost_high[rank]);

      EXPECT_EQ(s(m.template extents<ax::z_axis>()), zlogical[rank]);
      EXPECT_EQ(
        s(m.template extents<ax::z_axis, r::extended>()), zextended[rank]);
      EXPECT_EQ(s(m.template extents<ax::z_axis, r::all>()), zall[rank]);
      EXPECT_EQ(s(m.template extents<ax::z_axis, r::boundary_low>()),
        zboundary_low[rank]);
      EXPECT_EQ(s(m.template extents<ax::z_axis, r::boundary_high>()),
        zboundary_high[rank]);
      EXPECT_EQ(
        s(m.template extents<ax::z_axis, r::ghost_low>()), zghost_low[rank]);
      EXPECT_EQ(
        s(m.template extents<ax::z_axis, r::ghost_high>()), zghost_high[rank]);

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

      // left
      for(auto k = zlogical[rank].begin(); k != zlogical[rank].end(); ++k) {
        for(auto j = ylogical[rank].begin(); j != ylogical[rank].end(); ++j) {
          for(auto i = xghost_low[rank].begin(); i != xghost_low[rank].end();
              ++i) {
            EXPECT_EQ(c[*k][*j][*i], std::pow(10, ngb_ranks[rank][3]));
          }
        }
      }

      // right
      for(auto k = zlogical[rank].begin(); k != zlogical[rank].end(); ++k) {
        for(auto j = ylogical[rank].begin(); j != ylogical[rank].end(); ++j) {
          for(auto i = xghost_high[rank].begin(); i != xghost_high[rank].end();
              ++i) {
            EXPECT_EQ(c[*k][*j][*i], std::pow(10, ngb_ranks[rank][5]));
          }
        }
      }

      // down
      for(auto k = zlogical[rank].begin(); k != zlogical[rank].end(); ++k) {
        for(auto i = xlogical[rank].begin(); i != xlogical[rank].end(); ++i) {
          for(auto j = yghost_low[rank].begin(); j != yghost_low[rank].end();
              ++j) {
            EXPECT_EQ(c[*k][*j][*i], std::pow(10, ngb_ranks[rank][1]));
          }
        }
      }

      // up
      for(auto k = zlogical[rank].begin(); k != zlogical[rank].end(); ++k) {
        for(auto i = xlogical[rank].begin(); i != xlogical[rank].end(); ++i) {
          for(auto j = yghost_high[rank].begin(); j != yghost_high[rank].end();
              ++j) {
            EXPECT_EQ(c[*k][*j][*i], std::pow(10, ngb_ranks[rank][7]));
          }
        }
      }

      // corners
      for(auto k = zlogical[rank].begin(); k != zlogical[rank].end(); ++k) {
        for(auto j = yghost_low[rank].begin(); j != yghost_low[rank].end();
            ++j) {
          for(auto i = xghost_low[rank].begin(); i != xghost_low[rank].end();
              ++i) {
            EXPECT_EQ(c[*k][*j][*i], std::pow(10, ngb_ranks[rank][0]));
          }
        }
      }

      for(auto k = zlogical[rank].begin(); k != zlogical[rank].end(); ++k) {
        for(auto j = yghost_low[rank].begin(); j != yghost_low[rank].end();
            ++j) {
          for(auto i = xghost_high[rank].begin(); i != xghost_high[rank].end();
              ++i) {
            EXPECT_EQ(c[*k][*j][*i], std::pow(10, ngb_ranks[rank][2]));
          }
        }
      }

      for(auto k = zlogical[rank].begin(); k != zlogical[rank].end(); ++k) {
        for(auto j = yghost_high[rank].begin(); j != yghost_high[rank].end();
            ++j) {
          for(auto i = xghost_low[rank].begin(); i != xghost_low[rank].end();
              ++i) {
            EXPECT_EQ(c[*k][*j][*i], std::pow(10, ngb_ranks[rank][6]));
          }
        }
      }

      for(auto k = zlogical[rank].begin(); k != zlogical[rank].end(); ++k) {
        for(auto j = yghost_high[rank].begin(); j != yghost_high[rank].end();
            ++j) {
          for(auto i = xghost_high[rank].begin(); i != xghost_high[rank].end();
              ++i) {
            EXPECT_EQ(c[*k][*j][*i], std::pow(10, ngb_ranks[rank][8]));
          }
        }
      }
    };
  } // d=3
} // check_mesh_field
