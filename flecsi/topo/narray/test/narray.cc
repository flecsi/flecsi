#include "narray.hh"

#include "flecsi/execution.hh"
#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"

#include <set>

using namespace flecsi;

using mesh1d = mesh<1>;
using mesh2d = mesh<2>;
using mesh3d = mesh<3>;
using mesh4d = mesh<4>;

template<std::size_t D, typename F>
void
field_helper(typename mesh<D>::template accessor<ro> m,
  field<std::size_t>::accessor<wo, na> ca,
  F && fvalue) {
  auto c = m.template mdspan<topo::elements>(ca);

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
  auto c = m.template mdspan<topo::elements>(ca);
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
    UNIT("TASK") {
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
        {1, 1, 0, 1, 3, 0, 3, 3},
        {1, 1, 0, 1, 3, 0, 3, 5},
        {1, 1, 0, 1, 3, 0, 3, 7}};

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
      auto c = m.template mdspan<topo::elements>(ca);

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
    UNIT("TASK") {
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
        {1, 1, 0, 1, 5, 0, 5, 4},
        {2, 0, 0, 0, 6, 0, 6, 0},
        {1, 1, 0, 1, 5, 0, 5, 4}};

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
        {2, 2, 0, 2, 6, 0, 6, 4},
        {2, 2, 0, 2, 6, 0, 6, 4}};

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

      auto c = m.template mdspan<topo::elements>(ca);

      auto chk =
        [&c](std::set<util::id> & ybnd, std::set<util::id> & xbnd, int r) {
          bool iseq = true;
          for(auto j : ybnd) {
            for(auto i : xbnd) {
              iseq = iseq && (c[j][i] == size_t(std::pow(10, r)));
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
    UNIT("TASK") {
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
        {1, 1, 0, 1, 3, 0, 3, 2},
        {1, 0, 0, 0, 3, 0, 3, 0},
        {1, 1, 0, 1, 3, 0, 3, 2}};

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
        {1, 1, 0, 1, 3, 0, 3, 2},
        {1, 1, 0, 1, 3, 0, 3, 2}};

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

      auto c = m.template mdspan<topo::elements>(ca);

      auto chk = [&c](std::set<util::id> & zbnd,
                   std::set<util::id> & ybnd,
                   std::set<util::id> & xbnd,
                   int r) {
        bool iseq = true;
        for(auto k : zbnd) {
          for(auto j : ybnd) {
            for(auto i : xbnd) {
              iseq = iseq && (c[k][j][i] == size_t(std::pow(10, r)));
            }
          }
        }
        return iseq;
      };

      // clang-format off
      EXPECT_EQ(
        chk(zlogical[rank],
          ylogical[rank],
          xghost_low[rank],
          ngb_ranks[rank][3]),
        true
      );
      EXPECT_EQ(
        chk(zlogical[rank],
          ylogical[rank],
          xghost_high[rank],
          ngb_ranks[rank][5]),
        true
      );
      EXPECT_EQ(
        chk(zlogical[rank],
          yghost_low[rank],
          xlogical[rank],
          ngb_ranks[rank][1]),
        true
      );
      EXPECT_EQ(
        chk(zlogical[rank],
          yghost_high[rank],
          xlogical[rank],
          ngb_ranks[rank][7]),
        true
      );
      EXPECT_EQ(
        chk(zlogical[rank],
          yghost_low[rank],
          xghost_low[rank],
          ngb_ranks[rank][0]),
        true
      );
      EXPECT_EQ(
        chk(zlogical[rank],
          yghost_low[rank],
          xghost_high[rank],
          ngb_ranks[rank][2]),
        true
      );
      EXPECT_EQ(
        chk(zlogical[rank],
          yghost_high[rank],
          xghost_low[rank],
          ngb_ranks[rank][6]),
        true
      );
      EXPECT_EQ(
        chk(zlogical[rank],
          yghost_high[rank],
          xghost_high[rank],
          ngb_ranks[rank][8]),
        true
      );
      // clang-format on
    };
  } // d=3
} // check_mesh_field

int
coloring_driver() {
  UNIT() {
    // 9x9x9 with 3x3x1 colors, extend x-axis, full_ghosts: no
    //
    // 0   1   2   3   (3)  4   5   6   (6)  7   8   9
    // +---+---+---+    +---+---+---+    +---+---+---+
    // | 0 | 1 | 2 |    | 3 | 4 | 5 |    | 6 | 7 | 8 |
    // +---+---+---+    +---+---+---+    +---+---+---+

    mesh3d::gcoord indices{9, 9, 9};

    mesh3d::index_definition idef;
    idef.axes = topo::narray_utils::make_axes(9, indices);
    for(auto & a : idef.axes) {
      a.hdepth = 1;
    }

    auto coloring = idef.process_coloring(MPI_COMM_WORLD);
    auto adef = idef;
    adef.axes[0].auxiliary = true;
    adef.full_ghosts = false;
    auto avpc = adef.process_coloring(MPI_COMM_WORLD);

    mesh3d::gcoord global;
    std::map<Color, mesh3d::gcoord> extents, offset;
    std::map<Color, mesh3d::coord> logical_low, logical_high;
    std::map<Color, mesh3d::coord> extended_low, extended_high;

    global = {10, 9, 9};

    for(Color c = 0; c < 9; ++c) {
      extents[c] = {4, 3, 9};
      extended_high[c] = logical_high[c] = {4, 3, 9};
    }

    offset[0] = {0, 0, 0};
    offset[1] = {4, 0, 0};
    offset[2] = {7, 0, 0};
    offset[3] = {0, 3, 0};
    offset[4] = {4, 3, 0};
    offset[5] = {7, 3, 0};
    offset[6] = {0, 6, 0};
    offset[7] = {4, 6, 0};
    offset[8] = {7, 6, 0};

    extended_low[0] = logical_low[0] = {0, 0, 0};
    extended_low[1] = logical_low[1] = {1, 0, 0};
    extended_low[2] = logical_low[2] = {1, 0, 0};
    extended_low[3] = logical_low[3] = {0, 0, 0};
    extended_low[4] = logical_low[4] = {1, 0, 0};
    extended_low[5] = logical_low[5] = {1, 0, 0};
    extended_low[6] = logical_low[6] = {0, 0, 0};
    extended_low[7] = logical_low[7] = {1, 0, 0};
    extended_low[8] = logical_low[8] = {1, 0, 0};

    for(auto & p : avpc) {
      auto c = p.color();
      for(Dimension axis = 0; axis < 3; ++axis) {
        auto & axco = p.axis_colors[axis];
        EXPECT_EQ(axco.global(), global[axis]);
        EXPECT_EQ(axco.extent(), extents[c][axis]);
        EXPECT_EQ(axco.offset(), offset[c][axis]);
        EXPECT_EQ(axco.logical<0>(), logical_low[c][axis]);
        EXPECT_EQ(axco.logical<1>(), logical_high[c][axis]);
        EXPECT_EQ(axco.extended<0>(), extended_low[c][axis]);
        EXPECT_EQ(axco.extended<1>(), extended_high[c][axis]);
      }
    }

    auto print_colorings = [&]() {
      std::stringstream ss;
      ss << "primary" << std::endl;
      for(auto p : coloring) {
        ss << p << std::endl;
      } // for
      ss << "auxiliary" << std::endl;
      for(auto p : avpc) {
        ss << p << std::endl;
      } // for
      flog(warn) << ss.str() << std::endl;
    };

    print_colorings();

    // 9x9x9 with 3x3x1 colors, extend x-axis, full_ghosts: yes
    //
    // 0   1   2   3   4   (2) (3)  4   5   6  (7)  (5) (6)  7   8   9
    // +---+---+---+ ~ +    + ~ +---+---+---+ ~ +    + ~ +---+---+---+
    // | 0 | 1 | 2 |(3)|    |(2)| 3 | 4 | 5 |(6)|    |(5)| 6 | 7 | 8 |
    // +---+---+---+ ~ +    + ~ +---+---+---+ ~ +    + ~ +---+---+---+

    adef.full_ghosts = true;
    avpc = adef.process_coloring(MPI_COMM_WORLD);

    extents[0] = {5, 4, 9};
    extents[1] = {6, 4, 9};
    extents[2] = {5, 4, 9};
    extents[3] = {5, 5, 9};
    extents[4] = {6, 5, 9};
    extents[5] = {5, 5, 9};
    extents[6] = {5, 4, 9};
    extents[7] = {6, 4, 9};
    extents[8] = {5, 4, 9};

    offset[0] = {0, 0, 0};
    offset[1] = {4, 0, 0};
    offset[2] = {7, 0, 0};
    offset[3] = {0, 3, 0};
    offset[4] = {4, 3, 0};
    offset[5] = {7, 3, 0};
    offset[6] = {0, 6, 0};
    offset[7] = {4, 6, 0};
    offset[8] = {7, 6, 0};

    extended_low[0] = logical_low[0] = {0, 0, 0};
    extended_low[1] = logical_low[1] = {2, 0, 0};
    extended_low[2] = logical_low[2] = {2, 0, 0};
    extended_low[3] = logical_low[3] = {0, 1, 0};
    extended_low[4] = logical_low[4] = {2, 1, 0};
    extended_low[5] = logical_low[5] = {2, 1, 0};
    extended_low[6] = logical_low[6] = {0, 1, 0};
    extended_low[7] = logical_low[7] = {2, 1, 0};
    extended_low[8] = logical_low[8] = {2, 1, 0};

    extended_high[0] = logical_high[0] = {4, 3, 9};
    extended_high[1] = logical_high[1] = {5, 3, 9};
    extended_high[2] = logical_high[2] = {5, 3, 9};
    extended_high[3] = logical_high[3] = {4, 4, 9};
    extended_high[4] = logical_high[4] = {5, 4, 9};
    extended_high[5] = logical_high[5] = {5, 4, 9};
    extended_high[6] = logical_high[6] = {4, 4, 9};
    extended_high[7] = logical_high[7] = {5, 4, 9};
    extended_high[8] = logical_high[8] = {5, 4, 9};

    for(auto & p : avpc) {
      auto c = p.color();
      for(Dimension axis = 0; axis < 3; ++axis) {
        auto & axco = p.axis_colors[axis];
        EXPECT_EQ(axco.global(), global[axis]);
        EXPECT_EQ(axco.extent(), extents[c][axis]);
        EXPECT_EQ(axco.offset(), offset[c][axis]);
        EXPECT_EQ(axco.logical<0>(), logical_low[c][axis]);
        EXPECT_EQ(axco.logical<1>(), logical_high[c][axis]);
        EXPECT_EQ(axco.extended<0>(), extended_low[c][axis]);
        EXPECT_EQ(axco.extended<1>(), extended_high[c][axis]);
      }
    }

    print_colorings();
  };
}

flecsi::unit::driver<coloring_driver> cd;

// 1D Mesh
mesh1d::slot m1;
mesh1d::cslot coloring1;
const field<std::size_t>::definition<mesh1d> f1;

// 2D Mesh
mesh2d::slot m2;
mesh2d::cslot coloring2;
const field<std::size_t>::definition<mesh2d> f2;

// 3D Mesh
mesh3d::slot m3;
mesh3d::cslot coloring3;
const field<std::size_t>::definition<mesh3d> f3;

// 4D Mesh
mesh4d::slot m4;
mesh4d::cslot coloring4;

int
check_contiguous(data::multi<mesh1d::accessor<ro>> mm) {
  UNIT() {
    constexpr static auto x = mesh1d::axis::x_axis;
    using D = mesh1d::domain;
    std::size_t last, total = 0;
    for(auto [c, m] : mm.components()) { // presumed to be in order
      auto sz = m.size<x, D::global>(), off = m.offset<x, D::global>();
      if(total) {
        EXPECT_EQ(sz, total);
        EXPECT_EQ(last, off);
      }
      else {
        total = sz;
      }
      last = off + (m.offset<x, D::ghost_high>() - m.offset<x, D::logical>());
    }
    EXPECT_EQ(last, total);
  };
}

int
check_4dmesh(mesh4d::accessor<ro> m) {
  UNIT("TASK") {
    using r = mesh4d::domain;
    using ax = mesh4d::axis;

    std::set<util::id> logical[2] = {{1, 2}, {1, 2}};
    std::set<util::id> extended[2] = {{0, 1, 2}, {1, 2, 3}};
    std::set<util::id> all[2] = {{0, 1, 2, 3}, {0, 1, 2, 3}};
    std::set<util::id> ghost_low[2] = {{}, {0}};
    std::set<util::id> ghost_high[2] = {{3}, {}};
    std::set<util::id> boundary_low[2] = {{0}, {}};
    std::set<util::id> boundary_high[2] = {{}, {3}};

    const int nparts = 2; //#num colors on each axis

    const auto idx = [&] {
      auto rem = color();
      std::vector<int> idx(4);
      for(auto dim = 0; dim < 4; ++dim) {
        idx[dim] = rem % nparts;
        rem = (rem - idx[dim]) / nparts;
      }
      return idx;
    }();

    const auto s = [](auto && r) {
      return std::set<util::id>(r.begin(), r.end());
    };

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

int
narray_driver() {
  UNIT() {
    {
      using topo::narray_utils::factor;
      using V = std::vector<std::size_t>;
      EXPECT_EQ(factor(2 * 5 * 11 * 13 * 29), (V{29, 13, 11, 5, 2}));
      EXPECT_EQ(factor(2 * 2 * 23 * 23), (V{23, 23, 2, 2}));
    } // scope

    {
      // 1D Mesh
      mesh1d::gcoord indices{9};
      mesh1d::index_definition idef;
      idef.axes = topo::narray_utils::make_axes(processes(), indices);
      idef.axes[0].hdepth = 1;
      idef.axes[0].bdepth = 2;
      idef.diagonals = true;
      idef.full_ghosts = true;

      // coloring1.allocate(index_definitions);
      coloring1.allocate(idef);
      m1.allocate(coloring1.get());
      execute<init_field<1>, default_accelerator>(m1, f1(m1));
      execute<print_field<1>>(m1, f1(m1));
      execute<update_field<1>, default_accelerator>(m1, f1(m1));
      execute<print_field<1>>(m1, f1(m1));
      EXPECT_EQ(test<check_mesh_field<1>>(m1, f1(m1)), 0);

      if(FLECSI_BACKEND != FLECSI_BACKEND_mpi) {
        auto lm = data::launch::make<data::launch::gather>(m1, 1);
        EXPECT_EQ(test<check_contiguous>(lm), 0);
      }
    } // scope

    {
      // 2D Mesh
      mesh2d::gcoord indices{8, 8};
      mesh2d::index_definition idef;
      idef.axes = topo::narray_utils::make_axes(processes(), indices);
      idef.axes[0].hdepth = 1;
      idef.axes[1].hdepth = 2;
      idef.axes[0].bdepth = 2;
      idef.axes[1].bdepth = 1;
      idef.diagonals = true;
      idef.full_ghosts = true;

      coloring2.allocate(idef);
      m2.allocate(coloring2.get());
      execute<init_field<2>, default_accelerator>(m2, f2(m2));
      execute<print_field<2>>(m2, f2(m2));
      execute<update_field<2>, default_accelerator>(m2, f2(m2));
      execute<print_field<2>>(m2, f2(m2));
      EXPECT_EQ(test<check_mesh_field<2>>(m2, f2(m2)), 0);
    } // scope

    {
      // 3D Mesh
      mesh3d::gcoord indices{4, 4, 4};
      mesh3d::index_definition idef;
      idef.axes = topo::narray_utils::make_axes(processes(), indices);
      for(auto & a : idef.axes) {
        a.hdepth = 1;
        a.bdepth = 1;
      }
      idef.diagonals = true;
      idef.full_ghosts = true;

      coloring3.allocate(idef);
      m3.allocate(coloring3.get());
      execute<init_field<3>, default_accelerator>(m3, f3(m3));
      execute<print_field<3>>(m3, f3(m3));
      execute<update_field<3>, default_accelerator>(m3, f3(m3));
      execute<print_field<3>>(m3, f3(m3));
      EXPECT_EQ(test<check_mesh_field<3>>(m3, f3(m3)), 0);
    } // scope

    if(FLECSI_BACKEND != FLECSI_BACKEND_mpi) {
      // 4D Mesh
      mesh4d::gcoord indices{4, 4, 4, 4};
      mesh4d::index_definition idef;
      idef.axes = topo::narray_utils::make_axes(processes(), indices);
      for(auto & a : idef.axes) {
        a.hdepth = 1;
        a.bdepth = 1;
      }
      idef.diagonals = true;
      idef.full_ghosts = true;

      coloring4.allocate(idef);
      m4.allocate(coloring4.get());
      execute<check_4dmesh>(m4);
    }
  }; // UNIT
} // narray_driver

flecsi::unit::driver<narray_driver> nd;
