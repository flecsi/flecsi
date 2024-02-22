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

template<typename U>
struct init_field_functor {
  U c;
  template<typename T>
  void FLECSI_TARGET operator()(T & x) const {
    x = c;
  }
};

template<std::size_t D>
void
init_field(typename mesh<D>::template accessor<ro> m,
  field<std::size_t>::accessor<wo, na> ca) {
  auto c = color();
  return field_helper<D>(m, ca, init_field_functor<decltype(c)>{c});
} // init_field

template<typename U>
struct update_field_functor {
  U c;
  template<typename T>
  void FLECSI_TARGET operator()(T & x) const {
    x = pow(10, c);
  }
};

template<std::size_t D>
void
update_field(typename mesh<D>::template accessor<ro> m,
  field<std::size_t>::accessor<wo, na> ca) {
  auto c = color();
  return field_helper<D>(m, ca, update_field_functor<decltype(c)>{c});
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
  }
  flog(warn) << ss.rdbuf() << std::endl;

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
      std::set<util::id> xlogical = {1, 2, 3, 4};
      std::set<util::id> xextended[4] = {
        {0, 1, 2, 3, 4}, {1, 2, 3, 4, 5}, {0, 1, 2, 3, 4}, {1, 2, 3, 4, 5}};
      std::set<util::id> xall = {0, 1, 2, 3, 4, 5};
      std::set<util::id> xboundary_low[4] = {{0}, {}, {0}, {}};
      std::set<util::id> xboundary_high[4] = {{}, {5}, {}, {5}};
      std::set<util::id> xghost_low[4] = {{}, {0}, {}, {0}};
      std::set<util::id> xghost_high[4] = {{5}, {}, {5}, {}};

      std::set<util::id> ylogical = {2, 3, 4, 5};
      std::set<util::id> yextended[4] = {{0, 1, 2, 3, 4, 5},
        {0, 1, 2, 3, 4, 5},
        {2, 3, 4, 5, 6, 7},
        {2, 3, 4, 5, 6, 7}};
      std::set<util::id> yall = {0, 1, 2, 3, 4, 5, 6, 7};
      std::set<util::id> yboundary_low[4] = {{0, 1}, {0, 1}, {}, {}};
      std::set<util::id> yboundary_high[4] = {{}, {}, {6, 7}, {6, 7}};
      std::set<util::id> yghost_low[4] = {{}, {}, {0, 1}, {0, 1}};
      std::set<util::id> yghost_high[4] = {{6, 7}, {6, 7}, {}, {}};

      const int rank = process();
      const auto s = [](auto && r) {
        return std::set<util::id>(r.begin(), r.end());
      };

      EXPECT_EQ(s(m.template range<ax::x_axis>()), xlogical);
      EXPECT_EQ(
        s(m.template range<ax::x_axis, r::extended>()), xextended[rank]);
      EXPECT_EQ(s(m.template range<ax::x_axis, r::all>()), xall);
      EXPECT_EQ(s(m.template range<ax::x_axis, r::boundary_low>()),
        xboundary_low[rank]);
      EXPECT_EQ(s(m.template range<ax::x_axis, r::boundary_high>()),
        xboundary_high[rank]);
      EXPECT_EQ(
        s(m.template range<ax::x_axis, r::ghost_low>()), xghost_low[rank]);
      EXPECT_EQ(
        s(m.template range<ax::x_axis, r::ghost_high>()), xghost_high[rank]);

      EXPECT_EQ(s(m.template range<ax::y_axis>()), ylogical);
      EXPECT_EQ(
        s(m.template range<ax::y_axis, r::extended>()), yextended[rank]);
      EXPECT_EQ(s(m.template range<ax::y_axis, r::all>()), yall);
      EXPECT_EQ(s(m.template range<ax::y_axis, r::boundary_low>()),
        yboundary_low[rank]);
      EXPECT_EQ(s(m.template range<ax::y_axis, r::boundary_high>()),
        yboundary_high[rank]);
      EXPECT_EQ(
        s(m.template range<ax::y_axis, r::ghost_low>()), yghost_low[rank]);
      EXPECT_EQ(
        s(m.template range<ax::y_axis, r::ghost_high>()), yghost_high[rank]);

      // check sizes
      std::size_t xsizes_ex[4][8] = {{4, 5, 6, 1, 0, 0, 1, 8},
        {4, 5, 6, 0, 1, 1, 0, 8},
        {4, 5, 6, 1, 0, 0, 1, 8},
        {4, 5, 6, 0, 1, 1, 0, 8}};

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

      std::size_t ysizes_ex[4][8] = {{4, 6, 8, 2, 0, 0, 2, 8},
        {4, 6, 8, 2, 0, 0, 2, 8},
        {4, 6, 8, 0, 2, 2, 0, 8},
        {4, 6, 8, 0, 2, 2, 0, 8}};

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
      std::size_t xoffsets_ex[4][8] = {{1, 0, 0, 0, 5, 0, 5, 0},
        {1, 1, 0, 1, 5, 0, 5, 4},
        {1, 0, 0, 0, 5, 0, 5, 0},
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

      std::size_t yoffsets_ex[4][8] = {{2, 0, 0, 0, 6, 0, 6, 0},
        {2, 0, 0, 0, 6, 0, 6, 0},
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

      auto chk = [&c](const std::set<util::id> & ybnd,
                   const std::set<util::id> & xbnd,
                   int r) {
        bool iseq = true;
        for(auto j : ybnd) {
          for(auto i : xbnd) {
            iseq = iseq && (c[j][i] == size_t(std::pow(10, r)));
          }
        }
        return iseq;
      };

      EXPECT_EQ(chk(ylogical, xghost_low[rank], ngb_ranks[rank][3]), true);
      EXPECT_EQ(chk(ylogical, xghost_high[rank], ngb_ranks[rank][5]), true);
      EXPECT_EQ(chk(yghost_low[rank], xlogical, ngb_ranks[rank][1]), true);
      EXPECT_EQ(chk(yghost_high[rank], xlogical, ngb_ranks[rank][7]), true);

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

using ints = field<int, data::ragged>;

// This print task will only read the ghost values and not invoke any
// communication. The purpose is to aid in debugging and see the sequence
// of changes to field values as other tasks are performed.
template<std::size_t D>
void
print_rf(typename mesh<D>::template accessor<ro> m, ints::accessor<ro, na> tf) {
  std::stringstream ss;
  ss << " Color " << color() << std::endl;

  using tb = flecsi::topo::narray_impl::traverse<D, util::id>;
  std::array<util::id, D> lbnds{0};
  auto str_local = m.template strides();
  auto str_global = m.template strides<util::gid, mesh<D>::domain::global>();
  flecsi::topo::narray_impl::linearize<D, util::id> ln_local{str_local};
  flecsi::topo::narray_impl::linearize<D, util::gid> ln_global{str_global};

  for(auto && v : tb(lbnds, str_local)) {
    auto gids = m.global_ids(v);
    auto lid = ln_local(v);
    auto gid = ln_global(gids);

    ss << "For cell (" << lid << ", " << gid
       << "), field_size = " << tf[lid].size() << ", field vals = [";
    for(std::size_t k = 0; k < tf[lid].size(); ++k)
      ss << tf[lid][k] << "  ";
    ss << "]\n\n";
  }

  flog(info) << ss.rdbuf() << std::endl;
} // print_rf

template<std::size_t D>
void
allocate_field(field<std::size_t>::accessor<ro, ro> f,
  topo::resize::Field::accessor<wo> a,
  int sz) {
  a = f.span().size() * sz;
}

bool
check(const int & x, int y) {
  return (x == y);
}

bool
check(int & x, int y) {
  x = y;
  return true;
}

template<bool V>
bool
check_sz(
  std::conditional_t<V, ints::accessor<ro, ro>, ints::mutator<wo, na>> tf,
  util::id & lid,
  int sz) {
  if constexpr(V) {
    return (tf[lid].size() == (std::size_t)sz);
  }
  else {
    tf[lid].resize(sz);
  }
  return true;
}

template<std::size_t D, typename mesh<D>::domain DOM, bool V>
int
init_verify_rf(typename mesh<D>::template accessor<ro> m,
  std::conditional_t<V, ints::accessor<ro, ro>, ints::mutator<wo, na>> tf,
  int sz,
  bool diagonals) {
  UNIT("INIT_VERIFY_RAGGED_FIELD") {
    std::array<util::id, D> lbnds, ubnds;
    m.template bounds<DOM>(lbnds, ubnds);
    auto str_local = m.template strides();
    auto str_global = m.template strides<util::gid, mesh<D>::domain::global>();
    flecsi::topo::narray_impl::linearize<D, util::id> ln_local{str_local};
    flecsi::topo::narray_impl::linearize<D, util::gid> ln_global{str_global};

    using tb = flecsi::topo::narray_impl::traverse<D, util::id>;
    for(auto && v : tb(lbnds, ubnds)) {
      auto gids = m.global_ids(v);
      auto lid = ln_local(v);
      auto gid = ln_global(gids);
      int sz2 = (V && (!diagonals) && m.check_diag_bounds(v)) ? 0 : sz;
      EXPECT_EQ(check_sz<V>(tf, lid, sz2), true);
      for(int n = 0; n < sz2; ++n) {
        EXPECT_EQ(check(tf[lid][n], (int)(gid * 10000 + n)), true);
      }
    }
  };
}

const field<std::size_t>::definition<mesh1d> f1;
field<int, data::ragged>::definition<mesh1d> rf1;
const field<std::size_t>::definition<mesh2d> f2;
field<int, data::ragged>::definition<mesh2d> rf2;
const field<std::size_t>::definition<mesh3d> f3;
field<int, data::ragged>::definition<mesh3d> rf3;

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

template<std::size_t D, typename mesh<D>::domain DOM>
int
value_rewrite_rf(typename mesh<D>::template accessor<ro> m,
  ints::accessor<wo, na> a) {
  UNIT("REWRITE_RF") {
    std::array<util::id, D> lbnds, ubnds;
    m.template bounds<DOM>(lbnds, ubnds);
    auto str_local = m.template strides();
    flecsi::topo::narray_impl::linearize<D, util::id> ln_local{str_local};

    using tb = flecsi::topo::narray_impl::traverse<D, util::id>;
    for(auto && v : tb(lbnds, ubnds)) {
      for(auto & x : a[ln_local(v)])
        x = -x;
    }
  };
}

template<std::size_t D, typename mesh<D>::domain DOM>
int
value_rewrite_verify_rf(typename mesh<D>::template accessor<ro> m,
  ints::accessor<ro, ro> tf) {
  UNIT("REWRITE_VERIFY_RF") {
    using tb = flecsi::topo::narray_impl::traverse<D, util::id>;
    auto str_local = m.template strides();
    auto str_global = m.template strides<util::gid, mesh<D>::domain::global>();
    flecsi::topo::narray_impl::linearize<D, util::id> ln_local{str_local};
    flecsi::topo::narray_impl::linearize<D, util::gid> ln_global{str_global};
    std::array<util::id, D> lbnds, ubnds;

    m.template bounds<DOM>(lbnds, ubnds);
    for(auto && v : tb(lbnds, ubnds)) {
      auto gids = m.global_ids(v);
      auto lid = ln_local(v);
      auto gid = ln_global(gids);
      for(std::size_t k = 0; k < tf[lid].size(); ++k)
        EXPECT_EQ(tf[lid][k], (int)(-(gid * 10000 + k)));
    }
  };
}

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
      mesh1d::slot m1;

      mesh1d::gcoord indices{9};
      mesh1d::index_definition idef;
      idef.axes = topo::narray_utils::make_axes(processes(), indices);
      idef.axes[0].hdepth = 1;
      idef.axes[0].bdepth = 2;
      idef.diagonals = true;
      idef.full_ghosts = true;

      {
        mesh1d::cslot coloring1;
        coloring1.allocate(idef);
        m1.allocate(coloring1.get());
      }
      execute<init_field<1>, default_accelerator>(m1, f1(m1));
      execute<print_field<1>>(m1, f1(m1));
      execute<update_field<1>, default_accelerator>(m1, f1(m1));
      execute<print_field<1>>(m1, f1(m1));
      EXPECT_EQ(test<check_mesh_field<1>>(m1, f1(m1)), 0);

      if(FLECSI_BACKEND != FLECSI_BACKEND_mpi) {
        auto lm = data::launch::make(m1, data::launch::gather(m1.colors(), 1));
        EXPECT_EQ(test<check_contiguous>(lm), 0);
      }

      // ragged field
      int sz = 100;

      auto & tf = rf1(m1).get_elements();
      tf.growth = {0, 0, 0.25, 0.5, 1};
      execute<allocate_field<1>>(f1(m1), tf.sizes(), sz);

      execute<init_verify_rf<1, mesh1d::domain::logical, false>>(
        m1, rf1(m1), sz, idef.diagonals);
      execute<print_rf<1>>(m1, rf1(m1));

      // tests the case where the periodic flag is false, but with non-zero
      // bdepth, communication is expected only for the halo layers.
      EXPECT_EQ((test<init_verify_rf<1, mesh1d::domain::ghost_low, true>>(
                  m1, rf1(m1), sz, idef.diagonals)),
        0);
      execute<print_rf<1>>(m1, rf1(m1));
      EXPECT_EQ((test<init_verify_rf<1, mesh1d::domain::ghost_high, true>>(
                  m1, rf1(m1), sz, idef.diagonals)),
        0);

      // tests if a rewrite of values on ragged with an accessor triggers a
      // ghost copy
      execute<value_rewrite_rf<1, mesh1d::domain::logical>>(m1, rf1(m1));
      EXPECT_EQ((test<value_rewrite_verify_rf<1, mesh1d::domain::ghost_low>>(
                  m1, rf1(m1))),
        0);
      EXPECT_EQ((test<value_rewrite_verify_rf<1, mesh1d::domain::ghost_high>>(
                  m1, rf1(m1))),
        0);

    } // scope

    {
      // 2D Mesh
      auto test_2d = [&](const mesh2d::gcoord & indices,
                       const mesh2d::coord & hdepth,
                       const mesh2d::coord & bdepth,
                       std::array<bool, 2> periodic,
                       std::array<bool, 2> auxiliary,
                       bool diagonals,
                       bool full_ghosts,
                       int sz,
                       bool full_verify,
                       bool print_info) {
        mesh2d::index_definition idef;
        idef.axes = topo::narray_utils::make_axes(processes(), indices);
        int i = 0;
        for(auto & a : idef.axes) {
          a.hdepth = hdepth[i];
          a.bdepth = bdepth[i];
          a.periodic = periodic[i];
          a.auxiliary = auxiliary[i];
          ++i;
        }
        idef.diagonals = diagonals;
        idef.full_ghosts = full_ghosts;

        // create and allocate mesh slot
        mesh2d::slot m2;
        {
          mesh2d::cslot coloring2;
          coloring2.allocate(idef);
          m2.allocate(coloring2.get());
        }

        execute<init_field<2>, default_accelerator>(m2, f2(m2));

        if(print_info)
          execute<print_field<2>>(m2, f2(m2));

        if(full_verify) {
          execute<print_field<2>>(m2, f2(m2));
          execute<update_field<2>, default_accelerator>(m2, f2(m2));
          execute<print_field<2>>(m2, f2(m2));
          EXPECT_EQ(test<check_mesh_field<2>>(m2, f2(m2)), 0);
        }

        // ragged field
        auto & tf = rf2(m2).get_elements();
        tf.growth = {0, 0, 0.25, 0.5, 1};
        execute<allocate_field<2>>(f2(m2), tf.sizes(), sz);

        execute<init_verify_rf<2, mesh2d::domain::logical, false>>(
          m2, rf2(m2), sz, idef.diagonals);

        if(print_info)
          execute<print_rf<2>>(m2, rf2(m2));

        EXPECT_EQ((test<init_verify_rf<2, mesh2d::domain::all, true>>(
                    m2, rf2(m2), sz, idef.diagonals)),
          0);

        if(print_info)
          execute<print_rf<2>>(m2, rf2(m2));
      };

      // Primary cells: diagonals variation, full ghosts on, though here full
      // ghosts don't have any meaning.
      test_2d({8, 8},
        {1, 2},
        {1, 2},
        {true, true},
        {false, false},
        true,
        true,
        100,
        true,
        false);
      test_2d({8, 8},
        {1, 1},
        {1, 1},
        {true, true},
        {false, false},
        false,
        true,
        3,
        false,
        true);

      // Aux verts: diagonals on, full ghosts true
      test_2d({8, 8},
        {2, 2},
        {0, 0},
        {false, false},
        {true, true},
        true,
        true,
        3,
        false,
        false);

      test_2d({8, 8},
        {2, 2},
        {0, 0},
        {false, false},
        {true, true},
        true,
        false,
        3,
        false,
        false);

      // Aux x-edges: diagonals on, full ghosts true
      test_2d({8, 8},
        {2, 2},
        {0, 0},
        {false, false},
        {true, false},
        true,
        true,
        3,
        false,
        false);

      test_2d({8, 8},
        {2, 2},
        {0, 0},
        {false, false},
        {true, false},
        true,
        false,
        3,
        false,
        false);

      // Aux y-edges: diagonals on, full ghosts true
      test_2d({8, 8},
        {2, 2},
        {0, 0},
        {false, false},
        {false, true},
        true,
        true,
        3,
        false,
        false);

      test_2d({8, 8},
        {2, 2},
        {0, 0},
        {false, false},
        {false, true},
        true,
        false,
        3,
        false,
        false);

    } // scope

    {
      // 3D
      auto test_3d = [&](topo::narray_impl::colors color_dist,
                       const mesh3d::gcoord & indices,
                       const mesh3d::coord & hdepth,
                       const mesh3d::coord & bdepth,
                       std::array<bool, 3> periodic,
                       bool diagonals,
                       bool full_ghosts,
                       int sz,
                       bool full_verify) {
        mesh3d::index_definition idef;
        idef.axes = topo::narray_utils::make_axes(
          color_dist.empty()
            ? topo::narray_utils::distribute(processes(), indices)
            : std::move(color_dist),
          indices);
        int i = 0;
        for(auto & a : idef.axes) {
          a.hdepth = hdepth[i];
          a.bdepth = bdepth[i];
          a.periodic = periodic[i];
          ++i;
        }
        idef.diagonals = diagonals;
        idef.full_ghosts = full_ghosts;

        // create and allocate mesh slot
        mesh3d::slot m3;
        {
          mesh3d::cslot coloring3;
          coloring3.allocate(idef);
          m3.allocate(coloring3.get());
        }

        execute<init_field<3>, default_accelerator>(m3, f3(m3));

        if(full_verify) {
          execute<print_field<3>>(m3, f3(m3));
          execute<update_field<3>, default_accelerator>(m3, f3(m3));
          execute<print_field<3>>(m3, f3(m3));
          EXPECT_EQ(test<check_mesh_field<3>>(m3, f3(m3)), 0);
        }

        // ragged field
        auto & tf = rf3(m3).get_elements();

        // The usual growth policy (with lo = 0.25) will not work for this
        // particular problem setup since the number of cells initialized
        // (2*2*4 = 16) is much less than the quarter of the capacity (which
        // is 192), reducing the lo value from quarter to a tenth of the
        // capacity ensures that the correct size is maintained.
        tf.growth = {0, 0, 0.1, 0.5, 1};
        execute<allocate_field<3>>(f3(m3), tf.sizes(), sz);

        execute<init_verify_rf<3, mesh3d::domain::logical, false>>(
          m3, rf3(m3), sz, idef.diagonals);

        if(!full_verify)
          execute<print_rf<3>>(m3, rf3(m3));

        EXPECT_EQ((test<init_verify_rf<3, mesh3d::domain::all, true>>(
                    m3, rf3(m3), sz, idef.diagonals)),
          0);

        if(!full_verify)
          execute<print_rf<3>>(m3, rf3(m3));
      };

      test_3d({},
        {4, 4, 4},
        {
          1,
          1,
          1,
        },
        {1, 1, 1},
        {true, true, true},
        true,
        true,
        100,
        true);
      test_3d({2, 2, 1},
        {4, 4, 2},
        {
          1,
          1,
          1,
        },
        {0, 0, 0},
        {false, false, false},
        false,
        true,
        3,
        false);
    } // scope

    if(FLECSI_BACKEND != FLECSI_BACKEND_mpi) {
      // 4D Mesh
      mesh4d::slot m4;

      mesh4d::gcoord indices{4, 4, 4, 4};
      mesh4d::index_definition idef;
      idef.axes = topo::narray_utils::make_axes(16, indices);
      for(auto & a : idef.axes) {
        a.hdepth = 1;
        a.bdepth = 1;
      }
      idef.diagonals = true;
      idef.full_ghosts = true;

      {
        mesh4d::cslot coloring4;
        coloring4.allocate(idef);
        m4.allocate(coloring4.get());
      }
      EXPECT_EQ(test<check_4dmesh>(m4), 0);
    }

  }; // UNIT
} // narray_driver

util::unit::driver<narray_driver> nd;

/// Coloring testing
int
coloring_driver() {
  UNIT() {
    // 9x9x9 with 3x3x1 colors, extend x-axis and y-axis, full_ghosts: no
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

    auto coloring = idef.process_coloring();

    // Create definition for an auxiliary index space
    auto adef = idef;
    adef.axes[0].auxiliary = true;
    adef.axes[1].auxiliary = true;
    adef.full_ghosts = false;
    auto avpc = adef.process_coloring();

    mesh3d::gcoord global;
    std::map<Color, mesh3d::gcoord> extents, offset;
    std::map<Color, mesh3d::coord> logical_low, logical_high;
    std::map<Color, mesh3d::coord> extended_low, extended_high;

    global = {10, 10, 9};

    for(Color c = 0; c < 9; ++c) {
      extents[c] = {4, 4, 9};
      extended_high[c] = logical_high[c] = {4, 4, 9};
    }

    offset[0] = {0, 0, 0};
    offset[1] = {4, 0, 0};
    offset[2] = {7, 0, 0};
    offset[3] = {0, 4, 0};
    offset[4] = {4, 4, 0};
    offset[5] = {7, 4, 0};
    offset[6] = {0, 7, 0};
    offset[7] = {4, 7, 0};
    offset[8] = {7, 7, 0};

    extended_low[0] = logical_low[0] = {0, 0, 0};
    extended_low[1] = logical_low[1] = {1, 0, 0};
    extended_low[2] = logical_low[2] = {1, 0, 0};
    extended_low[3] = logical_low[3] = {0, 1, 0};
    extended_low[4] = logical_low[4] = {1, 1, 0};
    extended_low[5] = logical_low[5] = {1, 1, 0};
    extended_low[6] = logical_low[6] = {0, 1, 0};
    extended_low[7] = logical_low[7] = {1, 1, 0};
    extended_low[8] = logical_low[8] = {1, 1, 0};

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
      ss << "primary, full_ghosts : " << idef.full_ghosts << std::endl;
      for(auto p : coloring) {
        ss << p << std::endl;
      } // for
      ss << "auxiliary, full_ghosts : " << adef.full_ghosts << std::endl;
      for(auto p : avpc) {
        ss << p << std::endl;
      } // for
      flog(warn) << ss.rdbuf() << std::endl;
    };

    print_colorings();

    // 9x9x9 with 3x3x1 colors, extend x-axis, full_ghosts: yes
    //
    // 0   1   2   3  (4) (5)
    // +---+---+---+ ~ + ~ +
    // | 0 | 1 | 2 |(3)|(4)|
    // +---+---+---+ ~ + ~ +

    //
    // (1) (2) (3)  4   5   6  (7) (8)
    //  + ~ + ~ +---+---+---+ ~ + ~ +
    //  |(1)|(2)| 3 | 4 | 5 |(6)|(7)|
    //  + ~ + ~ +---+---+---+ ~ + ~ +

    //
    // (4) (5) (6)  7   8   9
    //  + ~ + ~ +---+---+---+
    //  |(4)|(5)| 6 | 7 | 8 |
    //  + ~ + ~ +---+---+---+

    // Change the halo depths for the primary index space
    for(auto & a : idef.axes) {
      a.hdepth = 2;
    }

    coloring = idef.process_coloring();
    adef = idef;
    adef.axes[0].auxiliary = true;
    adef.full_ghosts = true;
    avpc = adef.process_coloring();

    global = {10, 9, 9};
    extents[0] = {6, 5, 9};
    extents[1] = {8, 5, 9};
    extents[2] = {6, 5, 9};
    extents[3] = {6, 7, 9};
    extents[4] = {8, 7, 9};
    extents[5] = {6, 7, 9};
    extents[6] = {6, 5, 9};
    extents[7] = {8, 5, 9};
    extents[8] = {6, 5, 9};

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
    extended_low[1] = logical_low[1] = {3, 0, 0};
    extended_low[2] = logical_low[2] = {3, 0, 0};
    extended_low[3] = logical_low[3] = {0, 2, 0};
    extended_low[4] = logical_low[4] = {3, 2, 0};
    extended_low[5] = logical_low[5] = {3, 2, 0};
    extended_low[6] = logical_low[6] = {0, 2, 0};
    extended_low[7] = logical_low[7] = {3, 2, 0};
    extended_low[8] = logical_low[8] = {3, 2, 0};

    extended_high[0] = logical_high[0] = {4, 3, 9};
    extended_high[1] = logical_high[1] = {6, 3, 9};
    extended_high[2] = logical_high[2] = {6, 3, 9};
    extended_high[3] = logical_high[3] = {4, 5, 9};
    extended_high[4] = logical_high[4] = {6, 5, 9};
    extended_high[5] = logical_high[5] = {6, 5, 9};
    extended_high[6] = logical_high[6] = {4, 5, 9};
    extended_high[7] = logical_high[7] = {6, 5, 9};
    extended_high[8] = logical_high[8] = {6, 5, 9};

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

util::unit::driver<coloring_driver> cd;
