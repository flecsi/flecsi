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
  std::array<util::id, D> lbnds, ubnds;
  m.bounds(lbnds, ubnds);
  auto str_local = m.template strides();
  flecsi::topo::narray_impl::linearize<D, util::id> ln_local{str_local};

  using tb = flecsi::topo::narray_impl::traverse<D, util::id>;
  for(auto && v : tb(lbnds, ubnds)) {
    fvalue(ca[ln_local(v)]);
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

template<typename Mesh, typename Mesh::axis A, typename Mesh::domain DM>
struct range {
  using accessor = typename Mesh::template accessor<ro>;
  const accessor & m;

  friend std::ostream & operator<<(std::ostream & o, const range & r) {
    const auto off = r.m.template offset<A, DM>();
    o << " [" << off << ", " << off + r.m.template size<A, DM>() << ")";
    return o;
  }
};

template<std::size_t D>
struct Axes {
  using r = typename mesh<D>::domain;

  template<typename mesh<D>::axis A, r R>
  using rng = range<mesh<D>, A, R>;

  template<auto... A>
  static std::string data(typename mesh<D>::template accessor<ro> & m,
    util::constants<A...>) {
    std::stringstream ss;
    ((ss << "log:") << ... << rng<A, r::logical>{m}) << "\n";
    ((ss << "ext:") << ... << rng<A, r::extended>{m}) << "\n";
    ((ss << "all:") << ... << rng<A, r::all>{m}) << "\n";
    ((ss << "bd_lo:") << ... << rng<A, r::boundary_low>{m}) << "\n";
    ((ss << "bd_hi:") << ... << rng<A, r::boundary_high>{m}) << "\n";
    ((ss << "gh_lo:") << ... << rng<A, r::ghost_low>{m}) << "\n";
    ((ss << "gh_hi:") << ... << rng<A, r::ghost_high>{m}) << "\n";
    ((ss << "global:") << ... << rng<A, r::global>{m}) << "\n";
    return ss.str();
  }

  static std::string data(typename mesh<D>::template accessor<ro> & m) {
    return data(m, typename mesh<D>::axes());
  }
};

template<auto A>
using cnst = util::constant<A>;

template<std::size_t D>
int
check_mesh_field(std::string name,
  typename mesh<D>::template accessor<ro> m,
  field<std::size_t>::accessor<ro, ro> ca) {
  using flog::container;

  std::string output_file = "mesh_" + name + "_" + std::to_string(colors()) +
                            "_" + std::to_string(color()) + ".blessed";
  UNIT("TASK") {
    std::stringstream out;
    if constexpr(D == 1) {
      using r = mesh1d::domain;
      using ax = mesh1d::axis;

      out << Axes<D>::data(m);

      // check field values on the ghost layers
      auto c = m.template mdspan<topo::elements>(ca);
      std::vector<util::id> g_lo, g_hi;

      for(auto i : m.template range<ax::x_axis, r::ghost_low>()) {
        g_lo.push_back(c[i]);
      }

      for(auto i : m.template range<ax::x_axis, r::ghost_high>()) {
        g_hi.push_back(c[i]);
      }

      out << "x_gh_lo_val: " << container{g_lo} << "\n";
      out << "x_gh_hi_val: " << container{g_hi} << "\n";
    } // d=1
    else if constexpr(D == 2) {
      using r = mesh2d::domain;
      using ax = mesh2d::axis;
      out << Axes<D>::data(m);

      // check field values on the ghost layers
      auto c = m.template mdspan<topo::elements>(ca);

      auto elems = [&](auto y, auto x) {
        auto ybnd = m.template range<ax::y_axis, decltype(y)::value>();
        auto xbnd = m.template range<ax::x_axis, decltype(x)::value>();
        std::vector<std::string> result;
        for(auto j : ybnd) {
          std::vector<util::id> row;
          std::stringstream ss;
          for(auto i : xbnd) {
            row.push_back(c[j][i]);
          }
          ss << container{row};
          result.emplace_back(ss.str());
        }
        return result;
      };

      // Check field values on the ghost layers
      // Layout used to identify neighbor ranks
      // for computing the correct field values
      // on the ghost layers.
      // clang-format off
      out << "y_log_x_gh_lo: " << container{elems(cnst<r::logical>(), cnst<r::ghost_low>())} << "\n";
      out << "y_log_x_gh_hi: " << container{elems(cnst<r::logical>(), cnst<r::ghost_high>())} <<  "\n";
      out << "y_gh_lo_x_log: " << container{elems(cnst<r::ghost_low>(), cnst<r::logical>())} << "\n";
      out << "y_gh_hi_x_log: " << container{elems(cnst<r::ghost_high>(), cnst<r::logical>())} << "\n";
      out << "y_gh_lo_x_gh_lo: " << container{elems(cnst<r::ghost_low>(), cnst<r::ghost_low>())} << "\n";
      out << "y_gh_lo_x_gh_hi: " << container{elems(cnst<r::ghost_low>(), cnst<r::ghost_high>())} << "\n";
      out << "y_gh_hi_x_gh_lo: " << container{elems(cnst<r::ghost_high>(), cnst<r::ghost_low>())} << "\n";
      out << "y_gh_hi_x_gh_hi: " << container{elems(cnst<r::ghost_high>(), cnst<r::ghost_high>())} << "\n";
      // clang-format on
    } // d=2
    else {
      using r = mesh3d::domain;
      using ax = mesh3d::axis;

      out << Axes<D>::data(m);

      auto c = m.template mdspan<topo::elements>(ca);

      auto elems = [&](auto z, auto y, auto x) {
        auto zbnd = m.template range<ax::z_axis, decltype(z)::value>();
        auto ybnd = m.template range<ax::y_axis, decltype(y)::value>();
        auto xbnd = m.template range<ax::x_axis, decltype(x)::value>();
        std::vector<std::string> result;
        for(auto k : zbnd) {
          std::vector<std::string> col;
          std::stringstream ss_outer;
          for(auto j : ybnd) {
            std::vector<util::id> row;
            std::stringstream ss;
            for(auto i : xbnd) {
              row.push_back(c[k][j][i]);
            }
            ss << container{row};
            col.emplace_back(ss.str());
          }
          ss_outer << container{col};
          result.emplace_back(ss_outer.str());
        }
        return result;
      };

      // Check field values on the ghost layers
      // For this particular mesh, the partition
      // is 2x2 over the x-y plane, so, only the edge-connected
      // cells (diagonals) are the corners, so same neighbor rank
      // datastructure as 2d case can be used here.
      // clang-format off
      out << "z_log_y_log_x_gh_lo: "
          << container{elems(cnst<r::logical>(), cnst<r::logical>(), cnst<r::ghost_low>())} << "\n";
      out << "z_log_y_log_x_gh_hi: "
          << container{elems(cnst<r::logical>(), cnst<r::logical>(), cnst<r::ghost_low>())} << "\n";
      out << "z_log_y_gh_lo_x_log: "
          << container{elems(cnst<r::logical>(), cnst<r::ghost_low>(), cnst<r::logical>())} << "\n";
      out << "z_log_y_gh_hi_x_log: "
          << container{elems(cnst<r::logical>(), cnst<r::ghost_high>(), cnst<r::logical>())} << "\n";
      out << "z_log_y_gh_lo_x_gh_lo: "
          << container{elems(cnst<r::logical>(), cnst<r::ghost_low>(), cnst<r::ghost_low>())} << "\n";
      out << "z_log_y_gh_lo_x_gh_hi: "
          << container{elems(cnst<r::logical>(), cnst<r::ghost_low>(), cnst<r::ghost_high>())} << "\n";
      out << "z_log_y_gh_hi_x_gh_lo: "
          << container{elems(cnst<r::logical>(), cnst<r::ghost_high>(), cnst<r::ghost_low>())} << "\n";
      out << "z_log_y_gh_hi_x_gh_hi: "
          << container{elems(cnst<r::logical>(), cnst<r::ghost_high>(), cnst<r::ghost_high>())} << "\n";
      // clang-format on
    } // d=3
    UNIT_CAPTURE() << out.rdbuf();
    EXPECT_TRUE(UNIT_EQUAL_BLESSED(output_file));
  };
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
  std::size_t sz) {
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

bool
check_sz(ints::accessor<ro, ro> tf, util::id lid, std::size_t sz) {
  return (tf[lid].size() == (std::size_t)sz);
}

bool
check_sz(ints::mutator<wo, na> tf, util::id lid, std::size_t sz) {
  tf[lid].resize(sz);
  return true;
}

template<std::size_t D, typename mesh<D>::domain DOM, bool V>
int
init_verify_rf(typename mesh<D>::template accessor<ro> m,
  std::conditional_t<V, ints::accessor<ro, ro>, ints::mutator<wo, na>> tf,
  std::size_t sz,
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
      std::size_t sz2 = (V && (!diagonals) && m.check_diag_bounds(v)) ? 0 : sz;
      ASSERT_TRUE(check_sz(tf, lid, sz2));
      for(std::size_t n = 0; n < sz2; ++n) {
        EXPECT_TRUE(check(tf[lid][n], (int)(gid * 10000 + n)));
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

template<std::size_t D>
int
test_mesh(const std::string name,
  topo::narray_impl::colors color_dist,
  const typename mesh<D>::gcoord & indices,
  const typename mesh<D>::coord & hdepth,
  const typename mesh<D>::coord & bdepth,
  std::array<bool, D> periodic,
  std::array<bool, D> auxiliary,
  bool diagonals,
  bool full_ghosts,
  std::size_t sz,
  bool full_verify,
  bool print_info,
  const field<std::size_t>::definition<mesh<D>> & f,
  field<int, data::ragged>::definition<mesh<D>> & rf) {
  UNIT() {
    typename mesh<D>::index_definition idef;
    idef.axes = topo::narray_utils::make_axes(
      color_dist.empty() ? topo::narray_utils::distribute(processes(), indices)
                         : std::move(color_dist),
      indices);
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
    typename mesh<D>::slot m;
    {
      typename mesh<D>::cslot coloring;
      coloring.allocate(idef);
      m.allocate(coloring.get());
    }

    execute<init_field<D>>(m, f(m));

    if(print_info)
      execute<print_field<D>>(m, f(m));

    if(full_verify) {
      execute<print_field<D>>(m, f(m));
      execute<update_field<D>>(m, f(m));
      execute<print_field<D>>(m, f(m));
      EXPECT_EQ(test<check_mesh_field<D>>(name, m, f(m)), 0);
    }

    // ragged field
    auto & tf = rf(m).get_elements();

    // Set growth policy lo=0.1 to maintain correct size for all D
    tf.growth = {0, 0, 0.1, 0.5, 1};
    execute<allocate_field<D>>(f(m), tf.sizes(), sz);

    EXPECT_EQ((test<init_verify_rf<D, mesh2d::domain::logical, false>>(
                m, rf(m), sz, idef.diagonals)),
      0);

    if(print_info)
      execute<print_rf<D>>(m, rf(m));

    EXPECT_EQ((test<init_verify_rf<D, mesh<D>::domain::all, true>>(
                m, rf(m), sz, idef.diagonals)),
      0);

    if(print_info)
      execute<print_rf<D>>(m, rf(m));
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
      execute<init_field<1>>(m1, f1(m1));
      execute<print_field<1>>(m1, f1(m1));
      execute<update_field<1>>(m1, f1(m1));
      execute<print_field<1>>(m1, f1(m1));
      EXPECT_EQ(test<check_mesh_field<1>>("1d", m1, f1(m1)), 0);

      if(FLECSI_BACKEND != FLECSI_BACKEND_mpi) {
        auto lm = data::launch::make(m1, data::launch::gather(m1.colors(), 1));
        EXPECT_EQ(test<check_contiguous>(lm), 0);
      }

      // ragged field
      std::size_t sz = 100;

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
      // Primary cells: diagonals variation, full ghosts on, though here full
      // ghosts don't have any meaning.
      test_mesh<2>("2d",
        {},
        {8, 8},
        {1, 2},
        {1, 2},
        {true, true},
        {false, false},
        true,
        true,
        100,
        true,
        false,
        f2,
        rf2);

      test_mesh<2>("2d",
        {},
        {8, 8},
        {1, 1},
        {1, 1},
        {true, true},
        {false, false},
        false,
        true,
        3,
        false,
        true,
        f2,
        rf2);

      // Aux verts: diagonals on, full ghosts true
      test_mesh<2>("2d",
        {},
        {8, 8},
        {2, 2},
        {0, 0},
        {false, false},
        {true, true},
        true,
        true,
        3,
        false,
        false,
        f2,
        rf2);

      test_mesh<2>("2d",
        {},
        {8, 8},
        {2, 2},
        {0, 0},
        {false, false},
        {true, true},
        true,
        false,
        3,
        false,
        false,
        f2,
        rf2);

      // Aux x-edges: diagonals on, full ghosts true
      test_mesh<2>("2d",
        {},
        {8, 8},
        {2, 2},
        {0, 0},
        {false, false},
        {true, false},
        true,
        true,
        3,
        false,
        false,
        f2,
        rf2);

      test_mesh<2>("2d",
        {},
        {8, 8},
        {2, 2},
        {0, 0},
        {false, false},
        {true, false},
        true,
        false,
        3,
        false,
        false,
        f2,
        rf2);

      // Aux y-edges: diagonals on, full ghosts true
      test_mesh<2>("2d",
        {},
        {8, 8},
        {2, 2},
        {0, 0},
        {false, false},
        {false, true},
        true,
        true,
        3,
        false,
        false,
        f2,
        rf2);

      test_mesh<2>("2d",
        {},
        {8, 8},
        {2, 2},
        {0, 0},
        {false, false},
        {false, true},
        true,
        false,
        3,
        false,
        false,
        f2,
        rf2);
    } // scope

    {
      // 3D
      test_mesh<3>("3d",
        {},
        {4, 4, 4},
        {
          1,
          1,
          1,
        },
        {1, 1, 1},
        {true, true, true},
        {false, false, false},
        true,
        true,
        100,
        true,
        false,
        f3,
        rf3);
      test_mesh<3>("3d",
        {2, 2, 1},
        {4, 4, 2},
        {
          1,
          1,
          1,
        },
        {0, 0, 0},
        {false, false, false},
        {false, false, false},
        false,
        true,
        3,
        false,
        false,
        f3,
        rf3);
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

template<typename coloring>
int
verify_colors(std::string name, coloring & pcs) {
  UNIT() {
    std::string output_file = "coloring_" + name + "_" +
                              std::to_string(processes()) + "_" +
                              std::to_string(process()) + ".blessed";
    std::stringstream out;
    std::vector<std::size_t> color, extents;
    std::vector<std::string> global, extent, offset, logical, extended;

    auto seq = [](const auto & c) {
      std::stringstream ss;
      ss << flog::container{c};
      return ss.str();
    };

    for(const auto & pc : pcs) {
      std::vector<std::size_t> g, e, o;
      std::vector<std::string> log, ext;
      color.push_back(pc.color());
      extents.push_back(pc.extents());
      for(const topo::narray_impl::axis_color & axco : pc.axis_colors) {
        g.push_back(axco.global());
        e.push_back(axco.extent());
        o.push_back(axco.offset());
        log.push_back(seq(std::vector{axco.logical<0>(), axco.logical<1>()}));
        ext.push_back(seq(std::vector{axco.extended<0>(), axco.extended<1>()}));
      }
      global.emplace_back(seq(g));
      extent.emplace_back(seq(e));
      offset.emplace_back(seq(o));
      logical.emplace_back(seq(log));
      extended.emplace_back(seq(ext));
    }

    out << "color: " << flog::container{color} << "\n";
    out << "extents: " << flog::container{extents} << "\n";
    out << "global: " << flog::container{global} << "\n";
    out << "extent: " << flog::container{extent} << "\n";
    out << "offset: " << flog::container{offset} << "\n";
    out << "logical: " << flog::container{logical} << "\n";
    out << "extended: " << flog::container{extended} << "\n";

    UNIT_CAPTURE() << out.rdbuf();
    EXPECT_TRUE(UNIT_EQUAL_BLESSED(output_file));
  };
}

int
coloring_driver() {
  UNIT() {
    mesh3d::gcoord indices{9, 9, 9};

    mesh3d::index_definition idef;
    idef.axes = topo::narray_utils::make_axes(9, indices);
    for(auto & a : idef.axes) {
      a.hdepth = 1;
    }

    auto coloring = idef.process_coloring();
    EXPECT_EQ(verify_colors("primary_9x9x9_3x3x1", coloring), 0);

    // Create definition for an auxiliary index space
    auto adef = idef;
    adef.axes[0].auxiliary = true;
    adef.axes[1].auxiliary = true;
    adef.full_ghosts = false;
    auto avpc = adef.process_coloring();

    EXPECT_EQ(verify_colors("aux_xy_9x9x9_3x3x1", avpc), 0);

    // Change the halo depths for the primary index space
    for(auto & a : idef.axes) {
      a.hdepth = 2;
    }

    coloring = idef.process_coloring();
    adef = idef;
    adef.axes[0].auxiliary = true;
    adef.full_ghosts = true;
    avpc = adef.process_coloring();

    EXPECT_EQ(verify_colors("aux_x_9x9x9_3x3x1_fg", avpc), 0);
  };
}

util::unit::driver<coloring_driver> cd;
