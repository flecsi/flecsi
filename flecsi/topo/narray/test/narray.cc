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

using axis_info = topo::narray_base::axis_info;

template<std::size_t D, typename F>
void
field_helper(typename mesh<D>::template accessor<ro> m,
  field<std::size_t>::accessor<wo, na> ca,
  F && fvalue) {
  const auto ln_local = m.linear();
  for(auto && v : m.range(false)) {
    fvalue(ca[ln_local(v)]);
  }
} // field_helper

// Initialize even the ghosts: with !diagonals, some are never copied.
void
init_field(field<std::size_t>::accessor<wo, wo> ca) {
  const auto s = ca.span();
  std::fill(s.begin(), s.end(), color());
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
    for(auto i : m.template axis<mesh1d::axis::x_axis>().layout.all()) {
      ss << c[i] << "   ";
    } // for
    ss << std::endl;
  }
  else if constexpr(D == 2) {
    for(int j = m.template axis<mesh2d::axis::y_axis>().layout.extent(); j--;) {
      for(auto i : m.template axis<mesh2d::axis::x_axis>().layout.all()) {
        ss << c[j][i] << "   ";
      } // for
      ss << std::endl;
    } // for
  }
  else {
    for(int k = m.template axis<mesh3d::axis::z_axis>().layout.extent(); k--;) {
      for(int j = m.template axis<mesh3d::axis::y_axis>().layout.extent();
          j--;) {
        for(auto i : m.template axis<mesh3d::axis::x_axis>().layout.all()) {
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
struct Axes {
  template<auto... A>
  static std::string data(typename mesh<D>::template accessor<ro> & m,
    util::constants<A...>) {
    std::stringstream ss;
    (
      [&] {
        const axis_info ai = m.template axis<A>();
        ss << ai.color << '/' << ai.axis.colors << '@' << ai.offset << '+'
           << ai.logical << '/' << ai.axis.extent << '|' << ai.axis.hdepth
           << '|' << ai.axis.bdepth << '=' << ai.layout.extent() << ' ';
        if(ai.axis.periodic)
          ss << 'P';
        if(ai.axis.auxiliary)
          ss << 'A';
        if(ai.axis.full_ghosts)
          ss << 'F';
        ss << '\n';
      }(),
      ...);
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
  const auto bounds = [](const axis_info & a) {
    return std::array{a.layout.ghost<0>(),
      a.layout.logical<0>(),
      a.layout.logical<1>(),
      a.layout.ghost<1>()};
  };

  std::string output_file = "mesh_" + name + "_" + std::to_string(colors()) +
                            "_" + std::to_string(color()) + ".blessed";
  UNIT("TASK") {
    auto & out = UNIT_CAPTURE();
    if constexpr(D == 1) {
      using ax = mesh1d::axis;

      out << Axes<D>::data(m);

      // check field values on the ghost layers
      auto c = m.template mdspan<topo::elements>(ca);
      const auto x = bounds(m.template axis<ax::x_axis>());
      std::vector<util::id> g_lo, g_hi;

      for(auto i : util::iota_view(x[0], x[1])) {
        g_lo.push_back(c[i]);
      }

      for(auto i : util::iota_view(x[2], x[3])) {
        g_hi.push_back(c[i]);
      }

      out << "x_gh_lo_val: " << container{g_lo} << "\n";
      out << "x_gh_hi_val: " << container{g_hi} << "\n";
    } // d=1
    else if constexpr(D == 2) {
      using ax = mesh2d::axis;
      out << Axes<D>::data(m);

      // check field values on the ghost layers
      auto c = m.template mdspan<topo::elements>(ca);
      const auto x = bounds(m.template axis<ax::x_axis>()),
                 y = bounds(m.template axis<ax::y_axis>());

      auto elems = [&](util::id y0, util::id y1, util::id x0, util::id x1) {
        std::vector<std::string> result;
        for(auto j : util::iota_view(y0, y1)) {
          std::vector<util::id> row;
          std::stringstream ss;
          for(auto i : util::iota_view(x0, x1)) {
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
      out << "y_log_x_gh_lo: " << container{elems(y[1], y[2], x[0], x[1])} << '\n';
      out << "y_log_x_gh_hi: " << container{elems(y[1], y[2], x[2], x[3])} << '\n';
      out << "y_gh_lo_x_log: " << container{elems(y[0], y[1], x[1], x[2])} << '\n';
      out << "y_gh_hi_x_log: " << container{elems(y[2], y[3], x[1], x[2])} << '\n';
      out << "y_gh_lo_x_gh_lo: " << container{elems(y[0], y[1], x[0], x[1])} << '\n';
      out << "y_gh_lo_x_gh_hi: " << container{elems(y[0], y[1], x[2], x[3])} << '\n';
      out << "y_gh_hi_x_gh_lo: " << container{elems(y[2], y[3], x[0], x[1])} << '\n';
      out << "y_gh_hi_x_gh_hi: " << container{elems(y[2], y[3], x[2], x[3])} << '\n';
      // clang-format on
    } // d=2
    else {
      using ax = mesh3d::axis;

      out << Axes<D>::data(m);

      auto c = m.template mdspan<topo::elements>(ca);
      const auto x = bounds(m.template axis<ax::x_axis>()),
                 y = bounds(m.template axis<ax::y_axis>());
      const auto z = m.template axis<ax::z_axis>().layout.logical();

      auto elems = [&](util::id y0, util::id y1, util::id x0, util::id x1) {
        std::vector<std::string> result;
        for(auto k : z) {
          std::vector<std::string> col;
          std::stringstream ss_outer;
          for(auto j : util::iota_view(y0, y1)) {
            std::vector<util::id> row;
            std::stringstream ss;
            for(auto i : util::iota_view(x0, x1)) {
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
          << container{elems(y[1], y[2], x[0], x[1])} << '\n';
      out << "z_log_y_log_x_gh_hi: "
          << container{elems(y[1], y[2], x[2], x[3])} << '\n';
      out << "z_log_y_gh_lo_x_log: "
          << container{elems(y[0], y[1], x[1], x[2])} << '\n';
      out << "z_log_y_gh_hi_x_log: "
          << container{elems(y[2], y[3], x[1], x[2])} << '\n';
      out << "z_log_y_gh_lo_x_gh_lo: "
          << container{elems(y[0], y[1], x[0], x[1])} << '\n';
      out << "z_log_y_gh_lo_x_gh_hi: "
          << container{elems(y[0], y[1], x[2], x[3])} << '\n';
      out << "z_log_y_gh_hi_x_gh_lo: "
          << container{elems(y[2], y[3], x[0], x[1])} << '\n';
      out << "z_log_y_gh_hi_x_gh_hi: "
          << container{elems(y[2], y[3], x[2], x[3])} << '\n';
      // clang-format on
    } // d=3
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

  const auto ln_local = m.linear();
  const auto ln_global = m.glinear();

  for(auto && v : m.range(true)) {
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

template<class A, auto... VV>
bool
any_aux(const A & a, util::constants<VV...>) {
  return (a.template axis<VV>().axis.auxiliary || ...);
}
template<std::size_t D, bool V>
int
init_verify_rf(typename mesh<D>::template accessor<ro> m,
  std::conditional_t<V, ints::accessor<ro, ro>, ints::mutator<wo, na>> tf,
  std::size_t sz,
  bool diagonals) {
  UNIT("INIT_VERIFY_RAGGED_FIELD") {
    const auto ln_local = m.linear();
    const auto ln_global = m.glinear();
    const bool aux = any_aux(m, typename mesh<D>::axes());

    for(auto && v : m.range(V)) {
      auto gids = m.global_ids(v);
      auto lid = ln_local(v);
      auto gid = ln_global(gids);
      const bool d = m.check_diag_bounds(v);
      // Some but not all diagonal neighbors send auxiliaries, so don't check
      // for arrival from them:
      ASSERT_TRUE(check_sz(tf, lid, diagonals || !d ? sz : 0) ||
                  (!diagonals && aux && d));
      // Check correctness for all neighbors:
      for(auto n = tf[lid].size(); n--;)
        EXPECT_TRUE(check(tf[lid][n], (int)(gid * 10000 + n)));
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
    std::size_t last = 0, total = 0;
    for(auto [c, m] : mm.components()) { // presumed to be in order
      const auto a = m.axis<mesh1d::axis::x_axis>();
      const util::gid sz = a.axis.extent, off = a.offset;
      if(total) {
        EXPECT_EQ(sz, total);
        EXPECT_EQ(last, off);
      }
      else {
        total = sz;
      }
      last = off + a.logical;
    }
    EXPECT_EQ(last, total);
  };
}

template<mesh4d::axis... AA>
int
check4(const mesh4d::accessor<ro> & m, util::constants<AA...>) {
  UNIT() {
    (
      [&] {
        const axis_info a = m.axis<AA>();
        EXPECT_EQ(a.layout.extent(), 4);
        EXPECT_EQ(a.layout.logical<0>(), 1);
        EXPECT_EQ(a.layout.logical<1>(), 3);
        EXPECT_EQ(a.layout.ghost<0>(), a.low());
        EXPECT_EQ(a.layout.ghost<1>(), 3 + a.low());
        EXPECT_EQ(a.layout.extended<0>(), a.high());
        EXPECT_EQ(a.layout.extended<1>(), 3 + a.high());
      }(),
      ...);
  };
}
int
check_4dmesh(mesh4d::accessor<ro> m) {
  return check4(m, mesh4d::axes());
} // check_4dmesh

template<std::size_t D, bool A>
int
value_rewrite_rf(typename mesh<D>::template accessor<ro> m,
  ints::accessor<wo, na> a) {
  UNIT("REWRITE_RF") {
    const auto ln_local = m.linear();
    for(auto && v : m.range(A)) {
      for(auto & x : a[ln_local(v)])
        x = -x;
    }
  };
}

template<std::size_t D>
int
value_rewrite_verify_rf(typename mesh<D>::template accessor<ro> m,
  ints::accessor<ro, ro> tf) {
  UNIT("REWRITE_VERIFY_RF") {
    const auto ln_local = m.linear();
    const auto ln_global = m.glinear();
    for(auto && v : m.range(true)) {
      auto gids = m.global_ids(v);
      auto lid = ln_local(v);
      auto gid = ln_global(gids);
      for(std::size_t k = 0; k < tf[lid].size(); ++k)
        EXPECT_EQ(tf[lid][k], (int)(-(gid * 10000 + k)));
    }
  };
}

template<std::size_t D>
[[nodiscard]] int
test_mesh(topo::narray_impl::colors color_dist,
  const typename mesh<D>::gcoord & indices,
  const typename mesh<D>::coord & hdepth,
  const typename mesh<D>::coord & bdepth,
  std::array<bool, D> periodic,
  std::array<bool, D> auxiliary,
  bool diagonals,
  bool full_ghosts,
  std::size_t sz,
  const char * verify,
  bool print_info,
  const field<std::size_t>::definition<mesh<D>> & f,
  field<int, data::ragged>::definition<mesh<D>> & rf) {
  UNIT() {
    typename mesh<D>::index_definition idef;
    idef.axes = mesh<D>::base::make_axes(
      color_dist.empty() ? mesh<D>::base::distribute(processes(), indices)
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
    m.allocate(typename mesh<D>::mpi_coloring(idef));

    execute<init_field>(f(m));

    if(print_info)
      execute<print_field<D>>(m, f(m));

    if(verify) {
      execute<print_field<D>>(m, f(m));
      execute<update_field<D>>(m, f(m));
      execute<print_field<D>>(m, f(m));
      EXPECT_EQ(test<check_mesh_field<D>>(verify, m, f(m)), 0);
    }

    // ragged field
    auto & tf = rf(m).get_elements();

    // Set growth policy lo=0.1 to maintain correct size for all D
    tf.growth = {0, 0, 0.1, 0.5, 1};
    execute<allocate_field<D>>(f(m), tf.sizes(), sz);

    EXPECT_EQ((test<init_verify_rf<D, false>>(m, rf(m), sz, diagonals)), 0);

    if(print_info)
      execute<print_rf<D>>(m, rf(m));

    EXPECT_EQ((test<init_verify_rf<D, true>>(m, rf(m), sz, diagonals)), 0);

    if(print_info)
      execute<print_rf<D>>(m, rf(m));
  };
}

int
narray_driver() {
  UNIT() {
    {
      using topo::narray_impl::factor;
      using V = std::vector<std::size_t>;
      EXPECT_EQ(factor(2 * 5 * 11 * 13 * 29), (V{29, 13, 11, 5, 2}));
      EXPECT_EQ(factor(2 * 2 * 23 * 23), (V{23, 23, 2, 2}));
    } // scope

    {
      // 1D Mesh
      mesh1d::slot m1;

      mesh1d::gcoord indices{9};
      mesh1d::index_definition idef;
      idef.axes = mesh1d::base::make_axes(processes(), indices);
      idef.axes[0].hdepth = 1;
      idef.axes[0].bdepth = 2;
      idef.diagonals = true;
      idef.full_ghosts = true;

      m1.allocate(mesh1d::mpi_coloring(idef));
      execute<init_field>(f1(m1));
      execute<print_field<1>>(m1, f1(m1));
      execute<update_field<1>>(m1, f1(m1));
      execute<print_field<1>>(m1, f1(m1));
      EXPECT_EQ(test<check_mesh_field<1>>("1d", m1, f1(m1)), 0);

      if(FLECSI_BACKEND != FLECSI_BACKEND_mpi &&
         FLECSI_BACKEND != FLECSI_BACKEND_hpx) {
        auto lm = data::launch::make(m1, data::launch::gather(m1.colors(), 1));
        EXPECT_EQ(test<check_contiguous>(lm), 0);
      }

      // ragged field
      std::size_t sz = 100;

      auto & tf = rf1(m1).get_elements();
      tf.growth = {0, 0, 0.25, 0.5, 1};
      execute<allocate_field<1>>(f1(m1), tf.sizes(), sz);

      execute<init_verify_rf<1, false>>(m1, rf1(m1), sz, true);
      execute<print_rf<1>>(m1, rf1(m1));

      // tests the case where the periodic flag is false, but with non-zero
      // bdepth, communication is expected only for the halo layers.
      EXPECT_EQ((test<init_verify_rf<1, true>>(m1, rf1(m1), sz, true)), 0);
      execute<print_rf<1>>(m1, rf1(m1));

      // tests if a rewrite of values on ragged with an accessor triggers a
      // ghost copy
      execute<value_rewrite_rf<1, false>>(m1, rf1(m1));
      EXPECT_EQ((test<value_rewrite_verify_rf<1>>(m1, rf1(m1))), 0);
    } // scope

    {
      // 2D Mesh
      const auto test = [&](const mesh<2>::coord & hdepth,
                          const mesh<2>::coord & bdepth,
                          std::array<bool, 2> periodic,
                          std::array<bool, 2> auxiliary,
                          bool diagonals,
                          bool full_ghosts,
                          std::size_t sz,
                          const char * verify,
                          bool print_info,
                          int line) {
        EXPECT_EQ(test_mesh<2>({},
                    {8, 8},
                    hdepth,
                    bdepth,
                    periodic,
                    auxiliary,
                    diagonals,
                    full_ghosts,
                    sz,
                    verify,
                    print_info,
                    f2,
                    rf2),
          0)
          << "from " << line;
      };

      // Primary cells: diagonals variation, full ghosts on, though here full
      // ghosts don't have any meaning.
      test({1, 2},
        {1, 2},
        {true, true},
        {false, false},
        true,
        true,
        100,
        "2d",
        false,
        __LINE__);

      test({1, 1},
        {1, 1},
        {true, true},
        {false, false},
        false,
        true,
        3,
        nullptr,
        true,
        __LINE__);

      // Aux verts: diagonals on, full ghosts true
      test({2, 2},
        {0, 0},
        {false, false},
        {true, true},
        true,
        true,
        3,
        nullptr,
        false,
        __LINE__);

      test({2, 2},
        {0, 0},
        {false, false},
        {true, true},
        true,
        false,
        3,
        nullptr,
        false,
        __LINE__);

      // Aux x-edges: diagonals on, full ghosts true
      test({2, 2},
        {0, 0},
        {false, false},
        {true, false},
        true,
        true,
        3,
        nullptr,
        false,
        __LINE__);

      test({2, 2},
        {0, 0},
        {false, false},
        {true, false},
        true,
        false,
        3,
        nullptr,
        false,
        __LINE__);

      // Aux y-edges: diagonals != full ghosts
      test({2, 2},
        {0, 0},
        {false, false},
        {false, true},
        false,
        true,
        3,
        "diag",
        false,
        __LINE__);

      test({2, 2},
        {0, 0},
        {false, false},
        {false, true},
        true,
        false,
        3,
        nullptr,
        false,
        __LINE__);
    } // scope

    {
      // 3D
      const auto test = [&](topo::narray_impl::colors color_dist,
                          const mesh<3>::gcoord & indices,
                          const mesh<3>::coord & bdepth,
                          std::array<bool, 3> periodic,
                          bool diagonals,
                          std::size_t sz,
                          const char * verify,
                          int line) {
        EXPECT_EQ(test_mesh<3>(color_dist,
                    indices,
                    {1, 1, 1},
                    bdepth,
                    periodic,
                    {false, false, false},
                    diagonals,
                    true,
                    sz,
                    verify,
                    false,
                    f3,
                    rf3),
          0)
          << "from " << line;
      };

      test({},
        {4, 4, 4},
        {1, 1, 1},
        {true, true, true},
        true,
        100,
        "3d",
        __LINE__);
      test({2, 2, 1},
        {4, 4, 2},
        {0, 0, 0},
        {false, false, false},
        false,
        3,
        nullptr,
        __LINE__);
    } // scope

    if(FLECSI_BACKEND != FLECSI_BACKEND_mpi &&
       FLECSI_BACKEND != FLECSI_BACKEND_hpx) {
      // 4D Mesh
      mesh4d::slot m4;

      mesh4d::gcoord indices{4, 4, 4, 4};
      mesh4d::index_definition idef;
      idef.axes = mesh4d::base::make_axes(16, indices);
      for(auto & a : idef.axes) {
        a.hdepth = 1;
        a.bdepth = 1;
      }
      idef.diagonals = true;
      idef.full_ghosts = true;

      m4.allocate(mesh4d::mpi_coloring(idef));
      EXPECT_EQ(test<check_4dmesh>(m4), 0);
    }

  }; // UNIT
} // narray_driver

util::unit::driver<narray_driver> nd;

/// Coloring testing

int
coloring_driver() {
  UNIT() {
    mesh3d::gcoord indices{9, 9, 9};

    mesh3d::index_definition idef;
    idef.axes = mesh3d::base::make_axes(9, indices);
    topo::narray_impl::linearize<3, Color> glin;
    for(Dimension d = 0; d < 3; ++d) {
      auto & a = idef.axes[d];
      glin.strs[d] = a.colormap.size();
      a.hdepth = 1;
    }
    const auto cc = idef.process_colors();

    const auto verify = [&](const std::string & name,
                          const mesh3d::index_definition & id) {
      std::string output_file = "coloring_" + name + "_" +
                                std::to_string(processes()) + "_" +
                                std::to_string(process()) + ".blessed";
      auto & out = UNIT_CAPTURE();
      std::vector<std::size_t> color;
      std::vector<std::string> global, extent, offset, logical, extended;

      auto seq = [](const auto & c) {
        std::stringstream ss;
        ss << flog::container{c};
        return ss.str();
      };

      for(const auto & c3 : cc) {
        std::vector<std::size_t> g, e, o;
        std::vector<std::string> log, ext;
        color.push_back(glin({c3[0], c3[1], c3[2]}));
        for(Dimension d = 0; d < 3; ++d) {
          const auto axco = id.make_axis(d, c3[d]);
          const auto al = axco();
          g.push_back(axco.axis.extent);
          e.push_back(al.extent());
          o.push_back(axco.offset);
          log.push_back(seq(std::vector{al.logical<0>(), al.logical<1>()}));
          ext.push_back(seq(std::vector{al.extended<0>(), al.extended<1>()}));
        }
        global.emplace_back(seq(g));
        extent.emplace_back(seq(e));
        offset.emplace_back(seq(o));
        logical.emplace_back(seq(log));
        extended.emplace_back(seq(ext));
      }

      out << "color: " << flog::container{color} << "\n";
      out << "global: " << flog::container{global} << "\n";
      out << "extent: " << flog::container{extent} << "\n";
      out << "offset: " << flog::container{offset} << "\n";
      out << "logical: " << flog::container{logical} << "\n";
      out << "extended: " << flog::container{extended} << "\n";

      EXPECT_TRUE(UNIT_EQUAL_BLESSED(output_file));
    };

    verify("primary_9x9x9_3x3x1", idef);

    // Create definition for an auxiliary index space
    auto adef = idef;
    adef.axes[0].auxiliary = true;
    adef.axes[1].auxiliary = true;
    adef.full_ghosts = false;

    verify("aux_xy_9x9x9_3x3x1", adef);

    // Change the halo depths for the primary index space
    for(auto & a : idef.axes) {
      a.hdepth = 2;
    }

    adef = idef;
    adef.axes[0].auxiliary = true;
    adef.full_ghosts = true;

    verify("aux_x_9x9x9_3x3x1_fg", adef);
  };
}

util::unit::driver<coloring_driver> cd;
