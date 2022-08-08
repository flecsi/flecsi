// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#include "simple_definition.hh"

#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/topo/unstructured/coloring_utils.hh"
#include "flecsi/topo/unstructured/interface.hh"
#include "flecsi/util/parmetis.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;

struct unstructured : topo::specialization<topo::unstructured, unstructured> {

  /*--------------------------------------------------------------------------*
    Structure
   *--------------------------------------------------------------------------*/

  enum class index_space { vertices = 160, cells = 2718 };
  static constexpr auto cells = index_space::cells;
  static constexpr auto vertices = index_space::vertices;
  using index_spaces = has<cells, vertices>;
  using connectivities =
    list<from<cells, to<vertices>>, from<vertices, to<cells>>>;

  enum entity_list { special, owned };
  using entity_lists = list<entity<vertices, has<special, owned>>>;

  template<auto>
  static constexpr PrivilegeCount privilege_count = 3;

  /*--------------------------------------------------------------------------*
    Interface
   *--------------------------------------------------------------------------*/

  template<class B>
  struct interface : B {

    auto cells() {
      return B::template entities<index_space::cells>();
    }

    template<index_space F>
    auto cells(topo::id<F> from) {
      return B::template entities<index_space::cells>(from);
    }

    auto vertices() {
      return B::template entities<index_space::vertices>();
    }

    template<index_space F>
    auto vertices(topo::id<F> from) const {
      return B::template entities<index_space::vertices>(from);
    }

    template<typename B::entity_list L>
    auto vertices() {
      return B::template special_entities<index_space::vertices, L>();
    }
  }; // struct interface

  /*--------------------------------------------------------------------------*
    Coloring
   *--------------------------------------------------------------------------*/

  static coloring color(std::string const & filename) {

    topo::unstructured_impl::simple_definition sd(filename.c_str());

    const Color colors = 4;

    auto [naive, c2v, v2c, c2c] = topo::unstructured_impl::make_dcrs(sd, 1);
    auto raw = util::parmetis::color(naive, colors);

    auto [primaries, p2m, m2p] =
      topo::unstructured_impl::migrate(naive, colors, raw, c2v, v2c, c2c);

    // clang-format off
    topo::unstructured_impl::coloring_definition cd{
      colors /* global colors */,
      core::index<index_space::cells> /* primary index */,
      2 /* primary dimension */,
      1 /* halo depth */,
      core::index<index_space::vertices> /* vertex index */,
      {/* auxiliary entities */}
    };
    // clang-format on

    return topo::unstructured_impl::color(
      sd, cd, raw, primaries, c2v, v2c, c2c, m2p, p2m);
  } // color

  /*--------------------------------------------------------------------------*
    Initialization
   *--------------------------------------------------------------------------*/

  static void init_cnx(field<util::id, data::ragged>::mutator<wo, wo, na>) {}

  static void init_c2v(
    data::multi<field<util::id, data::ragged>::mutator<wo, wo, na>> mc2v,
    std::vector<base::process_coloring> const & prc_clrngs,
    std::vector<std::map<std::size_t, std::size_t>> const & vmaps) {

    auto pcs = prc_clrngs.begin();
    auto vms = vmaps.begin();
    for(auto & c2v : mc2v.accessors()) {
      auto const & pc = *pcs++;
      auto const & vm = *vms++;
      std::size_t off{0};

      auto const & cnx = pc.cnx_colorings[core::index<vertices>];
      for(std::size_t c{0}; c < cnx.offsets.size() - 1; ++c) {
        const std::size_t start = cnx.offsets[off];
        const std::size_t size = cnx.offsets[off + 1] - start;
        c2v[c].resize(size);

        for(std::size_t i{0}; i < size; ++i) {
          c2v[c][i] = vm.at(cnx.indices[start + i]);
        } // for

        ++off;
      } // for
    } // for
  } // init_c2v

  static void allocate_owned(
    data::multi<topo::array<unstructured>::accessor<wo>> aa,
    const std::vector<base::process_coloring> & vpc) {

    auto it = vpc.begin();
    for(auto & a : aa.accessors()) {
      a.size() = it++->coloring.owned.size();
    }

    // a = pc.coloring.owned.size();
  } // allocate_owned

  static void init_owned(data::multi<field<util::id>::accessor<wo>> m,
    const std::vector<base::process_coloring> & vpc,
    const std::vector<std::map<std::size_t, std::size_t>> & rmaps) {
    auto it = vpc.begin();
    std::size_t c = 0;
    for(auto & a : m.accessors()) {
      std::size_t i{0};
      for(auto e : it++->coloring.owned) {
        a[i++] = rmaps[c].at(e);
      }
      c++;
    }
  } // init_owned

  static void initialize(data::topology_slot<unstructured> & s,
    coloring const & c) {
    flog(warn) << flog::container{c.partitions} << std::endl;

    auto & c2v =
      s->get_connectivity<index_space::cells, index_space::vertices>();
    auto & v2c =
      s->get_connectivity<index_space::vertices, index_space::cells>();
    auto const & vmaps = s->reverse_map<index_space::vertices>();

    execute<init_cnx>(c2v(s));
    execute<init_cnx>(v2c(s));

    auto lm = data::launch::make(s);
    execute<init_c2v, mpi>(c2v(lm), c.idx_spaces[core::index<cells>], vmaps);
    constexpr PrivilegeCount NPC = privilege_count<index_space::cells>;
    constexpr PrivilegeCount NPV = privilege_count<index_space::vertices>;
    execute<topo::unstructured_impl::transpose<NPC, NPV>>(c2v(s), v2c(s));

    // owned vertices setup
    auto & owned_vert_f =
      s->special_.get<index_space::vertices>().get<entity_list::owned>();

    {
      auto slm = data::launch::make(owned_vert_f);
      execute<allocate_owned, flecsi::mpi>(
        slm, c.idx_spaces[core::index<vertices>]);
      owned_vert_f.resize();
    }

    // the launch maps need to be resized with the correct allocation
    auto slm = data::launch::make(owned_vert_f);
    execute<init_owned>(
      core::special_field(slm), c.idx_spaces[core::index<vertices>], vmaps);

  } // initialize
}; // struct unstructured

void
print(unstructured::accessor<ro, ro, ro> m,
  field<util::id>::accessor<ro, ro, ro> cids,
  field<util::id>::accessor<ro, ro, ro> vids) {

  std::stringstream ss;
  ss << "color(" << color() << ")" << std::endl;
  for(auto c : m.cells()) {
    ss << "cell(" << cids[c] << "," << c << "): ";
    for(auto v : m.vertices(c)) {
      ss << vids[v] << " ";
    }
    ss << std::endl;
  }
  flog(info) << ss.str() << std::endl;

  ss.str("");
  for(auto v : m.vertices()) {
    ss << "vertex(" << vids[v] << "," << v << "): ";
    for(auto c : m.cells(v)) {
      ss << cids[c] << " ";
    }
    ss << std::endl;
  }
  flog(info) << ss.str() << std::endl;
}

void
init_field(unstructured::accessor<ro, ro, ro> m,
  field<util::id>::accessor<ro, ro, ro> vids,
  field<int, data::ragged>::mutator<wo, wo, na> tf) {
  for(auto v : m.vertices<unstructured::owned>()) {
    tf[v].resize(100);
    for(int i = 0; i < 100; ++i)
      tf[v][i] = (int)(vids[v] * 10000 + i);
  }
} // init_field

void
print_field(unstructured::accessor<ro, ro, ro> m,
  field<int, data::ragged>::accessor<ro, ro, ro> tf) {

  std::stringstream ss;
  ss << " Number of vertices = " << m.vertices().size() << "\n";
  for(auto v : m.vertices()) {
    ss << "For vertex " << v << ", field_size = " << tf[v].size()
       << ", field vals = [ ";
    for(std::size_t i = 0; i < tf[v].size(); ++i)
      ss << tf[v][i] << "  ";
    ss << "]\n\n";
  }
  flog(info) << ss.str() << std::endl;
} // print_field

void
allocate_field(unstructured::accessor<ro, ro, ro> m,
  topo::resize::Field::accessor<wo> a) {
  a = m.vertices().size() * 100;
}

int
verify_field(unstructured::accessor<ro, ro, ro> m,
  field<util::id>::accessor<ro, ro, ro> vids,
  field<int, data::ragged>::accessor<ro, ro, ro> tf) {
  UNIT("VERIFY_FIELD") {
    for(auto v : m.vertices()) {
      EXPECT_EQ(tf[v].size(), 100);
      for(int i = 0; i < 100; ++i)
        EXPECT_EQ(tf[v][i], (int)(vids[v] * 10000 + i));
    }
  };
}

unstructured::slot mesh;
unstructured::cslot coloring;
field<int, data::ragged>::definition<unstructured, unstructured::vertices>
  test_field;

int
unstructured_driver() {
  UNIT() {
    coloring.allocate("simple2d-8x8.msh");
    mesh.allocate(coloring.get());

    auto const & cids = mesh->forward_map<unstructured::cells>();
    auto const & vids = mesh->forward_map<unstructured::vertices>();
    execute<print>(mesh, cids(mesh), vids(mesh));

    auto & tf = test_field(mesh).get_ragged();
    tf.growth = {0, 0, 0.25, 0.5, 1};
    execute<allocate_field>(mesh, tf.sizes());
    tf.resize();

    execute<init_field>(mesh, vids(mesh), test_field(mesh));
    // execute<print_field>(mesh, test_field(mesh));
    EXPECT_EQ(test<verify_field>(mesh, vids(mesh), test_field(mesh)), 0);

#if 0
    auto & neuf =
      mesh->special_.get<unstructured::edges>().get<unstructured::neumann>();
    execute<allocate>(neuf.sizes());
    neuf.resize();
    execute<init>(mesh->special_field(neuf));
    EXPECT_EQ(test<check>(mesh), 0);
#endif
  };
} // unstructured_driver

flecsi::unit::driver<unstructured_driver> driver;
