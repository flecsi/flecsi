// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#include "simple_definition.hh"

#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/topo/unstructured/coloring_utils.hh"
#include "flecsi/topo/unstructured/interface.hh"
#include "flecsi/util/parmetis.hh"
#include "flecsi/util/unit.hh"

#include <optional>

using namespace flecsi;
using namespace flecsi::topo::unstructured_impl;

struct unstructured : topo::specialization<topo::unstructured, unstructured> {

  /*--------------------------------------------------------------------------*
    Structure
   *--------------------------------------------------------------------------*/

  // The ids for the entities depend on the mesh definition ids.
  // This example uses the topological dimension of the entity type, which
  // corresponds to the ids understood by the simple definition used below.
  enum index_space { vertices = 0, cells = 2, edges = 1 };

  using index_spaces = has<cells, vertices, edges>;

  using connectivities = list<from<cells, to<vertices>>,
    from<edges, to<vertices, cells>>,
    from<vertices, to<cells>>>;

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

    auto edges() {
      return B::template entities<index_space::edges>();
    }

    template<index_space F>
    auto edges(topo::id<F> from) const {
      return B::template entities<index_space::edges>(from);
    }

  }; // struct interface

  /*--------------------------------------------------------------------------*
    Coloring
   *--------------------------------------------------------------------------*/

  static coloring color(std::string const & filename) {
    simple_definition sd(filename.c_str());

    const Color colors = 4;

    // clang-format off
    coloring_utils cu(&sd,
      {
        colors,
        {
          cells /* primary id */,
          core::index<cells> /* primary index */
        },
        1 /* halo depth */,
        {
          vertices /* vertex id */,
          core::index<vertices> /* vertex index */
        },
        {
          {
            edges /* edge id */,
            core::index<index_space::edges> /* edge index */
          }
        }
      },
      {
        { core::index<cells>, core::index<vertices>, false },
        { core::index<vertices>, core::index<cells>, true },
        { core::index<edges>, core::index<cells>, false },
        { core::index<edges>, core::index<vertices>, false }
      }
    );
    // clang-format on

    // Primaries
    cu.create_graph<true>(cells, 1);
    cu.color_primaries(util::parmetis::color);
    cu.migrate_primaries();
    cu.close_primaries();

    // Vertices
    cu.color_vertices();
    cu.migrate_vertices();
    cu.close_vertices();

    // Edges
    cu.build_auxiliary(edges);
    cu.color_auxiliary(edges);
    cu.close_auxiliary(edges, core::index<edges>);

    return cu.generate();
    // return cu.generate();
  } // color

  /*--------------------------------------------------------------------------*
    Initialization
   *--------------------------------------------------------------------------*/

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

  static void touch_cnx(field<util::id, data::ragged>::mutator<wo, wo, na>) {}

  static void initialize(data::topology_slot<unstructured> & s,
    coloring const & c) {
    auto & c2v = s->get_connectivity<cells, vertices>();
    auto & v2c = s->get_connectivity<vertices, cells>();
    auto & e2c = s->get_connectivity<edges, cells>();
    auto & e2v = s->get_connectivity<edges, vertices>();
    auto const & cmaps = s->reverse_map<cells>();
    auto const & vmaps = s->reverse_map<vertices>();

    execute<touch_cnx>(c2v(s));
    execute<touch_cnx>(v2c(s));
    execute<touch_cnx>(e2c(s));
    execute<touch_cnx>(e2v(s));

    auto lm = data::launch::make(s);
    constexpr PrivilegeCount NPC = privilege_count<index_space::cells>;
    constexpr PrivilegeCount NPV = privilege_count<index_space::vertices>;
    constexpr PrivilegeCount NPE = privilege_count<index_space::edges>;
    execute<init_connectivity<core::index<cells>, core::index<vertices>, NPC>,
      mpi>(c2v(lm), c, vmaps);
    execute<init_connectivity<core::index<edges>, core::index<cells>, NPE>,
      mpi>(e2c(lm), c, cmaps);
    execute<init_connectivity<core::index<edges>, core::index<vertices>, NPE>,
      mpi>(e2v(lm), c, vmaps);
    execute<transpose<NPC, NPV>>(c2v(s), v2c(s));

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
  field<util::id>::accessor<ro, ro, ro> vids,
  field<util::id>::accessor<ro, ro, ro> eids) {

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

  ss.str("");
  for(auto e : m.edges()) {
    ss << "edge(" << eids[e] << "," << e << "): ";
    for(auto c : m.cells(e)) {
      ss << cids[c] << " ";
    }
    ss << std::endl;
  }
  flog(info) << ss.str() << std::endl;

  ss.str("");
  for(auto e : m.edges()) {
    ss << "edge(" << eids[e] << "," << e << "): ";
    for(auto v : m.vertices(e)) {
      ss << vids[v] << " ";
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
  std::vector<std::string> files = {
    "simple2d-16x16.msh", "simple2d-8x8.msh", "disconnected.msh"};
  UNIT() {
    for(auto f : files) {
      flog(info) << "testing mesh: " << f << std::endl;
      coloring.allocate(f);
      mesh.allocate(coloring.get());

      auto const & cids = mesh->forward_map<unstructured::cells>();
      auto const & vids = mesh->forward_map<unstructured::vertices>();
      auto const & eids = mesh->forward_map<unstructured::edges>();
      execute<print>(mesh, cids(mesh), vids(mesh), eids(mesh));

      auto & tf = test_field(mesh).get_ragged();
      tf.growth = {0, 0, 0.25, 0.5, 1};
      execute<allocate_field>(mesh, tf.sizes());
      tf.resize();

      execute<init_field>(mesh, vids(mesh), test_field(mesh));
      // execute<print_field>(mesh, test_field(mesh));
      EXPECT_EQ(test<verify_field>(mesh, vids(mesh), test_field(mesh)), 0);
    }
  };
} // unstructured_driver

flecsi::unit::driver<unstructured_driver> driver;
