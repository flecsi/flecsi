#include "flecsi/data.hh"
#include "flecsi/topo/unstructured/coloring_utils.hh"
#include "flecsi/topo/unstructured/interface.hh"
#include "flecsi/util/parmetis.hh"
#include "simple_definition.hh"

struct unstructured
  : flecsi::topo::specialization<flecsi::topo::unstructured, unstructured> {

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
  using entity_lists =
    list<entity<vertices, has<special, owned>>, entity<cells, has<owned>>>;

  template<auto>
  static constexpr flecsi::PrivilegeCount privilege_count = 3;

  /*--------------------------------------------------------------------------*
    Interface
   *--------------------------------------------------------------------------*/

  template<class B>
  struct interface : B {

    auto cells() {
      return B::template entities<index_space::cells>();
    }

    template<typename B::entity_list L>
    auto cells() {
      return B::template special_entities<index_space::cells, L>();
    }

    template<index_space F>
    auto cells(flecsi::topo::id<F> from) {
      return B::template entities<index_space::cells>(from);
    }

    auto vertices() {
      return B::template entities<index_space::vertices>();
    }

    template<index_space F>
    auto vertices(flecsi::topo::id<F> from) const {
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
    auto edges(flecsi::topo::id<F> from) const {
      return B::template entities<index_space::edges>(from);
    }

  }; // struct interface

  /*--------------------------------------------------------------------------*
    Coloring
   *--------------------------------------------------------------------------*/

  static coloring color(std::string const & filename) {
    using namespace flecsi::topo::unstructured_impl;
    simple_definition sd(filename.c_str());

    const flecsi::Color colors = 4;

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
    cu.color_primaries(flecsi::util::parmetis::color);
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
    flecsi::data::multi<flecsi::topo::array<unstructured>::accessor<flecsi::wo>>
      aa,
    const std::vector<base::process_coloring> & vpc) {

    auto it = vpc.begin();
    for(auto & a : aa.accessors()) {
      a.size() = it++->coloring.owned.size();
    }

    // a = pc.coloring.owned.size();
  } // allocate_owned

  static void init_owned(
    flecsi::data::multi<flecsi::field<flecsi::util::id>::accessor<flecsi::wo>>
      m,
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

  static void initialize(flecsi::data::topology_slot<unstructured> & s,
    coloring const & c) {
    using namespace flecsi;
    using namespace topo::unstructured_impl;

    auto & c2v = s->get_connectivity<cells, vertices>();
    auto & v2c = s->get_connectivity<vertices, cells>();
    auto & e2c = s->get_connectivity<edges, cells>();
    auto & e2v = s->get_connectivity<edges, vertices>();
    auto const & cmaps = s->reverse_map<cells>();
    auto const & vmaps = s->reverse_map<vertices>();

    c2v(s).get_ragged().resize();
    v2c(s).get_ragged().resize();
    e2c(s).get_ragged().resize();
    e2v(s).get_ragged().resize();

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

    // owned cells setup
    auto & owned_cell_f =
      s->special_.get<index_space::cells>().get<entity_list::owned>();

    {
      auto slm = data::launch::make(owned_cell_f);
      execute<allocate_owned, flecsi::mpi>(
        slm, c.idx_spaces[core::index<cells>]);
      owned_cell_f.resize();
    }

    // the launch maps need to be resized with the correct allocation
    auto slmc = data::launch::make(owned_cell_f);
    execute<init_owned>(
      core::special_field(slmc), c.idx_spaces[core::index<cells>], cmaps);
  } // initialize
}; // struct unstructured
