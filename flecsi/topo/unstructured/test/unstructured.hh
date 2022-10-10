#include "flecsi/data.hh"
#include "flecsi/flog.hh"
#include "flecsi/topo/unstructured/coloring_utils.hh"
#include "flecsi/topo/unstructured/interface.hh"
#include "flecsi/util/parmetis.hh"
#include "flecsi/util/tikz.hh"
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

  enum entity_list { owned, shared, ghost };
  using entity_lists = list<entity<vertices, has<owned, shared, ghost>>,
    entity<cells, has<owned, shared, ghost>>>;

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
    cu.create_graph(cells);
    cu.color_primaries(1, flecsi::util::parmetis::color);
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

    auto co = cu.generate();

    if(flecsi::processes() <= colors) {
      std::size_t const entities =
        co.idx_spaces[core::index<cells>][0].entities;
      std::size_t M = std::sqrt(entities);

      if(M * M == entities) {
        flecsi::util::tikz::write_closure(M,
          M,
          co.idx_spaces[core::index<cells>],
          co.idx_spaces[core::index<vertices>],
          MPI_COMM_WORLD);
      } // if
    } // if

    return co;
  } // color

  /*--------------------------------------------------------------------------*
    Initialization
   *--------------------------------------------------------------------------*/

  template<entity_list E>
  static void allocate_list(
    flecsi::data::multi<flecsi::topo::array<unstructured>::accessor<flecsi::wo>>
      aa,
    const std::vector<base::process_coloring> & vpc) {

    auto it = vpc.begin();
    for(auto & a : aa.accessors()) {
      if constexpr(E == owned) {
        a.size() = it++->coloring.owned.size();
      }
      else if(E == shared) {
        a.size() = it++->coloring.shared.size();
      }
      else if(E == ghost) {
        a.size() = it++->coloring.ghost.size();
      }
    }
  } // allocate_list

  template<entity_list E>
  static void populate_list(
    flecsi::data::multi<flecsi::field<flecsi::util::id>::accessor<flecsi::wo>>
      m,
    const std::vector<base::process_coloring> & vpc,
    const base::reverse_maps_t & rmaps) {
    auto it = vpc.begin();
    std::size_t c = 0;
    for(auto & a : m.accessors()) {
      std::size_t i{0};
      if constexpr(E == owned) {
        for(auto e : it++->coloring.owned) {
          a[i++] = rmaps[c].at(e);
        }
      }
      else if(E == shared) {
        for(auto e : it++->coloring.shared) {
          a[i++] = rmaps[c].at(e.id);
        }
      }
      else if(E == ghost) {
        for(auto e : it++->coloring.ghost) {
          a[i++] = rmaps[c].at(e.id);
        }
      }
      c++;
    }
  } // populate_list

  template<index_space I, entity_list E>
  static void init_list(flecsi::data::topology_slot<unstructured> & s,
    coloring const & c) {
    using namespace flecsi;
    using namespace topo::unstructured_impl;

    auto & el = s->special_.get<I>().template get<E>();

    {
      auto slm = data::launch::make(el);
      execute<allocate_list<E>, flecsi::mpi>(slm, c.idx_spaces[core::index<I>]);
      el.resize();
    }

    // the launch maps need to be resized with the correct allocation
    auto slm = data::launch::make(el);
    auto const & rmaps = s->reverse_map<I>();
    execute<populate_list<E>>(
      core::special_field(slm), c.idx_spaces[core::index<I>], rmaps);
  } // init_list

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

    init_list<index_space::cells, entity_list::owned>(s, c);
    init_list<index_space::cells, entity_list::shared>(s, c);
    init_list<index_space::cells, entity_list::ghost>(s, c);
    init_list<index_space::vertices, entity_list::owned>(s, c);
    init_list<index_space::vertices, entity_list::shared>(s, c);
    init_list<index_space::vertices, entity_list::ghost>(s, c);
  } // initialize
}; // struct unstructured
