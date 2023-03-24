#include "flecsi/data.hh"
#include "flecsi/flog.hh"
#include "flecsi/topo/unstructured/coloring_utils.hh"
#include "flecsi/topo/unstructured/interface.hh"
#include "flecsi/util/parmetis.hh"
#include "simple_definition.hh"

struct unstructured
  : flecsi::topo::specialization<flecsi::topo::unstructured, unstructured> {

  using point = flecsi::topo::unstructured_impl::simple_definition::point;

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

  static const inline flecsi::field<std::size_t>::definition<unstructured,
    cells>
    cid;
  static const inline flecsi::field<std::size_t>::definition<unstructured,
    vertices>
    vid;
  static const inline flecsi::field<point>::definition<unstructured, vertices>
    coords;

  struct init {
    std::vector<std::vector<std::size_t>> cid;
    std::vector<std::vector<std::size_t>> id;
    std::vector<std::vector<point>> coords;
    std::vector<flecsi::util::crs> c2v_connectivity;
    std::vector<flecsi::util::crs> e2c_connectivity;
    std::vector<flecsi::util::crs> e2v_connectivity;
  };

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

  static coloring color(std::string const & filename, init & fields) {
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
    cu.color_primaries(1, flecsi::util::parmetis::color);
    cu.migrate_primaries();
    cu.close_primaries();

    // Vertices
    cu.color_vertices();
    cu.close_vertices();

    // Edges
    cu.build_auxiliary(edges);
    cu.color_auxiliary(edges);
    cu.close_auxiliary(edges, core::index<edges>);

    // Create distributed vector of vector for extra field: cell id
    std::vector<std::size_t> lcid;
    if(flecsi::process() == 0) {
      const flecsi::util::iota_view cs({}, sd.num_entities(cells));
      lcid.assign(cs.begin(), cs.end());
    }
    fields.cid = cu.send_field(cells, lcid);

    // Create distributed vector of vector for both the extra fields: vertex id
    // and coordinates
    std::vector<std::size_t> lvid;
    if(flecsi::process() == 0) {
      auto nv = sd.num_entities(vertices);
      lvid.reserve(nv);
      for(unsigned long i = 0; i < nv; ++i)
        lvid.push_back(sd.vertex_field(i));
    }
    fields.id = cu.send_field(vertices, lvid);

    std::vector<point> lcoords;
    if(flecsi::process() == 0) {
      auto nv = sd.num_entities(vertices);
      lcoords.reserve(nv);
      for(unsigned long i = 0; i < nv; ++i)
        lcoords.push_back(sd.vertex(i));
    }
    fields.coords = cu.send_field(vertices, lcoords);

    fields.c2v_connectivity = cu.get_connectivity(cells, vertices);
    fields.e2c_connectivity = cu.get_connectivity(edges, cells);
    fields.e2v_connectivity = cu.get_connectivity(edges, vertices);

    return cu.generate();
  } // color

  /*--------------------------------------------------------------------------*
    Initialization
   *--------------------------------------------------------------------------*/

  template<entity_list E>
  static auto get_list(const base::index_color & ic) {
    if constexpr(E == owned) {
      return ic.owned();
    }
    else if constexpr(E == shared) {
      return ic.shared();
    }
    else {
      static_assert(E == ghost);
      return flecsi::util::transform_view(
        ic.ghost, [](auto & g) { return g.lid; });
    }
  }

  template<entity_list E>
  static void allocate_list(
    flecsi::data::multi<flecsi::topo::array<unstructured>::accessor<flecsi::wo>>
      aa,
    const std::vector<base::index_color> & vic) {

    auto it = vic.begin();
    for(auto & a : aa.accessors()) {
      a.size() = get_list<E>(*it++).size();
    }
  } // allocate_list

  template<entity_list E>
  static void populate_list(
    flecsi::data::multi<flecsi::field<flecsi::util::id>::accessor<flecsi::wo>>
      m,
    const std::vector<base::index_color> & vic) {
    auto it = vic.begin();
    for(auto & a : m.accessors()) {
      auto elements = get_list<E>(*it++);
      std::copy(elements.begin(), elements.end(), a.span().begin());
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
      execute<allocate_list<E>, flecsi::mpi>(
        slm, c.idx_spaces[core::index<I>].colors);
      el.resize();
    }

    // the launch maps need to be resized with the correct allocation
    auto slm = data::launch::make(el);
    execute<populate_list<E>, flecsi::mpi>(
      core::special_field(slm), c.idx_spaces[core::index<I>].colors);
  } // init_list

  static void init_vid_coords(
    flecsi::data::multi<
      unstructured::accessor<flecsi::ro, flecsi::ro, flecsi::ro>> m,
    flecsi::data::multi<flecsi::field<
      flecsi::util::gid>::accessor<flecsi::wo, flecsi::wo, flecsi::na>> mcid,
    flecsi::data::multi<flecsi::field<
      flecsi::util::gid>::accessor<flecsi::wo, flecsi::wo, flecsi::na>> mvid,
    flecsi::data::multi<
      flecsi::field<point>::accessor<flecsi::wo, flecsi::wo, flecsi::na>>
      mcoords,
    const std::vector<std::vector<std::size_t>> & cid,
    const std::vector<std::vector<std::size_t>> & vid,
    const std::vector<std::vector<point>> & coords) {
    const auto ma = m.accessors();
    auto acid = mcid.accessors();
    auto avid = mvid.accessors();
    auto acoords = mcoords.accessors();
    for(unsigned int i = 0; i < m.depth(); ++i) {
      int j = 0;
      for(auto v : ma[i].vertices<unstructured::owned>()) {
        avid[i][v] = vid[i][j];
        acoords[i][v] = coords[i][j++];
      }
      j = 0;
      for(auto c : ma[i].cells<unstructured::owned>()) {
        acid[i][c] = cid[i][j++];
      }
    }
  } // init_vid_coords

  static void initialize(flecsi::data::topology_slot<unstructured> & s,
    coloring const & c,
    const init & fields) {
    using namespace flecsi;
    using namespace topo::unstructured_impl;

    auto & c2v = s->get_connectivity<cells, vertices>();
    auto & v2c = s->get_connectivity<vertices, cells>();
    auto & e2c = s->get_connectivity<edges, cells>();
    auto & e2v = s->get_connectivity<edges, vertices>();

    c2v(s).get_elements().resize();
    v2c(s).get_elements().resize();
    e2c(s).get_elements().resize();
    e2v(s).get_elements().resize();

    {
      auto lm = data::launch::make(s);
      constexpr PrivilegeCount NPC = privilege_count<index_space::cells>;
      constexpr PrivilegeCount NPV = privilege_count<index_space::vertices>;
      constexpr PrivilegeCount NPE = privilege_count<index_space::edges>;
      execute<init_connectivity<NPC>, mpi>(c2v(lm), fields.c2v_connectivity);
      execute<init_connectivity<NPE>, mpi>(e2c(lm), fields.e2c_connectivity);
      execute<init_connectivity<NPE>, mpi>(e2v(lm), fields.e2v_connectivity);
      execute<transpose<NPC, NPV>>(c2v(s), v2c(s));
    }

    init_list<index_space::cells, entity_list::owned>(s, c);
    init_list<index_space::cells, entity_list::shared>(s, c);
    init_list<index_space::cells, entity_list::ghost>(s, c);
    init_list<index_space::vertices, entity_list::owned>(s, c);
    init_list<index_space::vertices, entity_list::shared>(s, c);
    init_list<index_space::vertices, entity_list::ghost>(s, c);

    // Init the specific fields on the topology using user data
    auto lm = data::launch::make(s);
    execute<init_vid_coords, flecsi::mpi>(
      lm, cid(lm), vid(lm), coords(lm), fields.cid, fields.id, fields.coords);

  } // initialize
}; // struct unstructured
