#ifndef FLECSI_TOPO_UNSTRUCTURED_TEST_FIXED_HH
#define FLECSI_TOPO_UNSTRUCTURED_TEST_FIXED_HH

#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/topo/unstructured/interface.hh"
#include "flecsi/topo/unstructured/types.hh"
#include "flecsi/util/geometry/point.hh"
#include "simple_definition.hh"

#include <string>
#include <vector>

// 'interface' is defined as a macro on some platforms
#undef interface

struct fixed_mesh
  : flecsi::topo::specialization<flecsi::topo::unstructured, fixed_mesh> {

  /*--------------------------------------------------------------------------*
    Structure
   *--------------------------------------------------------------------------*/

  enum index_space { cells, vertices };
  using index_spaces = has<cells, vertices>;
  using connectivities =
    list<from<cells, to<vertices>>, from<vertices, to<cells>>>;

  enum entity_list { owned, shared, ghost };
  using entity_lists = list<entity<cells, has<owned, shared, ghost>>,
    entity<vertices, has<owned, shared, ghost>>>;

  template<auto>
  static constexpr flecsi::PrivilegeCount privilege_count = 3;

  static const inline flecsi::field<flecsi::util::gid>::definition<fixed_mesh,
    cells>
    cid;
  static const inline flecsi::field<flecsi::util::gid>::definition<fixed_mesh,
    vertices>
    vid;

  using point = flecsi::util::point<double, 2>;
  static const inline flecsi::field<point>::definition<fixed_mesh, vertices>
    coords;

  struct init {
    std::vector<std::vector<flecsi::util::gid>> cid;
    std::vector<std::vector<flecsi::util::gid>> vid;
    std::vector<std::vector<point>> vertex_coords;
    std::vector<flecsi::util::crs> c2v_connectivity;
  };

  /*--------------------------------------------------------------------------*
    Interface
   *--------------------------------------------------------------------------*/

  template<class B>
  struct interface : B {

    auto cells() const {
      return B::template entities<index_space::cells>();
    }

    template<typename B::entity_list L>
    auto cells() const {
      return B::template special_entities<index_space::cells, L>();
    }

    template<index_space From>
    auto cells(flecsi::topo::id<From> from) const {
      return B::template entities<index_space::cells>(from);
    }

    auto vertices() const {
      return B::template entities<index_space::vertices>();
    }

    template<typename B::entity_list L>
    auto vertices() const {
      return B::template special_entities<index_space::vertices, L>();
    }

    template<index_space From>
    auto vertices(flecsi::topo::id<From> from) const {
      return B::template entities<index_space::vertices>(from);
    }

  }; // struct interface

  /*--------------------------------------------------------------------------*
    Coloring
   *--------------------------------------------------------------------------*/

  static coloring color(const flecsi::runtime & r,
    std::string const & filename,
    flecsi::Color ncolors,
    init & fields) {
    using namespace flecsi;
    using namespace flecsi::topo::unstructured_impl;
    flog_assert(r.processes() == ncolors, "color to process mismatch");

    simple_definition sd(filename + "." + std::to_string(r.process()));
    fields.cid.push_back(std::move(sd.l2g_cells));
    fields.vid.push_back(std::move(sd.l2g_vertices));
    fields.vertex_coords.push_back(std::move(sd.vertex_coords));
    fields.c2v_connectivity.push_back(std::move(sd.c2v));
    return {// number of global colors
      ncolors,
      {// over index spaces
        {std::move(sd.cell_peers),
          std::move(sd.cell_partitions),
          {// over process colors
            {static_cast<util::id>(fields.cid[0].size()),
              std::move(sd.peer_cells),
              // cnx_allocs
              {0, fields.c2v_connectivity[0].values.size()}}},
          std::move(sd.cell_num_intervals)},
        {std::move(sd.vertex_peers),
          std::move(sd.vertex_partitions),
          {// over process colors
            {static_cast<util::id>(fields.vid[0].size()),
              std::move(sd.peer_vertices),
              // cnx_allocs
              {fields.c2v_connectivity[0].values.size(), 0}}},
          std::move(sd.vertex_num_intervals)}},
      sd.color_peers};
  } // color

  /*--------------------------------------------------------------------------*
    Initialization
   *--------------------------------------------------------------------------*/

  static void init_mesh_ids(
    flecsi::data::multi<
      fixed_mesh::accessor<flecsi::ro, flecsi::ro, flecsi::ro>> m,
    flecsi::data::multi<flecsi::field<
      flecsi::util::gid>::accessor<flecsi::wo, flecsi::wo, flecsi::na>> mcid,
    flecsi::data::multi<flecsi::field<
      flecsi::util::gid>::accessor<flecsi::wo, flecsi::wo, flecsi::na>> mvid,
    flecsi::data::multi<
      flecsi::field<point>::accessor<flecsi::wo, flecsi::wo, flecsi::na>>
      mcoords,
    const std::vector<std::vector<flecsi::util::gid>> & cid,
    const std::vector<std::vector<flecsi::util::gid>> & vid,
    const std::vector<std::vector<point>> & coords) {
    const auto ma = m.accessors();
    auto acid = mcid.accessors();
    auto avid = mvid.accessors();
    auto acoords = mcoords.accessors();
    for(unsigned int i = 0; i < m.depth(); ++i) {
      for(auto v : ma[i].vertices()) {
        avid[i][v] = vid[i][v];
        acoords[i][v] = coords[i][v];
      }
      for(auto c : ma[i].cells()) {
        acid[i][c] = cid[i][c];
      }
    }
  } // init_mesh_ids

  static auto get_owned(const base::index_color & ic) {
    using namespace flecsi;
    std::vector<util::id> ownd;
    std::set<util::id> ghst = ic.ghosts();

    for(util::id e = 0; e < ic.entities; ++e) {
      if(!ghst.count(e)) {
        ownd.push_back(e);
      }
    }
    return ownd;
  }

  static auto get_shared(const base::index_color & ic) {
    std::set<flecsi::util::id> shr;
    for(auto & p : ic.peers) {
      shr.insert(p.second.shared.begin(), p.second.shared.end());
    }
    return shr;
  }

  static auto get_exclusive(const base::index_color & ic) {
    const auto ss = get_shared(ic);
    std::vector<flecsi::util::id> ex;
    for(auto o : get_owned(ic))
      if(!ss.count(o))
        ex.push_back(o);
    return ex;
  }

  template<entity_list E>
  static auto get_list(const base::index_color & ic) {
    if constexpr(E == owned) {
      return get_owned(ic);
    }
    else if constexpr(E == shared) {
      return get_shared(ic);
    }
    else {
      static_assert(E == ghost);
      return ic.ghosts();
    }
  }

  template<entity_list E>
  static void allocate_list(
    flecsi::data::multi<flecsi::topo::resize::Field::accessor<flecsi::wo>> aa,
    const std::vector<base::index_color> & vic) {

    auto it = vic.begin();
    for(auto & a : aa.accessors()) {
      a = get_list<E>(*it++).size();
    }
  }

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
  }

  template<index_space I, entity_list E>
  static void init_list(flecsi::scheduler & s,
    fixed_mesh::topology & m,
    coloring const & c) {
    using namespace flecsi;
    using namespace topo::unstructured_impl;

    auto & el = m.get_special_entities<I, E>();
    auto slm = data::launch::make(s, el.sz);

    execute<allocate_list<E>, flecsi::mpi>(flecsi::topo::resize::field(slm),
      c.idx_spaces[topology::index<I>].colors);
    el.resize();

    auto slm2 = data::launch::make(s, el);

    execute<populate_list<E>, flecsi::mpi>(
      m.special_field(slm2), c.idx_spaces[topology::index<I>].colors);
  }

  static void initialize(flecsi::scheduler & s,
    fixed_mesh::topology & m,
    coloring const & c,
    const init & fields) {
    using namespace flecsi;
    auto & c2v = m.get_connectivity<fixed_mesh::cells, fixed_mesh::vertices>();
    auto & v2c = m.get_connectivity<fixed_mesh::vertices, fixed_mesh::cells>();

    auto lm = data::launch::make(s, m);
    execute<topo::unstructured_impl::init_connectivity<privilege_count<cells>>,
      mpi>(c2v(lm), fields.c2v_connectivity);

    constexpr PrivilegeCount NPC = privilege_count<index_space::cells>;
    constexpr PrivilegeCount NPV = privilege_count<index_space::vertices>;
    s.execute<topo::unstructured_impl::transpose<NPC, NPV>>(c2v(m), v2c(m));

    init_list<index_space::cells, entity_list::owned>(s, m, c);
    init_list<index_space::cells, entity_list::shared>(s, m, c);
    init_list<index_space::cells, entity_list::ghost>(s, m, c);
    init_list<index_space::vertices, entity_list::owned>(s, m, c);
    init_list<index_space::vertices, entity_list::shared>(s, m, c);
    init_list<index_space::vertices, entity_list::ghost>(s, m, c);

    execute<init_mesh_ids, mpi>(lm,
      cid(lm),
      vid(lm),
      coords(lm),
      fields.cid,
      fields.vid,
      fields.vertex_coords);
  } // initialize
}; // struct fixed_mesh

#endif
