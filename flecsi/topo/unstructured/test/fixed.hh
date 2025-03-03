#ifndef FLECSI_TOPO_UNSTRUCTURED_TEST_FIXED_HH
#define FLECSI_TOPO_UNSTRUCTURED_TEST_FIXED_HH

#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/topo/unstructured/interface.hh"
#include "flecsi/topo/unstructured/types.hh"
#include "simple_definition.hh"

#include <string>
#include <vector>

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

  static const inline flecsi::field<std::size_t>::definition<fixed_mesh, cells>
    cid;
  static const inline flecsi::field<std::size_t>::definition<fixed_mesh,
    vertices>
    vid;

  struct init {
    std::vector<std::vector<std::size_t>> cid;
    std::vector<std::vector<std::size_t>> vid;
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

    template<index_space From>
    auto cells(flecsi::topo::id<From> from) const {
      return B::template entities<index_space::cells>(from);
    }

    auto vertices() const {
      return B::template entities<index_space::vertices>();
    }

    template<index_space From>
    auto vertices(flecsi::topo::id<From> from) const {
      return B::template entities<index_space::vertices>(from);
    }

  }; // struct interface

  /*--------------------------------------------------------------------------*
    Coloring
   *--------------------------------------------------------------------------*/

  static coloring
  color(std::string const & filename, flecsi::Color ncolors, init & fields) {
    using namespace flecsi;
    using namespace flecsi::topo::unstructured_impl;
    flog_assert(processes() == ncolors, "color to process mismatch");

    simple_definition sd(filename + "." + std::to_string(process()));
    fields.cid.push_back(std::move(sd.l2g_cells));
    fields.vid.push_back(std::move(sd.l2g_vertices));
    fields.c2v_connectivity.push_back(std::move(sd.c2v));
    return {// number of global colors
      ncolors,
      {// over index spaces
        {std::move(sd.cell_peers),
          std::move(sd.cell_partitions),
          sd.total_cells,
          {// over process colors
            {static_cast<util::id>(fields.cid[0].size()),
              std::move(sd.peer_cells),
              // cnx_allocs
              {0, fields.c2v_connectivity[0].values.size()}}},
          std::move(sd.cell_num_intervals)},
        {std::move(sd.vertex_peers),
          std::move(sd.vertex_partitions),
          sd.total_vertices,
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
    const std::vector<std::vector<std::size_t>> & cid,
    const std::vector<std::vector<std::size_t>> & vid) {
    const auto ma = m.accessors();
    auto acid = mcid.accessors();
    auto avid = mvid.accessors();
    for(unsigned int i = 0; i < m.depth(); ++i) {
      for(auto v : ma[i].vertices()) {
        avid[i][v] = vid[i][v];
      }
      for(auto c : ma[i].cells()) {
        acid[i][c] = cid[i][c];
      }
    }
  } // init_mesh_ids

  static void initialize(flecsi::data::topology_slot<fixed_mesh> & s,
    coloring const &,
    const init & fields) {
    using namespace flecsi;
    auto & c2v = s->get_connectivity<fixed_mesh::cells, fixed_mesh::vertices>();
    auto & v2c = s->get_connectivity<fixed_mesh::vertices, fixed_mesh::cells>();

    auto lm = data::launch::make(s);
    execute<topo::unstructured_impl::init_connectivity<privilege_count<cells>>,
      mpi>(c2v(lm), fields.c2v_connectivity);

    constexpr PrivilegeCount NPC = privilege_count<index_space::cells>;
    constexpr PrivilegeCount NPV = privilege_count<index_space::vertices>;
    execute<topo::unstructured_impl::transpose<NPC, NPV>>(c2v(s), v2c(s));

    execute<init_mesh_ids, mpi>(lm, cid(lm), vid(lm), fields.cid, fields.vid);
  } // initialize
}; // struct fixed_mesh

#endif
