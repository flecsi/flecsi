#include "unstructured.hh"
#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/topo/unstructured/coloring_utils.hh"
#include "flecsi/topo/unstructured/interface.hh"
#include "flecsi/util/unit.hh"

#include <optional>

using namespace flecsi;

void
init_rf(unstructured::accessor<ro, ro, ro> m,
  field<util::gid>::accessor<ro, ro, ro> vids,
  field<int, data::ragged>::mutator<wo, wo, na> tf,
  bool is_cell) {
  int sz = 3;
  if(is_cell) {
    for(auto c : m.cells<unstructured::owned>()) {
      tf[c].resize(sz);
      for(int i = 0; i < sz; ++i)
        tf[c][i] = (int)(vids[c] * 10000 + i);
    }
  }
  else {
    for(auto v : m.vertices<unstructured::owned>()) {
      tf[v].resize(sz);
      for(int i = 0; i < sz; ++i)
        tf[v][i] = (int)(vids[v] * 10000 + i);
    }
  }
} // init_rf

void
print_rf(unstructured::accessor<ro, ro, ro> m,
  field<util::gid>::accessor<ro, ro, ro> gids,
  field<int, data::ragged>::accessor<ro, ro, ro> tf,
  bool is_cell) {

  std::stringstream ss;
  ss << " Color " << color() << std::endl;
  if(is_cell) {
    ss << " Number of cells = " << m.cells().size() << "\n";
    for(auto c : m.cells()) {
      ss << "For cell (" << c << ", " << gids[c]
         << "), field_size = " << tf[c].size() << ", field vals = [ ";
      for(std::size_t i = 0; i < tf[c].size(); ++i)
        ss << tf[c][i] << "  ";
      ss << "]\n\n";
    }
  }
  else {
    ss << " Number of vertices = " << m.vertices().size() << "\n";
    for(auto v : m.vertices()) {
      ss << "For vertex (" << v << ", " << gids[v]
         << "), field_size = " << tf[v].size() << ", field vals = [ ";
      for(std::size_t i = 0; i < tf[v].size(); ++i)
        ss << tf[v][i] << "  ";
      ss << "]\n\n";
    }
  }

  flog(info) << ss.str() << std::endl;
} // print_rf

void
allocate_field(unstructured::accessor<ro, ro, ro> m,
  topo::resize::Field::accessor<wo> a,
  bool is_cell) {
  int sz = 3;
  if(is_cell)
    a = m.cells().size() * sz;
  else
    a = m.vertices().size() * sz;
}

int
verify_rf(unstructured::accessor<ro, ro, ro> m,
  field<util::gid>::accessor<ro, ro, ro> vids,
  field<int, data::ragged>::accessor<ro, ro, ro> tf,
  bool is_cell) {

  UNIT("VERIFY_FIELD") {
    int sz = 3;
    if(is_cell) {
      for(auto c : m.cells()) {
        EXPECT_EQ(tf[c].size(), sz);
        for(int i = 0; i < sz; ++i)
          EXPECT_EQ(tf[c][i], (int)(vids[c] * 10000 + i));
      }
    }
    else {
      for(auto v : m.vertices()) {
        EXPECT_EQ(tf[v].size(), sz);
        for(int i = 0; i < sz; ++i)
          EXPECT_EQ(tf[v][i], (int)(vids[v] * 10000 + i));
      }
    }
  };
}

unstructured::slot mesh, m1, m2;
unstructured::cslot coloring, c1, c2;
field<int, data::ragged>::definition<unstructured, unstructured::cells> rcf;
field<int, data::ragged>::definition<unstructured, unstructured::vertices> rvf;

int
unstructured_driver() {
  std::vector<std::string> files = {
    "simple2d-16x16.msh", "simple2d-8x8.msh", "disconnected.msh"};
  UNIT() {
    for(auto f : files) {
      flog(info) << "testing mesh: " << f << std::endl;
      coloring.allocate(f);
      mesh.allocate(coloring.get());

      {
        auto & tf = rcf(mesh).get_ragged();
        tf.growth = {0, 0, 0.25, 0.5, 1};
        execute<allocate_field>(mesh, tf.sizes(), true);
        tf.resize();

        auto const & cids = mesh->forward_map<unstructured::cells>();
        execute<init_rf>(mesh, cids(mesh), rcf(mesh), true);
        EXPECT_EQ(test<verify_rf>(mesh, cids(mesh), rcf(mesh), true), 0);
      } // scope

      {
        auto & tf = rvf(mesh).get_ragged();
        tf.growth = {0, 0, 0.25, 0.5, 1};
        execute<allocate_field>(mesh, tf.sizes(), false);
        tf.resize();

        auto const & vids = mesh->forward_map<unstructured::vertices>();
        execute<init_rf>(mesh, vids(mesh), rvf(mesh), false);
        execute<print_rf>(mesh, vids(mesh), rvf(mesh), false);
        EXPECT_EQ(test<verify_rf>(mesh, vids(mesh), rvf(mesh), false), 0);
      } // scope
    } // for
  };
} // unstructured_driver

flecsi::unit::driver<unstructured_driver> driver;
