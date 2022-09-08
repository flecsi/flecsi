// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#include "unstructured.hh"
#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/topo/unstructured/coloring_utils.hh"
#include "flecsi/topo/unstructured/interface.hh"
#include "flecsi/util/unit.hh"

#include <optional>

using namespace flecsi;

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
  field<int, data::ragged>::mutator<wo, wo, na> tf,
  bool is_cell) {
  int sz = 100;
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

} // init_field

void
print_field(unstructured::accessor<ro, ro, ro> m,
  field<int, data::ragged>::accessor<ro, ro, ro> tf,
  bool is_cell) {

  std::stringstream ss;
  if(is_cell) {
    ss << " Number of cells = " << m.cells().size() << "\n";
    for(auto c : m.cells()) {
      ss << "For cell " << c << ", field_size = " << tf[c].size()
         << ", field vals = [ ";
      for(std::size_t i = 0; i < tf[c].size(); ++i)
        ss << tf[c][i] << "  ";
      ss << "]\n\n";
    }
  }
  else {
    ss << " Number of vertices = " << m.vertices().size() << "\n";
    for(auto v : m.vertices()) {
      ss << "For vertex " << v << ", field_size = " << tf[v].size()
         << ", field vals = [ ";
      for(std::size_t i = 0; i < tf[v].size(); ++i)
        ss << tf[v][i] << "  ";
      ss << "]\n\n";
    }
  }

  flog(info) << ss.str() << std::endl;
} // print_field

void
allocate_field(unstructured::accessor<ro, ro, ro> m,
  topo::resize::Field::accessor<wo> a,
  bool is_cell) {
  int sz = 100;
  if(is_cell)
    a = m.cells().size() * sz;
  else
    a = m.vertices().size() * sz;
}

int
verify_field(unstructured::accessor<ro, ro, ro> m,
  field<util::id>::accessor<ro, ro, ro> vids,
  field<int, data::ragged>::accessor<ro, ro, ro> tf,
  bool is_cell) {

  UNIT("VERIFY_FIELD") {
    int sz = 100;
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
field<int, data::ragged>::definition<unstructured, unstructured::cells>
  cellfield;

int
unstructured_driver() {
  std::vector<std::string> files = {
    "simple2d-16x16.msh", "simple2d-8x8.msh", "disconnected.msh"};
  UNIT() {
    for(auto f : files) {
      flog(info) << "testing mesh: " << f << std::endl;
      coloring.allocate(f);
      mesh.allocate(coloring.get());

      auto & tf = cellfield(mesh).get_ragged();
      tf.growth = {0, 0, 0.25, 0.5, 1};
      execute<allocate_field>(mesh, tf.sizes(), true);
      tf.resize();

      auto const & cids = mesh->forward_map<unstructured::cells>();
      execute<init_field>(mesh, cids(mesh), cellfield(mesh), true);
      EXPECT_EQ(test<verify_field>(mesh, cids(mesh), cellfield(mesh), true), 0);

#if 0
      auto const & vids = mesh->forward_map<unstructured::vertices>();
      auto const & eids = mesh->forward_map<unstructured::edges>();
      execute<print>(mesh, cids(mesh), vids(mesh), eids(mesh));
      execute<print_field>(mesh, cellfield(mesh), true);
#endif
    } // for
  };
} // unstructured_driver

flecsi::unit::driver<unstructured_driver> driver;
