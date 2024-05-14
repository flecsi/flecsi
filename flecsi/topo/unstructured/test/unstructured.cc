#include "unstructured.hh"
#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/topo/unstructured/coloring_utils.hh"
#include "flecsi/topo/unstructured/interface.hh"
#include "flecsi/topo/unstructured/types.hh"
#include "flecsi/util/geometry/kdtree.hh"
#include "flecsi/util/unit.hh"

#include <iostream>
#include <limits>
#include <set>
#include <string>

using namespace flecsi;

int
verify_entities(unstructured::accessor<ro, ro, ro> m,
  std::string source_file,
  field<util::gid>::accessor<ro, ro, ro> cids,
  field<util::gid>::accessor<ro, ro, ro> vids) {
  UNIT("TASK") {
    std::string source_mesh = source_file.substr(0, source_file.size() - 4);
    std::string output_file = "unstructured_" + source_mesh + "_entities_" +
                              std::to_string(processes()) + "_" +
                              std::to_string(color()) + ".blessed";

    auto & out = UNIT_CAPTURE();
    out << "{\n";
    out << "\"process\": " << process() << ",\n";
    out << "\"color\": " << color() << ",\n";

    auto write_sequence = [&](std::string title, const auto & entities) {
      out << "\"" << title << "\": [";
      decltype(entities.size()) count = 0;
      for(auto c : entities) {
        out << c;
        if(count + 1 < entities.size())
          out << ",";
        if(count > 0 && (count % 32) == 0)
          out << "\n";
        ++count;
      }
      out << "]";
    };

    auto write_entities =
      [&](std::string title, const auto & entities, const auto & gids) {
        write_sequence(title + "_local", entities);
        out << ",\n";
        write_sequence(title + "_global",
          flecsi::util::transform_view(
            entities, [&](auto e) { return gids[e]; }));
      };

    auto write_exclusive_entities = [&](std::string title,
                                      const auto & owned,
                                      const auto & shared,
                                      const auto & gids) {
      const std::set<util::id> ss(shared.begin(), shared.end());
      std::vector<util::id> ex;
      for(auto o : owned)
        if(!ss.count(o))
          ex.push_back(o);
      write_entities(title, ex, gids);
    };

    write_exclusive_entities("exclusive_cells",
      m.cells<unstructured::owned>(),
      m.cells<unstructured::shared>(),
      cids);
    out << ",\n";
    write_entities("shared_cells", m.cells<unstructured::shared>(), cids);
    out << ",\n";
    write_entities("ghost_cells", m.cells<unstructured::ghost>(), cids);
    out << ",\n";
    write_exclusive_entities("exclusive_vertices",
      m.vertices<unstructured::owned>(),
      m.vertices<unstructured::shared>(),
      vids);
    out << ",\n";
    write_entities("shared_vertices", m.vertices<unstructured::shared>(), vids);
    out << ",\n";
    write_entities("ghost_vertices", m.vertices<unstructured::ghost>(), vids);
    out << "\n}\n";

    EXPECT_TRUE(UNIT_EQUAL_BLESSED(output_file));
  };
} // verify entities

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
verify_coords(unstructured::accessor<ro, ro, ro> m,
  field<unstructured::point>::accessor<ro, ro, ro> coords) {
  UNIT() { EXPECT_EQ(coords.span().size(), m.vertices().size()); };
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

void
mesh_setup(std::string file, unstructured::slot & mesh) {
  unstructured::cslot coloring;
  unstructured::init fields;
  coloring.allocate(file, fields);
  mesh.allocate(coloring.get(), fields);
}

field<int, data::ragged>::definition<unstructured, unstructured::cells> rcf;
field<int, data::ragged>::definition<unstructured, unstructured::vertices> rvf;

int
unstructured_driver() {
  std::vector<std::string> files = {"simple2d-8x8.msh",
    "disconnected.msh",
    "simple2d-3x3.msh",
    "single-cell.msh",
    "two-cell.msh"};
  UNIT() {
    for(auto f : files) {
      unstructured::slot mesh;
      flog(info) << "testing mesh: " << f << std::endl;
      mesh_setup(f, mesh);

      {
        EXPECT_EQ(test<verify_entities>(
                    mesh, f, unstructured::cid(mesh), unstructured::vid(mesh)),
          0);
      }

      {
        auto & tf = rcf(mesh).get_elements();
        tf.growth = {0, 0, 0.25, 0.5, 1};
        execute<allocate_field>(mesh, tf.sizes(), true);

        execute<init_rf>(mesh, unstructured::cid(mesh), rcf(mesh), true);
        EXPECT_EQ(
          test<verify_rf>(mesh, unstructured::cid(mesh), rcf(mesh), true), 0);
      } // scope

      {
        auto & tf = rvf(mesh).get_elements();
        tf.growth = {0, 0, 0.1, 0.5, 1};
        execute<allocate_field>(mesh, tf.sizes(), false);

        execute<init_rf>(mesh, unstructured::vid(mesh), rvf(mesh), false);
        execute<print_rf>(mesh, unstructured::vid(mesh), rvf(mesh), false);
        EXPECT_EQ(
          test<verify_rf>(mesh, unstructured::vid(mesh), rvf(mesh), false), 0);
      } // scope

      {
        EXPECT_EQ(test<verify_coords>(mesh, unstructured::coords(mesh)), 0);
      } // scope

    } // for
  };
} // unstructured_driver

util::unit::driver<unstructured_driver> driver;

// Geometric Rendezvous
int
verify_overlap(data::multi<unstructured::accessor<ro, ro, ro>> src_meshes,
  data::multi<field<unstructured::point>::accessor<ro, ro, ro>> src_coords,
  unstructured::accessor<ro, ro, ro> trg_mesh,
  field<unstructured::point>::accessor<ro, ro, ro> trg_coords) {
  UNIT("VERIFY_OVERLAP") {

    using ft =
      typename field<unstructured::point>::template accessor<ro, ro, ro>;
    using mt = typename unstructured::accessor<ro, ro, ro>;

    std::vector<std::vector<Color>> ref_colors = {
      {0, 1}, {1, 3}, {0, 1, 2, 3}, {0, 2}};
    std::vector<Color> src_co;

    const auto co = src_coords.components();
    auto fs = co.begin();
    auto src_box = util::BBox<2>::empty();

    for(auto [clr, m] : src_meshes.components()) {
      src_co.emplace_back(clr);
      auto [clr2, coords] = *fs++;
      src_box += unstructured::base::bounding_box<unstructured::cells,
        unstructured::owned,
        unstructured::vertices,
        mt,
        ft>(m, coords);
    }

    auto trg_box = unstructured::base::bounding_box<unstructured::cells,
      unstructured::owned,
      unstructured::vertices,
      mt,
      ft>(trg_mesh, trg_coords);

    auto c = color();
    ASSERT_EQ(ref_colors[c].size(), src_co.size());
    int i = 0;
    for(auto & sc : src_co)
      EXPECT_EQ(ref_colors[c][i++], sc);

    for(Dimension d = 0; d < 2; ++d) {
      EXPECT_TRUE((src_box.lower[d] <= trg_box.lower[d] &&
                   trg_box.upper[d] <= src_box.upper[d]));
    }
  };
}

int
geomrz_driver() {
  UNIT() {
    // Since ParMetis returns a different partition for the same number of
    // colors based on the number of ranks(np 1 vs np > 1), we skip the test for
    // one rank runs.
    if((processes() == 1) || (FLECSI_BACKEND == FLECSI_BACKEND_mpi))
      return;

    std::string src_file = "simple2d-8x8.msh";
    std::string trg_file = "simple2d-3x3.msh";
    unstructured::slot src_mesh, trg_mesh;

    flog(info) << "Loading source mesh: " << src_file << std::endl;
    mesh_setup(src_file, src_mesh);

    flog(info) << "\n\nLoading target mesh: " << src_file << std::endl;
    mesh_setup(trg_file, trg_mesh);

    using ft =
      typename field<unstructured::point>::template accessor<ro, ro, ro>;
    using mt = typename unstructured::accessor<ro, ro, ro>;
    // Find bounding boxes of each mesh
    auto get_boxes = [](unstructured::slot & mesh) {
      auto box_ft =
        execute<unstructured::base::bounding_box<unstructured::cells,
          unstructured::owned,
          unstructured::vertices,
          mt,
          ft>>(mesh, unstructured::coords(mesh));
      std::vector<util::BBox<2>> boxes;
      for(Color c = 0; c < box_ft.size(); ++c) {
        boxes.push_back(box_ft.get(c));
      }
      return boxes;
    };

    const auto candidates_map =
      util::KDTree<2>(get_boxes(trg_mesh)).intersect(get_boxes(src_mesh));

    // launch maps
    data::launch::Claims cmap(trg_mesh.colors());
    for(std::size_t c = 0; c < trg_mesh.colors(); ++c) {
      if(auto v = candidates_map.find(c); v != candidates_map.end()) {
        cmap[c].assign(v->second.begin(), v->second.end());
      }
    }

    // verify overlap
    auto lm = data::launch::make(src_mesh, cmap);
    EXPECT_EQ(
      test<verify_overlap>(
        lm, unstructured::coords(lm), trg_mesh, unstructured::coords(trg_mesh)),
      0);
  };
} // geometric rendezvous driver

util::unit::driver<geomrz_driver> grz_driver;
