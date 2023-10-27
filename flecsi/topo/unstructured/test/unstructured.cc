#include "unstructured.hh"
#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/topo/unstructured/coloring_utils.hh"
#include "flecsi/topo/unstructured/interface.hh"
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
      {
        unstructured::cslot coloring;
        unstructured::init fields;
        coloring.allocate(f, fields);
        mesh.allocate(coloring.get(), fields);
      }

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

void
print_mesh(unstructured::accessor<ro, ro, ro> m,
  field<util::gid>::accessor<ro, ro, ro> cgids,
  field<util::gid>::accessor<ro, ro, ro> vgids,
  field<unstructured::point>::accessor<ro, ro, ro> coords) {

  std::stringstream ss;

  ss << "COLOR  " << color() << "\n";
  ss << "#Cells = " << m.cells().size()
     << ", #Vertices = " << m.vertices().size() << "\n";
  ss << "\nVertex Info---->\n";
  for(auto v : m.vertices()) {
    ss << "Vertex [" << v << ", " << vgids[v] << "], coords = [" << coords[v][0]
       << ", " << coords[v][1] << "]\n";
  }

  ss << "\nCell Info---->\n";
  for(auto c : m.cells()) {
    ss << "Cell [" << c << ", " << cgids[c] << "], conn = [";
    for(auto v : m.vertices(c)) {
      ss << "  " << v;
    }
    ss << "]\n";
  }

  ss << "\nOwned Cells---->\n";
  for(auto c : m.cells<unstructured::owned>()) {
    ss << "Cell [" << c << ", " << cgids[c] << "]\n";
  }

  flog(info) << ss.str() << std::endl;
}

util::BBox<2>
bounding_boxes(unstructured::accessor<ro, ro, ro> mesh,
  field<unstructured::point>::accessor<ro, ro, ro> coords,
  bool is_src) {

  using point_t = util::point<double, 2>;
  using nl = std::numeric_limits<double>;

  auto lo_comp = [](const double & a, double & b) {
    if(a < b)
      b = a;
  };

  auto hi_comp = [](const double & a, double & b) {
    if(a > b)
      b = a;
  };

  point_t lo{nl::max(), nl::max()}, hi{nl::min(), nl::min()};

  if(is_src) {
    for(auto v : mesh.vertices()) {
      lo_comp(coords[v][0], lo[0]);
      lo_comp(coords[v][1], lo[1]);
      hi_comp(coords[v][0], hi[0]);
      hi_comp(coords[v][1], hi[1]);
    }
    return {lo, hi};
  }
  else {
    for(auto c : mesh.cells<unstructured::owned>()) {
      for(auto v : mesh.vertices(c)) {
        lo_comp(coords[v][0], lo[0]);
        lo_comp(coords[v][1], lo[1]);
        hi_comp(coords[v][0], hi[0]);
        hi_comp(coords[v][1], hi[1]);
      }
    }
    return {lo, hi};
  }
}

int
verify_overlap(data::multi<unstructured::accessor<ro, ro, ro>> src_meshes,
  data::multi<field<unstructured::point>::accessor<ro, ro, ro>> src_coords,
  unstructured::accessor<ro, ro, ro> trg_mesh,
  field<unstructured::point>::accessor<ro, ro, ro> trg_coords) {
  UNIT("VERIFY_OVERLAP") {
    std::vector<std::vector<Color>> ref_colors = {
      {0, 1}, {1, 3}, {0, 1, 2, 3}, {0, 2}};
    std::vector<Color> src_co;
    std::vector<util::BBox<2>> src_boxes;

    const auto co = src_coords.components();
    auto fs = co.begin();

    for(auto [clr, m] : src_meshes.components()) {
      src_co.emplace_back(clr);
      auto [clr2, coords] = *fs++;
      auto bb = bounding_boxes(m, coords, true);
      src_boxes.emplace_back(bb);
    }

    util::BBox<2> src_box = src_boxes[0];
    for(std::size_t k = 1; k < src_boxes.size(); ++k)
      src_box += src_boxes[k];

    auto trg_box = bounding_boxes(trg_mesh, trg_coords, false);

    auto c = color();
    EXPECT_EQ(ref_colors[c].size(), src_co.size());
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
    if(processes() == 4) {
      std::string src_file = "simple2d-8x8.msh";
      std::string trg_file = "simple2d-3x3.msh";
      unstructured::slot src_mesh, trg_mesh;

      flog(info) << "Loading source mesh: " << src_file << std::endl;
      {
        unstructured::cslot src_coloring;
        unstructured::init fields;
        src_coloring.allocate(src_file, fields);
        src_mesh.allocate(src_coloring.get(), fields);
        execute<print_mesh>(src_mesh,
          unstructured::cid(src_mesh),
          unstructured::vid(src_mesh),
          unstructured::coords(src_mesh));
      }

      flog(info) << "\n\nLoading target mesh: " << src_file << std::endl;
      {
        unstructured::cslot trg_coloring;
        unstructured::init fields;
        trg_coloring.allocate(trg_file, fields);
        trg_mesh.allocate(trg_coloring.get(), fields);
        execute<print_mesh>(trg_mesh,
          unstructured::cid(trg_mesh),
          unstructured::vid(trg_mesh),
          unstructured::coords(trg_mesh));
      }

      // Find bounding boxes of each mesh in this color
      auto src_box_ft =
        execute<bounding_boxes>(src_mesh, unstructured::coords(src_mesh), true);
      src_box_ft.wait();

      auto trg_box_ft = execute<bounding_boxes>(
        trg_mesh, unstructured::coords(trg_mesh), false);
      trg_box_ft.wait();

      auto src_nc = src_mesh.colors();
      auto trg_nc = trg_mesh.colors();

      std::vector<util::BBox<2>> src_boxes, trg_boxes;
      for(int c = 0; c < src_nc; ++c) {
        auto src_box = src_box_ft.get(c);
        src_boxes.push_back(src_box);
      }

      for(int c = 0; c < trg_nc; ++c) {
        auto trg_box = trg_box_ft.get(c);
        trg_boxes.push_back(trg_box);
      }

      // construct kdtree
      util::KDTree<2> src_tree(src_boxes);
      util::KDTree<2> trg_tree(trg_boxes);

      // search two kdtrees
      std::map<long, std::vector<long>> candidates_map;
      util::intersect<2>(src_tree, trg_tree, candidates_map);

      // launch maps
      data::launch::Claims cmap(trg_nc);
      int c = 0;
      for(auto & cm : candidates_map) {
        for(auto & s : cm.second)
          cmap[c].push_back(s);
        c++;
      }

      // verify overlap
      auto lm = data::launch::make(src_mesh, cmap);
      EXPECT_EQ(test<verify_overlap>(lm,
                  unstructured::coords(lm),
                  trg_mesh,
                  unstructured::coords(trg_mesh)),
        0);
    } // test only for 4 process
  };
} // geometric rendezvous driver

util::unit::driver<geomrz_driver> grz_driver;
