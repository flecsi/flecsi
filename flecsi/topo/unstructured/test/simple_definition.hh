#ifndef FLECSI_TOPO_UNSTRUCTURED_TEST_SIMPLE_DEFINITION_HH
#define FLECSI_TOPO_UNSTRUCTURED_TEST_SIMPLE_DEFINITION_HH

#include "flecsi/flog.hh"
#include "flecsi/topo/unstructured/types.hh"
#include "flecsi/util/common.hh"
#include "flecsi/util/crs.hh"

#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace flecsi {
namespace topo {
namespace unstructured_impl {

class simple_definition {
  template<typename T>
  T read(std::ifstream & in) {
    T val;
    if(!(in >> val))
      flog_fatal("parsing error: expected a value");
    return val;
  }

  void
  expect_string(std::ifstream & in, std::string s, bool consume_line = false) {
    auto val = read<std::string>(in);
    if(val != s)
      flog_fatal("parsing error: expected " + s + ", got " + val);
    if(consume_line) {
      in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
  }

  util::gid read_kv(std::ifstream & in, std::string key) {
    expect_string(in, key);
    return read<util::gid>(in);
  }

  template<typename T>
  std::vector<T> read_values(std::ifstream & in, const char * key = nullptr) {
    if(key)
      expect_string(in, key);
    std::string line;
    std::getline(in, line);
    std::istringstream iss(line);
    return std::vector<T>(
      std::istream_iterator<T>(iss), std::istream_iterator<T>());
  }

public:
  std::vector<std::size_t> color_peers;
  std::vector<std::vector<Color>> cell_peers;
  std::vector<std::size_t> cell_partitions;
  std::vector<std::size_t> cell_num_intervals;

  std::vector<std::vector<Color>> vertex_peers;
  std::vector<std::size_t> vertex_partitions;
  std::vector<std::size_t> vertex_num_intervals;

  std::vector<flecsi::util::gid> l2g_vertices;
  std::vector<flecsi::util::gid> l2g_cells;
  util::crs c2v;
  std::map<Color, topo::unstructured_impl::peer_entities> peer_vertices;
  std::map<Color, topo::unstructured_impl::peer_entities> peer_cells;

  simple_definition(const std::string filename) {
    std::ifstream in(filename);

    Color colors = read_kv(in, "colors");

    color_peers = read_values<std::size_t>(in, "color_peers");
    expect_string(in, "cell_peers", true);
    for(Color c = 0; c < colors; c++) {
      cell_peers.push_back(read_values<Color>(in));
    }
    cell_partitions = read_values<std::size_t>(in, "cell_partitions");
    cell_num_intervals = read_values<std::size_t>(in, "cell_num_intervals");

    expect_string(in, "vertex_peers", true);
    for(Color c = 0; c < colors; c++) {
      vertex_peers.push_back(read_values<Color>(in));
    }
    vertex_partitions = read_values<std::size_t>(in, "vertex_partitions");
    vertex_num_intervals = read_values<std::size_t>(in, "vertex_num_intervals");

    const util::gid nvertices = read_kv(in, "nvertices"),
                    ncells = read_kv(in, "ncells"),
                    shared_vertices = read_kv(in, "shared_vertices"),
                    shared_cells = read_kv(in, "shared_cells"),
                    ghost_vertices = read_kv(in, "ghost_vertices"),
                    ghost_cells = read_kv(in, "ghost_cells");

    l2g_vertices = read_values<flecsi::util::gid>(in, "vertices");
    if(l2g_vertices.size() != nvertices)
      flog_fatal("parse error: wrong number of vertices");

    expect_string(in, "cells");
    std::string line;
    for(util::gid c = 0; c < ncells; ++c) {
      auto global_id = read<flecsi::util::gid>(in);
      l2g_cells.push_back(global_id);
      std::getline(in, line);
      std::istringstream iss(line);
      c2v.add_row(std::vector<flecsi::util::gid>(
        std::istream_iterator<flecsi::util::gid>(iss),
        std::istream_iterator<flecsi::util::gid>()));
    }

    expect_string(in, "shared_vertices");
    for(util::gid v = 0; v < shared_vertices; ++v) {
      auto local_id = read<flecsi::util::id>(in);
      auto color = read<Color>(in);
      peer_vertices[color].shared.insert(local_id);
    }

    expect_string(in, "shared_cells");
    for(util::gid c = 0; c < shared_cells; ++c) {
      auto local_id = read<flecsi::util::id>(in);
      auto color = read<Color>(in);
      peer_cells[color].shared.insert(local_id);
    }

    expect_string(in, "ghost_vertices");
    for(util::gid v = 0; v < ghost_vertices; ++v) {
      auto local_id = read<flecsi::util::id>(in);
      auto remote_id = read<flecsi::util::id>(in);
      auto color = read<Color>(in);
      peer_vertices[color].ghost.insert({remote_id, local_id});
    }

    expect_string(in, "ghost_cells");
    for(util::gid c = 0; c < ghost_cells; ++c) {
      auto local_id = read<flecsi::util::id>(in);
      auto remote_id = read<flecsi::util::id>(in);
      auto color = read<Color>(in);
      peer_cells[color].ghost.emplace(remote_id, local_id);
    }
  }
};

} // namespace unstructured_impl
} // namespace topo
} // namespace flecsi

#endif
