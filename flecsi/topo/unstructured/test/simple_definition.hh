/*
    @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
   /@@/////  /@@          @@////@@ @@////// /@@
   /@@       /@@  @@@@@  @@    // /@@       /@@
   /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
   /@@////   /@@/@@@@@@@/@@       ////////@@/@@
   /@@       /@@/@@//// //@@    @@       /@@/@@
   /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
   //       ///  //////   //////  ////////  //

   Copyright (c) 2016, Triad National Security, LLC
   All rights reserved.
                                                                              */
#pragma once

#include "flecsi/flog.hh"
#include "flecsi/util/crs.hh"

#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace flecsi {
namespace topo {
namespace unstructured_impl {

class simple_definition
{
public:
  using point = std::array<double, 2>;
  static constexpr Dimension dimension() {
    return 2;
  }

  simple_definition(const char * filename) {
    file_.open(filename, std::ifstream::in);

    if(file_.good()) {
      std::string line;
      std::getline(file_, line);
      std::istringstream iss(line);

      // Read the number of vertices and cells
      iss >> num_vertices_ >> num_cells_;

      // Get the offset to the beginning of the vertices
      vertex_start_ = file_.tellg();

      for(size_t i(0); i < num_vertices_; ++i) {
        std::getline(file_, line);
      } // for

      cell_start_ = file_.tellg();
    }
    else {
      flog_fatal("failed opening " << filename);
    } // if

    // Go to the start of the cells.
    std::string line;
    file_.seekg(cell_start_);
    for(size_t l(0); l < num_cells_; ++l) {
      std::getline(file_, line);
      std::istringstream iss(line);
      e2v_.add_row(std::vector<size_t>(
        std::istream_iterator<size_t>(iss), std::istream_iterator<size_t>()));
    }

  } // simple_definition

  simple_definition(const simple_definition &) = delete;
  simple_definition & operator=(const simple_definition &) = delete;

  std::size_t num_entities(Dimension dimension) const {
    flog_assert(dimension == 0 || dimension == 2, "invalid dimension");
    return dimension == 0 ? num_vertices_ : num_cells_;
  }

  util::crs const & entities(Dimension from_dim, Dimension to_dim) const {
    flog_assert(from_dim == 2, "invalid dimension " << from_dim);
    flog_assert(to_dim == 0, "invalid dimension " << to_dim);
    return e2v_;
  }

  std::vector<size_t>
  entities(Dimension from_dim, Dimension to_dim, std::size_t entity_id) const {
    flog_assert(from_dim == 2, "invalid dimension " << from_dim);
    flog_assert(to_dim == 0, "invalid dimension " << to_dim);

    std::string line;
    std::vector<size_t> ids;
    size_t v0, v1, v2, v3;

    // Go to the start of the cells.
    file_.seekg(cell_start_);

    // Walk to the line with the requested id.
    for(size_t l(0); l < entity_id; ++l) {
      std::getline(file_, line);
    } // for

    // Get the line with the information for the requested id.
    std::getline(file_, line);
    std::istringstream iss(line);

    // Read the cell definition.
    iss >> v0 >> v1 >> v2 >> v3;

    ids.push_back(v0);
    ids.push_back(v1);
    ids.push_back(v2);
    ids.push_back(v3);

    return ids;
  } // vertices

  /*
    Return the vertex with the given id.

    @param id The vertex id.
   */

  point vertex(size_t id) const {
    std::string line;
    point v;

    // Go to the start of the vertices.
    file_.seekg(vertex_start_);

    // Walk to the line with the requested id.
    for(size_t l(0); l < id; ++l) {
      std::getline(file_, line);
    } // for

    // Get the line with the information for the requested id.
    std::getline(file_, line);
    std::istringstream iss(line);

    // Read the vertex coordinates.
    iss >> v[0] >> v[1];

    return v;
  } // vertex

private:
  mutable std::ifstream file_;
  util::crs e2v_;

  size_t num_vertices_;
  size_t num_cells_;

  mutable std::iostream::pos_type vertex_start_;
  mutable std::iostream::pos_type cell_start_;

}; // class simple_definition

} // namespace unstructured_impl
} // namespace topo
} // namespace flecsi
