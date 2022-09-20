// Copyright (c) 2016, Los Alamos National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_TIKZ_HH
#define FLECSI_UTIL_TIKZ_HH

#include "flecsi/flog.hh"
#include "flecsi/topo/unstructured/types.hh"
#include "flecsi/util/mpi.hh"
#include "flecsi/util/types.hh"

#include <fstream>
#include <sstream>
#include <tuple>
#include <vector>

namespace flecsi {
namespace util {
namespace tikz {

using palette_data_t = std::tuple<std::string, std::string, std::string>;

const std::vector<palette_data_t> palette = {
  std::make_tuple("blue", "blue!40!white", "blue!5!white"),
  std::make_tuple("green!60!black", "green!90!black", "green!10!white"),
  std::make_tuple("teal", "teal!40!white", "teal!5!white"),
  std::make_tuple("red", "red!40!white", "red!5!white"),
  std::make_tuple("violet", "violet!40!white", "violet!5!white"),
  std::make_tuple("cyan", "cyan!40!white", "cyan!5!white")};

inline void
write_node(std::ofstream & stream,
  double xoff,
  double yoff,
  std::size_t id,
  Color color,
  bool fill = false,
  bool shared = false) {
  stream << "\\node[";

  if(shared) {
    stream << std::get<1>(palette[color]);
  }
  else {
    stream << std::get<0>(palette[color]);
  } // if

  if(fill) {
    stream << ","
           << "fill=" << std::get<2>(palette[color]);
  } // if

  stream << "] at (" << xoff << ", " << yoff << ") {" << id << "};"
         << std::endl;
} // write_node

inline void
write_owned(std::size_t M,
  std::size_t N,
  std::unordered_map<Color, std::vector<std::size_t>> const & cell_colors,
  std::unordered_map<Color, std::vector<std::size_t>> const & vertex_colors) {
  std::stringstream filename;
  filename << "primary-" << M << "x" << N << ".tex";
  std::ofstream output(filename.str(), std::ofstream::out);

  // Document header and start
  output << "% Mesh visualization" << std::endl;
  output << "\\documentclass[tikz,border=7mm]{standalone}" << std::endl;
  output << std::endl;

  output << "\\begin{document}" << std::endl;
  output << std::endl;

  // Picture start
  output << "\\begin{tikzpicture}" << std::endl;
  output << std::endl;

  // Draw the grid
  output << "\\draw[step=1cm,black] (0, 0) grid (" << M << ", " << N << ");"
         << std::endl;

  for(auto [co, cells] : cell_colors) {
    const size_t round_robin(co % palette.size());

    for(auto id : cells) {
      write_node(
        output, id % N + 0.5, std::size_t(id / N) + 0.5, id, round_robin, true);
    } // for
  } // for

  for(auto [co, vertices] : vertex_colors) {
    const size_t round_robin(co % palette.size());

    for(auto id : vertices) {
      write_node(
        output, id % (N + 1), std::size_t(id / (N + 1)), id, round_robin, true);
    } // for
  } // for

  // Picture end
  output << "\\end{tikzpicture}" << std::endl;
  output << std::endl;

  // Document end
  output << "\\end{document}" << std::endl;
} // write_owned

static void
write_color(Color const color,
  size_t const M,
  size_t const N,
  std::set<std::size_t> const & exclusive_cells,
  std::set<std::size_t> const & shared_cells,
  std::map<std::size_t, std::size_t> const & ghost_cells,
  std::set<std::size_t> const & exclusive_vertices,
  std::set<std::size_t> const & shared_vertices,
  std::map<std::size_t, std::size_t> const & ghost_vertices) {
  const Color round_robin = color % palette.size();

  std::stringstream output;
  output << "color-" << color << "-" << M << "x" << N << ".tex";
  std::ofstream tex(output.str(), std::ofstream::out);

  // Document header and start
  tex << "% Mesh visualization" << std::endl;
  tex << "\\documentclass[tikz,border=7mm]{standalone}" << std::endl;
  tex << std::endl;

  tex << "\\begin{document}" << std::endl;
  tex << std::endl;

  // Picture start
  tex << "\\begin{tikzpicture}" << std::endl;
  tex << std::endl;

  tex << "\\draw[step=1cm,black] (0, 0) grid (" << M << ", " << N << ");"
      << std::endl;

  // Content
  size_t cell(0);
  for(size_t j(0); j < M; ++j) {
    double yoff(0.5 + j);
    for(size_t i(0); i < M; ++i) {
      double xoff(0.5 + i);

      auto const & gcell = ghost_cells.find(cell);

      // Cells
      if(exclusive_cells.count(cell)) {
        write_node(tex, xoff, yoff, cell, round_robin);
      }
      if(shared_cells.count(cell)) {
        write_node(tex, xoff, yoff, cell, round_robin, false, true);
      }
      if(gcell != ghost_cells.end()) {
        const Color off_round_robin = gcell->second % palette.size();
        write_node(tex, xoff, yoff, cell, off_round_robin, false, false);
      } // if

      ++cell;
    } // for
  } // for

  size_t vertex(0);
  for(size_t j(0); j < M + 1; ++j) {
    double yoff(j);
    for(size_t i(0); i < N + 1; ++i) {
      double xoff(i);

      auto gvertex = ghost_vertices.find(vertex);

      if(exclusive_vertices.count(vertex)) {
        write_node(tex, xoff, yoff, vertex, round_robin);
      }
      if(shared_vertices.count(vertex)) {
        write_node(tex, xoff, yoff, vertex, round_robin, false, true);
      }
      if(gvertex != ghost_vertices.end()) {
        const Color off_round_robin = gvertex->second % palette.size();
        write_node(tex, xoff, yoff, vertex, off_round_robin, false, false);
      } // if

      ++vertex;
    } // for
  } // for

  // Picture end
  tex << "\\end{tikzpicture}" << std::endl;
  tex << std::endl;

  // Document end
  tex << "\\end{document}" << std::endl;
} // write_color

using entity_vector = std::vector<topo::unstructured_impl::process_coloring>;

inline void
write_closure(std::size_t M,
  std::size_t N,
  entity_vector const & process_cells,
  entity_vector const & process_vertices,
  MPI_Comm comm) {

  // This is a little dumb: repack info for all_gatherv.
  std::vector<std::pair<Color, std::vector<std::size_t>>> local_cells;
  std::vector<std::pair<Color, std::vector<std::size_t>>> local_vertices;
  auto vb = process_vertices.begin();
  for(auto cells : process_cells) {
    auto & vertices = *vb++;

    local_cells.emplace_back(std::make_pair(cells.color, cells.coloring.owned));
    local_vertices.emplace_back(
      std::make_pair(vertices.color, vertices.coloring.owned));

    std::set<std::size_t> exclcells, shrdcells, exclvertices, shrdvertices;
    std::map<std::size_t, std::size_t> ghstcells, ghstvertices;

    for(auto c : cells.coloring.exclusive) {
      exclcells.insert(c);
    }
    for(auto c : cells.coloring.shared) {
      shrdcells.insert(c.id);
    }
    for(auto c : cells.coloring.ghost) {
      ghstcells.emplace(c.id, c.global);
    }

    for(auto c : vertices.coloring.exclusive) {
      exclvertices.insert(c);
    }
    for(auto c : vertices.coloring.shared) {
      shrdvertices.insert(c.id);
    }
    for(auto c : vertices.coloring.ghost) {
      ghstvertices.emplace(c.id, c.global);
    }

    write_color(cells.color,
      M,
      N,
      exclcells,
      shrdcells,
      ghstcells,
      exclvertices,
      shrdvertices,
      ghstvertices);
  } // for

  auto all_cells = util::mpi::all_gatherv(local_cells, comm);
  auto all_vertices = util::mpi::all_gatherv(local_vertices, comm);

  int rank;
  MPI_Comm_rank(comm, &rank);

  if(rank == 0) {
    std::unordered_map<Color, std::vector<std::size_t>> cell_colors;
    for(auto vp : all_cells) {
      for(auto [co, cells] : vp) {
        cell_colors[co] = cells;
      } // for
    } // for

    std::unordered_map<Color, std::vector<std::size_t>> vertex_colors;
    for(auto vp : all_vertices) {
      for(auto [co, vertices] : vp) {
        vertex_colors[co] = vertices;
      } // for
    } // for

    write_owned(M, N, cell_colors, vertex_colors);
  } // if
} // write_closure

} // namespace tikz
} // namespace util
} // namespace flecsi

#endif
