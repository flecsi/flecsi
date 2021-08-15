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

/*! @file */

#include "flecsi/flog.hh"
#include "flecsi/topo/narray/types.hh"
#include "flecsi/util/color_map.hh"
#include "flecsi/util/common.hh"

#include <functional>
#include <optional>
#include <vector>

namespace flecsi {
namespace topo {
namespace narray_utils {

inline std::vector<std::size_t>
factor(std::size_t np) {
  std::vector<std::size_t> facs;
  const auto test = [&](std::size_t f) {
    while(!(np % f)) {
      facs.push_back(f);
      np /= f;
    }
  };

  // Trivial wheel factorization:
  test(2);
  test(3);
  for(std::size_t i = 5, step = 2; i * i <= np; i += step, step = 2 * 3 - step)
    test(i);
  if(np > 1) // i.e., the largest prime factor is >3 and unique
    facs.push_back(np);
  std::reverse(facs.begin(), facs.end());

  return facs;
} // factor

/*
  Create an axial color distribution for the given number of processes.
 */

inline narray_impl::colors
distribute(Color np, std::vector<std::size_t> indices) {
  narray_impl::colors parts(indices.size(), 1);

  auto facs = factor(np);

  // greedy decomposition
  for(auto fac : facs) {
    auto maxind = std::distance(
      indices.begin(), std::max_element(indices.begin(), indices.end()));
    parts[maxind] *= fac;
    indices[maxind] /= fac;
  }

  return parts;
} // decomp

/*
  Using the index colors for the given color, this function determines the
  position of the local partition within the color space. The "orientation"
  variable is a bit-array that stores (for each axis) boolean values
  for "low", and "high". If an axis is neither low nor high, it is
  interior. The enumeration defining these masks is in types.hh.
 */

inline auto
orientation(Dimension dimension,
  const narray_impl::colors & color_indices,
  const narray_impl::colors & axis_colors) {
  using namespace narray_impl;

#define FACE(e, c, c0, c1, c2)                                                 \
  ((((e) == 0) && ((e) == ((c)-1)))                                            \
      ? c1 | c2                                                                \
      : ((0 < (e) && e < ((c)-1)) ? c0 : ((e) == 0 ? c1 : c2)))

  std::uint32_t o = interior;
  std::uint32_t shft{low};
  for(Dimension axis = 0; axis < dimension; ++axis) {
    o |=
      FACE(color_indices[axis], axis_colors[axis], interior, shft, shft << 1);
    shft <<= 2;
  } // for
#undef FACE

  return o;
} // orientation

/*
  Create an index coloring for the given color (as defined by color_indices).
 */

inline auto
make_color(Dimension dimension,
  const narray_impl::colors & color_indices,
  std::vector<util::color_map> const & axcm,
  uint32_t orient,
  narray_impl::coord const & hdepths,
  narray_impl::coord const & bdepths,
  std::vector<bool> periodic) {
  using namespace narray_impl;

  process_color idxco;
  idxco.orientation = orient;
  idxco.global.resize(dimension);
  idxco.extents.resize(dimension);
  idxco.offset.resize(dimension);
  idxco.logical[0].resize(dimension);
  idxco.logical[1].resize(dimension);
  idxco.extended[0] = idxco.logical[0];
  idxco.extended[1] = idxco.logical[1];

  std::vector</* over axes */
    std::vector</* over intervals */
      std::pair<std::size_t, /* owner color */
        std::pair<std::size_t, std::size_t> /* interval */
        >>>
    ghstitvls(dimension);

  std::size_t extents{1};
  for(Dimension axis = 0; axis < dimension; ++axis) {
    auto axis_color = color_indices[axis];
    const Color axis_colors = axcm[axis].colors();
    idxco.global[axis] = axcm[axis].indices();
    idxco.extents[axis] = axcm[axis].indices(axis_color, 0);
    idxco.offset[axis] = axcm[axis].index_offset(color_indices[axis], 0);
    idxco.logical[0][axis] = 0;
    idxco.logical[1][axis] = axcm[axis].indices(axis_color, 0);

    std::uint32_t bits = orient >> axis * 2;

    if(bits & low && bits & high) {
      /*
        This is a degenerate dimension, i.e., it is flat with a single
        color layer. Therefore, we do not add halo extensions.
       */

      idxco.extents[axis] += 2 * bdepths[axis];

      idxco.logical[0][axis] += bdepths[axis];
      idxco.logical[1][axis] += bdepths[axis];

      idxco.extended[0][axis] = 0;
      idxco.extended[1][axis] = idxco.logical[1][axis] + bdepths[axis];
    }
    else if(bits & low) {
      /*
        This dimension is a low edge.
       */

      idxco.extents[axis] += bdepths[axis] + hdepths[axis];

      // Shift the logical hypercube by boundary depth
      idxco.logical[0][axis] += bdepths[axis];
      idxco.logical[1][axis] += bdepths[axis];

      idxco.extended[0][axis] = 0;
      idxco.extended[1][axis] = idxco.logical[1][axis];

      ghstitvls[axis].push_back({color_indices[axis] + 1,
        {idxco.logical[1][axis], idxco.logical[1][axis] + hdepths[axis]}});

      if(periodic[axis]) {
        flog_assert(bdepths[axis] > 0,
          "periodic boundaries require non-zero boundary depth");

        ghstitvls[axis].push_back({axis_colors - 1,
          {idxco.extended[0][axis], idxco.extended[0][axis] + bdepths[axis]}});
      } // if
    }
    else if(bits & high) {
      /*
        This dimension is a high edge.
       */

      idxco.extents[axis] += hdepths[axis] + bdepths[axis];

      idxco.offset[axis] -= hdepths[axis];

      idxco.logical[0][axis] += hdepths[axis];
      idxco.logical[1][axis] += hdepths[axis];

      idxco.extended[0][axis] = idxco.logical[0][axis];
      idxco.extended[1][axis] = idxco.logical[1][axis] + bdepths[axis];

      ghstitvls[axis].push_back({color_indices[axis] - 1,
        {idxco.logical[0][axis] - hdepths[axis], idxco.logical[0][axis]}});

      if(periodic[axis]) {
        flog_assert(bdepths[axis] > 0,
          "periodic boundaries require non-zero boundary depth");

        ghstitvls[axis].push_back({0,
          {idxco.logical[1][axis], idxco.logical[1][axis] + bdepths[axis]}});
      } // if
    }
    else {
      /*
        This dimension is interior.
       */

      idxco.extents[axis] += 2 * hdepths[axis];

      idxco.offset[axis] -= hdepths[axis];

      idxco.logical[0][axis] += hdepths[axis];
      idxco.logical[1][axis] += hdepths[axis];

      idxco.extended[0][axis] = idxco.logical[0][axis];
      idxco.extended[1][axis] = idxco.logical[1][axis];

      ghstitvls[axis].push_back({color_indices[axis] + 1,
        {idxco.logical[1][axis], idxco.logical[1][axis] + hdepths[axis]}});
      ghstitvls[axis].push_back({color_indices[axis] - 1,
        {idxco.logical[0][axis] - hdepths[axis], idxco.logical[0][axis]}});
    } // if

    extents *= idxco.extents[axis];
  } // for

  return std::make_tuple(idxco, ghstitvls, extents);
} // make_color

/*
  Generate a coloring for the provided coloring definition.
 */

inline auto
color(narray_impl::coloring_definition const & cd,
  MPI_Comm comm = MPI_COMM_WORLD) {
  using namespace narray_impl;

  const Dimension dimension = cd.axis_colors.size();

  flog_assert(dimension < 17,
    "current implementation is limited to 16 dimensions (uint32_t)");

  flog_assert(cd.axis_colors.size() == cd.axis_extents.size(),
    "argument mismatch: sizes(" << cd.axis_colors.size() << "vs. "
                                << cd.axis_extents.size()
                                << ") must be consistent");
  flog_assert(cd.axis_colors.size() == dimension,
    "size must match the intended dimension(" << dimension << ")");

  auto [rank, size] = util::mpi::info(comm);

  /*
    Create a color map for each dimension. Because we are using a
    tensor-product strategy below, we can use each map to define the
    sub colors and offsets for an individual axis.
   */

  Color nc = 1;
  std::size_t indices{1};
  std::vector<util::color_map> axcm;
  for(Dimension d = 0; d < dimension; ++d) {
    nc *= cd.axis_colors[d];
    indices *= cd.axis_extents[d];
    axcm.emplace_back(cd.axis_colors[d], cd.axis_colors[d], cd.axis_extents[d]);
  } // for

  /*
    Create a color map for the total number of colors (product of axis
    colors) to the number of processes.
   */

  util::color_map cm(size, nc, indices);

  /*
    Create a coloring for each color on this process.
   */

  std::vector<process_color> coloring;
  std::vector<std::size_t> partitions;
  for(Color c = 0; c < cm.colors(rank); ++c) {
    /*
      Convenience functions to map between colors and indices.
     */

    const auto co2idx = [](Color co, const colors & szs) {
      colors indices;
      for(auto sz : szs) {
        indices.emplace_back(co % sz);
        co /= sz;
      }
      return indices;
    };

    const auto idx2co = [](const auto & idx, const auto & szs) {
      auto pr = szs[0];
      decltype(pr) co = 0;
      for(std::size_t i{0}; i < idx.size() - 1; ++i) {
        co += idx[i + 1] * pr;
        pr *= szs[i + 1];
      }
      return co + idx[0];
    };

    /*
      Get the indices representation of our color.
     */

    auto color_indices = co2idx(cm.color_id(rank, c), cd.axis_colors);

    /*
      Find our orientation within the color space.
     */

    uint32_t orient = orientation(dimension, color_indices, cd.axis_colors);

    /*
      Make the coloring information for our color.
     */

    auto [idxco, ghstitvls, extents] = make_color(dimension,
      color_indices,
      axcm,
      orient,
      cd.axis_hdepths,
      cd.axis_bdepths,
      cd.axis_periodic);

    /*
      Save extents information for creating partitions.
     */

    partitions.emplace_back(extents);

    // This won't be necessary with c++20. In c++17 lambdas can't
    // capture structured bindings.
    auto & idx_coloring = idxco;
    auto & ghost_intervals = ghstitvls;

    /*
      Here, we compose the intervals from each sub-dimension to form
      the actual full-dimensional subregions. These define the coloring.
     */

    bool const diagonals = cd.diagonals;
    std::function<std::vector<std::pair<colors, hypercube>>(
      Dimension, Dimension)>
      expand = [&idx_coloring, &color_indices, &ghost_intervals, diagonals](
                 Dimension dim, Dimension top) {
        std::vector<std::pair<colors, hypercube>> sregs;

        for(Dimension axis = 0; axis < dim; ++axis) {
          std::optional<std::vector<std::pair<colors, hypercube>>> ssubs;

          if(sregs.size()) {
            /*
              Save a copy of the lower-dimensional subregions to
              create diagonal entries.
             */

            if(diagonals) {
              ssubs.emplace(sregs);
            } // if

            /*
              Expand the subregions from the lower dimensions.
             */

            auto subs = std::move(sregs);
            sregs.clear();
            for(size_t off{idx_coloring.logical[0][axis]};
                off < idx_coloring.logical[1][axis];
                ++off) {
              for(auto & s : subs) {
                s.first[axis] = color_indices[axis];
                s.second[0][axis] = off;
                s.second[1][axis] = off + 1;
                sregs.emplace_back(s);
              } // for
            } // for
          } // if

          /*
            Add the subregions for this dimension.
           */

          for(auto i : ghost_intervals[axis]) {
            colors co(top, 0);
            coord start(top, 0);
            coord end(top, 0);

            /*
              For dimensions lower than the current axis, just populate
              the axes with the logical start and end because they are
              not the ghost part of the subregion, but they should
              cover the local axis extents. These use the local color
              indices.
             */

            for(Dimension a = 0; a < axis; ++a) {
              co[a] = color_indices[a];
              start[a] = idx_coloring.logical[0][a];
              end[a] = idx_coloring.logical[1][a];
            } // for
            /*
              Add the ghost part that comes from the current axis.
             */

            co[axis] = i.first;
            start[axis] = i.second.first;
            end[axis] = i.second.second;
            sregs.push_back({co, hypercube{start, end}});

            if(ssubs.has_value()) {
              /*
                Add axis information from this dimension to new diagonals.
               */

              for(auto ss : *ssubs) {
                ss.first[axis] = i.first;
                ss.second[0][axis] = i.second.first;
                ss.second[1][axis] = i.second.second;
                sregs.emplace_back(ss);
              } // for
            } // if
          } // for
        } // for

        return sregs;
      };

    auto subregions = expand(dimension, dimension);

    // Remove duplicate subregions
    util::force_unique(subregions);

    /*
      Compute a remote index from a global coordinate.
     */

    auto const & axis_bdepths = cd.axis_bdepths;
    auto rmtidx = [dimension, &axis_bdepths](
                    process_color const & idxco, coord const & gidx) {
      coord result(dimension);
      for(Dimension axis = 0; axis < dimension; ++axis) {
        uint32_t bits = idxco.orientation >> axis * 2;
        if(bits & low) {
          result[axis] = gidx[axis] + axis_bdepths[axis];
        }
        else {
          result[axis] = gidx[axis] - idxco.offset[axis];
        }
      }
      return result;
    };

    /*
      Map a local coordinate to a global one.
     */

    auto l2g = [dimension, orient, &axis_bdepths](
                 process_color const & idxco, coord const & idx) {
      coord result(dimension);
      for(Dimension axis = 0; axis < dimension; ++axis) {
        uint32_t bits = orient >> axis * 2;
        if(bits & low) {
          if(idx[axis] < idxco.logical[0][axis]) {
            /* periodic low */
            result[axis] = idxco.global[axis] - axis_bdepths[axis] + idx[axis];
          }
          else {
            result[axis] = idx[axis] - axis_bdepths[axis];
          }
        }
        else if(bits & high && idx[axis] >= idxco.logical[1][axis]) {
          /* periodic high */
          result[axis] = idx[axis] - idxco.logical[1][axis];
        }
        else {
          result[axis] = idxco.offset[axis] + idx[axis];
        }
      }
      return result;
    };

    /*
      The intervals computed in the tensor product strategy above are
      closed on the start of the interval, and open on the end. This
      function is used below to close the end, so that the interval
      can be converted into a memory offset interval.
     */

    auto op2cls = [dimension](coord const & idx) {
      coord result(dimension);
      for(Dimension axis = 0; axis < dimension; ++axis) {
        result[axis] = idx[axis] - 1;
      }
      return result;
    };

    /*
      Loop through the subregions and create the actual coloring.
     */

    std::unordered_map<Color, process_color> idxmap;
    for(auto s : subregions) {
      auto co = idx2co(s.first, cd.axis_colors);
      if(idxmap.find(co) == idxmap.end()) {
        /*
          Create basic coloring information for the owning color, so
          that we can determine the remote offsets for our points.
          The coloring informaiton is stored for subsequent use.
         */

        auto [ridxco, rghstitvls, extents] = make_color(dimension,
          s.first,
          axcm,
          orientation(dimension, s.first, cd.axis_colors),
          cd.axis_hdepths,
          cd.axis_bdepths,
          cd.axis_periodic);
        idxmap[co] = ridxco;
      } // if

      /*
        The subregions are defined by hypercubes. These must be broken
        up into contiguous intervals. This lambda recurses each subregion
        to break up the volume into contiguous chunks.
       */

      std::function<void(Color, hypercube, Dimension)> make =
        [&idx_coloring, &idxmap, &op2cls, &idx2co, &l2g, &rmtidx, &make](
          Color clr, hypercube const & subregion, Dimension axis) {
          if(axis == 0) {
            // Compute the local memory interval.
            auto const start = idx2co(subregion[0], idx_coloring.extents);
            auto const end = idx2co(op2cls(subregion[1]), idx_coloring.extents);

            // The output intervals are closed on the start
            // and open on the end, i.e., [start, end)

            idx_coloring.intervals.push_back({start, end + 1});

            /*
              Loop through the local interval sizes, and add the remote
              pointer offsets.
             */

            auto const gidx = l2g(idx_coloring, subregion[0]);
            auto const ridx = rmtidx(idxmap.at(clr), gidx);
            auto rmtoff = idx2co(ridx, idxmap.at(clr).extents);

            for(std::size_t off{0}; off < (end + 1) - start; ++off) {
              idx_coloring.points[clr].push_back({start + off, rmtoff + off});
            } // for
          }
          else {
            // Recurse ranges at this axis to create contiguous intervals.
            for(std::size_t r = subregion[0][axis]; r < subregion[1][axis];
                ++r) {
              hypercube rct = subregion;
              rct[0][axis] = r;
              rct[1][axis] = r + 1;

              make(clr, rct, axis - 1);
            } // for
          } // if
        };

      if(cd.create_plan) {
        make(co, s.second, dimension - 1);
      } // if
    } // for

    coloring.emplace_back(std::move(idx_coloring));
  } // for

  {
    auto pgthr = util::mpi::all_gatherv(partitions, comm);

    partitions.clear();
    partitions.resize(nc);
    std::size_t p{0};
    for(auto vp : pgthr) {
      std::size_t c{0};
      for(auto v : vp) {
        partitions[cm.color_id(p, c)] = v;
        ++c;
      } // for
      ++p;
    } // for
  }

  return std::make_tuple(nc, indices, coloring, partitions);
} // color

/*!
  Generate an auxiliary coloring from the given primary coloring. This is
  similar to the strategy used by topo::unstructured.

  @param idx         The primary coloring.
  @param extend      Specify which axes should be extended to create storage in
                     that dimension.
  @param create_plan Create a copy plan so that distributed-memory dependencies
                     are can be updated.
  @full_ghosts       Populate all ghost dependencies of the given dimension to
                     match the behavior of topo::unstructured.
 */

inline auto
color_auxiliary(std::size_t ne,
  Color nc,
  std::vector<narray_impl::process_color> const & vpc,
  std::vector<bool> const & extend,
  MPI_Comm comm = MPI_COMM_WORLD,
  bool create_plan = false,
  bool full_ghosts = false) {
  using namespace narray_impl;

  flog_assert(create_plan == false, "memory consistency is not yet supported");

  std::vector<process_color> avpc;
  std::vector<std::size_t> partitions;
  for(auto pc : vpc) {
    const std::size_t axes = pc.extents.size();
    process_color apc;
    apc.orientation = pc.orientation;
    apc.global.resize(axes);
    apc.extents.resize(axes);
    apc.logical[0].resize(axes);
    apc.logical[1].resize(axes);
    apc.extended[0].resize(axes);
    apc.extended[1].resize(axes);

    std::size_t extents{1};
    for(std::size_t axis{0}; axis < axes; ++axis) {
      // Use the primary offset to initialize the auxiliary offset,
      // which will be corrected below.
      apc.offset.emplace_back(pc.offset[axis]);

      // Calculate some values
      const std::size_t le = pc.logical[1][axis] - pc.logical[0][axis];
      const std::size_t ee = pc.extended[1][axis] - pc.extended[0][axis];
      const std::size_t bd = pc.logical[0][axis] - pc.extended[0][axis];
      const std::size_t hd = pc.logical[0][axis];
      const std::size_t ex = extend[axis] ? 1 : 0;

      // We can set these explicitly without reference to the orientation.
      apc.global[axis] = pc.global[axis] + ex;
      apc.extents[axis] = ee + ex;

      const std::uint32_t bits = pc.orientation >> axis * 2;

      // Settings and corrections depending on orientation
      if(bits & low && bits & high) {
        apc.logical[0][axis] = bd;
        apc.logical[1][axis] = apc.logical[0][axis] + le + ex;
        apc.extended[0][axis] = 0;
        apc.extended[1][axis] = ee + ex;
      }
      else if(bits & low) {
        apc.extents[axis] += full_ghosts ? hd : 0;
        apc.logical[0][axis] = bd;
        apc.logical[1][axis] = apc.logical[0][axis] + le + ex;
        apc.extended[0][axis] = 0;
        apc.extended[1][axis] = ee + ex;
      }
      else if(bits & high) {
        apc.extents[axis] += full_ghosts ? hd : 0;
        apc.offset[axis] += full_ghosts ? 0 : hd;
        apc.logical[0][axis] = full_ghosts ? hd + ex : ex;
        apc.logical[1][axis] = apc.logical[0][axis] + le;
        apc.extended[0][axis] = full_ghosts ? hd + ex : ex;
        apc.extended[1][axis] = apc.extended[0][axis] + ee;
      }
      else {
        apc.extents[axis] += full_ghosts ? 2 * hd : 0;
        apc.offset[axis] += full_ghosts ? 0 : hd;
        apc.logical[0][axis] = full_ghosts ? hd + ex : ex;
        apc.logical[1][axis] = apc.logical[0][axis] + le;
        apc.extended[0][axis] = full_ghosts ? hd + ex : ex;
        apc.extended[1][axis] = apc.extended[0][axis] + ee;
      } // if

      extents *= apc.extents[axis];
    } // for

    avpc.emplace_back(apc);
    partitions.emplace_back(extents);
  } // for

  {
    auto [rank, size] = util::mpi::info(comm);
    auto pgthr = util::mpi::all_gatherv(partitions, comm);

    partitions.clear();
    partitions.resize(nc);
    util::color_map cm(size, nc, ne);
    std::size_t p{0};
    for(auto vp : pgthr) {
      std::size_t c{0};
      for(auto v : vp) {
        partitions[cm.color_id(p, c)] = v;
        ++c;
      } // for
      ++p;
    } // for
  }

  return std::make_pair(avpc, partitions);
} // color_auxiliary

} // namespace narray_utils
} // namespace topo
} // namespace flecsi
