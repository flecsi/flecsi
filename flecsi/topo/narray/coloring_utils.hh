// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_NARRAY_COLORING_UTILS_HH
#define FLECSI_TOPO_NARRAY_COLORING_UTILS_HH

#include "flecsi/flog.hh"
#include "flecsi/topo/narray/types.hh"
#include "flecsi/topo/types.hh"
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

/*!
  Create an axial color distribution for the given number of processes.

  @param np total number of colors
  @param indices number of entities per axis

  \return number of colors per axis after decomposition
 */

inline narray_impl::colors
distribute(Color np, narray_impl::gcoord indices) {
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

/*!
  Create a color map of the extents for the given number of colors. The method
  first finds the distribution of colors per axis. Then, the end offsets for
  each color per axis is computed.

  @param num_colors total number of colors
  @param indices number of entities per axis

  \return offsets encoding the end offset per color  per axis after
  decomposition
 */
inline narray_impl::color_map
make_color_maps(std::size_t num_colors, const narray_impl::gcoord & indices) {
  narray_impl::color_map cm;
  auto colors = distribute(num_colors, indices);
  flog(warn) << flog::container{colors} << std::endl;
  for(std::size_t d = 0; d < indices.size(); d++) {
    flecsi::util::equal_map em{indices[d], colors[d]};
    cm.push_back(em);
  }
  return cm;
} // make_color_map
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
  const narray_impl::color_map & axis_colormaps) {
  using namespace narray_impl;

  std::uint32_t o{mask::interior};
  std::uint32_t shft{mask::low};
  // clang-format off
  for(Dimension axis = 0; axis < dimension; ++axis) {
    o |= [ci = color_indices[axis], nc = axis_colormaps[axis].size(), l = shft,
      h = shft << 1]() {
      return
        (ci == 0 && ci == (nc - 1)) ?
          l | h :
        (0 < ci && ci < (nc - 1)) ?
          mask::interior :
        (ci == 0) ?
          l :
          h;
    }();
    shft <<= 2;
  } // for
  // clang-format on

  return o;
} // orientation

/*!
  Create an index coloring for the given color (as defined by color_indices).

  @param dimension mesh dimension
  @param color_indices indices of the given color w.r.t the grid of colors in
  the decomposition.
  \param axem \c offsets of entities over colors on each axis
  @param orient encodes orientation of the given color w.r.t the domain
  @param hdepths halo depths per axis
  @param bdepths domain boundary depths per axis
  @periodic bool indicating is the mesh is to be considered periodic per axis

  \return index coloring and ghost intervals for the given color
 */

inline auto
make_color(Dimension dimension,
  const narray_impl::colors & color_indices,
  std::vector<util::offsets> const & axem,
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
    const util::offsets & em = axem[axis];
    const std::uint32_t bits = orient >> axis * 2;
    const bool lo = bits & mask::low, hi = bits & mask::high;
    auto &log0 = idxco.logical[0][axis], &log1 = idxco.logical[1][axis],
         &ext0 = idxco.extended[0][axis], &tot = idxco.extents[axis];
    idxco.global[axis] = em.total();
    // Build the layout incrementally from the bottom edge,
    // adding boundary or halo layers as appropriate:
    log0 = (lo ? bdepths : hdepths)[axis];
    ext0 = lo ? 0 : log0; // NB: the number of low-side ghosts
    log1 = log0 + em[axis_color].size();
    tot = log1 + (hi ? bdepths : hdepths)[axis];
    idxco.extended[1][axis] = hi ? tot : log1;
    idxco.offset[axis] = em(axis_color);

    auto & gi = ghstitvls[axis];
    if(!lo)
      gi.push_back({axis_color - 1, {0, log0}});
    if(!hi)
      gi.push_back({axis_color + 1, {log1, tot}});

    if(periodic[axis]) {
      flog_assert(hdepths[axis] == bdepths[axis],
        "halo and boundary depth must be identical for periodic axes");

      if(lo)
        gi.push_back({em.size() - 1, {0, log0}});
      if(hi)
        gi.push_back({0, {log1, tot}});
    }

    extents *= idxco.extents[axis];
  } // for

  return std::make_tuple(idxco, ghstitvls, extents);
} // make_color

/*!
  Generate a coloring for the provided coloring definition. This method
  can be used to generate coloring information for more colors than the number
  of processes on which it is being run. It returns two pieces of information,
  first, the total number of colors for which the decomposition is created.
  Second, a list of coloring information for each color on the current process
  for an index-space.

  @param index_spaces coloring_definitions per index space, \sa
  coloring_definition

  \return [number of colors, vector of map of process colors to index_colorings]
 */

inline auto
color(narray_impl::coloring_definition const & cd,
  MPI_Comm comm = MPI_COMM_WORLD) {
  using namespace narray_impl;

  const Dimension dimension = cd.axis_colormaps.size();

  flog_assert(dimension < 17,
    "current implementation is limited to 16 dimensions (uint32_t)");

  flog_assert(cd.axis_hdepths.size() == dimension,
    "size must match the intended dimension(" << dimension << ")");
  flog_assert(cd.axis_bdepths.size() == dimension,
    "size must match the intended dimension(" << dimension << ")");
  flog_assert(cd.axis_periodic.size() == dimension,
    "size must match the intended dimension(" << dimension << ")");

  auto [rank, size] = util::mpi::info(comm);

  /*
    Create a color map for each dimension. Because we are using a
    tensor-product strategy below, we can use each map to define the
    sub colors and offsets for an individual axis.
   */

  Color nc = 1;
  std::size_t indices{1};
  for(Dimension d = 0; d < dimension; ++d) {
    nc *= cd.axis_colormaps[d].size();
    indices *= cd.axis_colormaps[d].total();
  } // for

  /*
    Create a color map for the total number of colors (product of axis
    colors) to the number of processes.
   */

  const util::equal_map cm(nc, size);

  /*
    Create a coloring for each color on this process.
   */

  std::vector<process_color> coloring;
  std::vector<std::size_t> partitions;
  for(const Color c : cm[rank]) {
    /*
      Convenience functions to map between colors and indices.
     */

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

    colors color_indices;
    {
      Color i = c;
      for(auto & cm : cd.axis_colormaps) {
        color_indices.push_back(i % cm.size());
        i /= cm.size();
      }
    }

    /*
      Find our orientation within the color space.
     */

    uint32_t orient = orientation(dimension, color_indices, cd.axis_colormaps);

    /*
      Make the coloring information for our color.
     */

    auto [idxco, ghstitvls, extents] = make_color(dimension,
      color_indices,
      cd.axis_colormaps,
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

    auto rmtidx = [dimension](process_color const & idxco, coord const & gidx) {
      coord result(dimension);
      auto & log0 = idxco.logical[0];
      for(Dimension axis = 0; axis < dimension; ++axis) {
        result[axis] = gidx[axis] - idxco.offset[axis] + log0[axis];
      }
      return result;
    };

    /*
      Map a local coordinate to a global one.
     */

    auto l2g = [dimension, orient](
                 process_color const & idxco, coord const & idx) {
      const auto & [start, end] = idxco.logical;
      coord result(dimension);
      for(Dimension axis = 0; axis < dimension; ++axis) {
        uint32_t bits = orient >> axis * 2;
        const std::size_t sa = start[axis], ea = end[axis], i = idx[axis];
        if(bits & high && i >= ea) // periodic high
          result[axis] = i - ea;
        else if(bits & low && i < sa) {
          /* periodic low */
          result[axis] = idxco.global[axis] - sa + i;
        }
        else {
          result[axis] = idxco.offset[axis] + i - sa;
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
      colors axis_colors;
      for(auto & ac : cd.axis_colormaps)
        axis_colors.push_back(ac.size());
      auto co = idx2co(s.first, axis_colors);
      if(idxmap.find(co) == idxmap.end()) {
        /*
          Create basic coloring information for the owning color, so
          that we can determine the remote offsets for our points.
          The coloring informaiton is stored for subsequent use.
         */

        auto [ridxco, rghstitvls, extents] = make_color(dimension,
          s.first,
          cd.axis_colormaps,
          orientation(dimension, s.first, cd.axis_colormaps),
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

  concatenate(partitions, nc, comm);

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
color_auxiliary(Color nc,
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
      if(bits & mask::low && bits & mask::high) {
        apc.logical[0][axis] = bd;
        apc.logical[1][axis] = apc.logical[0][axis] + le + ex;
        apc.extended[0][axis] = 0;
        apc.extended[1][axis] = ee + ex;
      }
      else if(bits & mask::low) {
        apc.extents[axis] += full_ghosts ? hd : 0;
        apc.logical[0][axis] = bd;
        apc.logical[1][axis] = apc.logical[0][axis] + le + ex;
        apc.extended[0][axis] = 0;
        apc.extended[1][axis] = ee + ex;
      }
      else if(bits & mask::high) {
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

  concatenate(partitions, nc, comm);

  return std::make_pair(avpc, partitions);
} // color_auxiliary

} // namespace narray_utils
} // namespace topo
} // namespace flecsi

#endif
