// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_NARRAY_TYPES_HH
#define FLECSI_TOPO_NARRAY_TYPES_HH

#include "flecsi/data/copy.hh"
#include "flecsi/execution.hh"
#include "flecsi/topo/index.hh"
#include "flecsi/util/color_map.hh"
#include "flecsi/util/mpi.hh"
#include "flecsi/util/serialize.hh"

#include <algorithm>
#include <cstddef>
#include <map>
#include <set>
#include <vector>

namespace flecsi {
namespace topo {
namespace narray_impl {
/// \addtogroup narray
/// \{

enum axes : Dimension { x_axis, y_axis, z_axis };

using coord = std::vector<util::id>;
using gcoord = std::vector<util::gid>;
using hypercube = std::array<coord, 2>;
using interval = std::pair<std::size_t, std::size_t>;
using colors = std::vector<Color>;

/// \cond core

template<Dimension D>
struct neighbors_view {
  using S = short;
  struct iterator {
    using M = std::array<S, D>;

    iterator() : iterator(-1) {}
    explicit iterator(S s) : iterator(s, std::make_index_sequence<D - 1>()) {}

    const M & operator*() const {
      return m;
    }

    iterator & operator++() {
      for(Dimension d = 0; ++m[d] == 2 && ++d < D; m[d - 1] = -1)
        ;
      if(m == M())
        m[0] = 1; // skip origin; note that std::none_of is actually faster
      return *this;
    }
    iterator operator++(int) {
      iterator ret = *this;
      ++*this;
      return ret;
    }

    bool operator==(const iterator & i) const {
      return m == i.m;
    }
    bool operator!=(const iterator & i) const {
      return !(*this == i);
    }

  private:
    template<std::size_t... II>
    iterator(S last, std::index_sequence<II...>) : m{(void(II), -1)..., last} {}

    M m;
  };

  iterator begin() const {
    return {};
  }
  iterator end() const {
    return iterator(2);
  }
};

template<Dimension D, typename I = int>
struct traverse {
  using M = std::array<I, D>;
  M lbnds, ubnds;

  struct iterator {
    using M = std::array<I, D>;

    iterator(M lower_bnds, M upper_bnds, M bnds, I i)
      : iterator(lower_bnds, upper_bnds, bnds) {
      m[D - 1] = i;
    };
    iterator(M lower_bnds, M upper_bnds, M bnds)
      : lbnds(lower_bnds), ubnds(upper_bnds), m(bnds){};

    const M & operator*() const {
      return m;
    }

    iterator & operator++() {
      for(Dimension d = 0; ++m[d] == ubnds[d] && ++d < D;
          m[d - 1] = lbnds[d - 1]) {
      }
      return *this;
    }

    iterator operator++(int) {
      iterator ret = *this;
      ++*this;
      return ret;
    }

    bool operator==(const iterator & i) const {
      return m == i.m;
    }
    bool operator!=(const iterator & i) const {
      return !(*this == i);
    }

  private:
    M lbnds, ubnds, m;
  };

  iterator begin() const {
    return iterator(lbnds, ubnds, lbnds);
  }
  iterator end() const {
    return iterator(lbnds, ubnds, lbnds, ubnds[D - 1]);
  }
  traverse(M lower_bnds, M upper_bnds) : lbnds(lower_bnds), ubnds(upper_bnds){};
};

template<Dimension D, typename I = int>
struct linearize {
  using M = std::array<I, D>;
  M strs;

  I operator()(M indices) {
    I lid = indices[D - 1];
    for(Dimension k = D - 1; k--;) {
      lid = lid * strs[k] + indices[k];
    }
    return lid;
  }
};

/*!
 Type to store the coloring information for one axis of one color.
 */
struct axis_color {
  /// the number of colors on this axis
  Color colors;

  /// the index of this color along this axis
  util::id color_index;

  /// the extent of this color
  util::gid global_extent;

  /// number of boundary layers to be added to the domain of axis
  util::id bdepth;

  /// halo depth (or number of ghost layers) of axis
  util::id hdepth;

  /// global offsets of this color
  std::array<util::gid, 2> offsets;

  /// specify whether the axis is periodic
  /// \showinitializer
  bool periodic = false;

  /// specify whether the axis is extended for auxiliary
  /// \showinitializer
  bool auxiliary = false;

  /// Whether the current color is at the low end of the axis.
  /// This function is supported for GPU execution.
  FLECSI_INLINE_TARGET bool is_low() const {
    return color_index == 0;
  }

  /// Whether the color is at the high end of the axis.
  /// This function is supported for GPU execution.
  FLECSI_INLINE_TARGET bool is_high() const {
    return color_index == (colors - 1);
  }

  /// The global extent of this axis.
  /// This function is supported for GPU execution.
  FLECSI_INLINE_TARGET util::gid global() const {
    return global_extent;
  }

  /// The global index for a given logical index on the local axis.
  /// This function is supported for GPU execution.
  FLECSI_INLINE_TARGET util::gid global_id(util::id i) const {
    util::gid id;
    const util::gid sa = logical<0>(), ea = logical<1>();
    if(is_high() && i >= ea) // periodic high
      id = i - ea;
    else if(is_low() && i < sa) { // periodic low
      id = global() - sa + i;
    }
    else {
      id = offset() + i - sa;
    }
    return id;
  }

  /// The global coordinate offset of the local axis.
  /// Local to global id translation can be computed with this.
  /// This function is supported for GPU execution.
  FLECSI_INLINE_TARGET util::gid offset() const {
    return offsets[0];
  }

  /// The local extent of this color. This is the full size including
  /// boundary depth, and ghosts. The "extent" coordinate implicitly
  /// defines a range [0, extent[.
  /// This function is supported for GPU execution.
  FLECSI_INLINE_TARGET util::id extent() const {
    return logical<1>() + (is_high() ? bdepth : hdepth);
  }

  /// The logical entities, i.e., the entities for this color without
  /// boundary padding or ghosts.
  /// This function is supported for GPU execution.
  template<std::size_t P>
  FLECSI_INLINE_TARGET util::id logical() const {
    static_assert(P == 0 || P == 1);
    auto halo_low = hdepth + auxiliary;
    auto log0 = (is_low() ? bdepth : halo_low);
    return log0 + P * (offsets[1] - offsets[0]);
  }

  /// The extended entities, i.e., the logical entities including boundary
  /// padding. The boundary depth can be computed like:
  ///   boundary_depth_low = logical<0>() - extended<0>();
  ///   boundary_depth_high = extended<1>() - logical<1>();
  /// The ghost depth can be computed like:
  ///   halo_depth_low = extended<0>();
  ///   halo_depth_high = extent() - extended<1>();
  /// This function is supported for GPU execution.
  template<std::size_t P>
  FLECSI_INLINE_TARGET util::id extended() const {
    static_assert(P == 0 || P == 1);
    if constexpr(P == 0) {
      return is_low() ? 0 : logical<0>();
    }
    return is_high() ? extent() : logical<1>();
  }

  /// The index intervals for ghosts of neighboring colors along this axis
  /// @return vector of pairs storing the owner color and its ghost interval
  auto ghost_intervals() const {
    std::vector</* over intervals */
      std::pair<std::size_t, /* owner color */
        std::pair<std::size_t, std::size_t> /* interval */
        >>
      gi;

    auto log0 = logical<0>(), log1 = logical<1>(), tot = extent();
    auto lo = is_low(), hi = is_high();

    if(hdepth) {
      if(!lo)
        gi.push_back({color_index - 1, {0, log0}});
      if(!hi)
        gi.push_back({color_index + 1, {log1, tot}});

      if(periodic) {
        flog_assert(
          bdepth != 0, "boundary depth must be non-zero for periodic axes");
        if(lo)
          gi.push_back({colors - 1, {0, log0}});
        if(hi)
          gi.push_back({0, {log1, tot}});
      }
    }
    return gi;
  }
};

/*!
 Type to store the coloring information for one color.
 */
struct index_color {
  /// Coloring information for each axis
  std::vector<axis_color> axis_colors;

  /// Offsets on the remote color.
  using points = std::map<Color,
    std::vector<std::pair</* local ghost offset, remote shared offset */
      std::size_t,
      std::size_t>>>;

  /// Local ghost intervals.
  using intervals = std::vector<std::pair<std::size_t, std::size_t>>;

  Color color() const {
    auto pr = axis_colors[0].colors;
    Color c = 0;
    for(std::size_t i{0}; i < axis_colors.size() - 1; ++i) {
      c += axis_colors[i + 1].color_index * pr;
      pr *= axis_colors[i + 1].colors;
    }
    return c + axis_colors[0].color_index;
  }

  /// Total extents of this color
  util::gid extents() const {
    util::gid total{1};
    for(const auto & axco : axis_colors) {
      total *= axco.extent();
    }
    return total;
  }

  /// Map a local coordinate to a global one.
  gcoord global_index(coord const & idx) const {
    const auto dimension = axis_colors.size();
    gcoord result(dimension);
    for(Dimension axis = 0; axis < dimension; ++axis) {
      result[axis] = axis_colors[axis].global_id(idx[axis]);
    }
    return result;
  }

  /// Compute a local index from a global coordinate.
  coord local_index(gcoord const & gidx) const {
    const auto dimension = axis_colors.size();
    coord result(dimension);
    for(Dimension axis = 0; axis < dimension; ++axis) {
      auto & axco = axis_colors[axis];
      result[axis] = gidx[axis] - axco.offset() + axco.logical<0>();
    }
    return result;
  }

}; // struct index_color

/// \endcond

/*!
  This type is part of the index definition, and defines the coloring of an
  individal axis
 */
struct axis_definition {
  /// Encodes the number of colors into which axis will be
  /// divided, and the extents to be partitioned
  util::offsets colormap;

  /// halo depth (or number of ghost layers) of axis
  /// \showinitializer
  util::id hdepth = 0;

  /// number of boundary layers to be added to the domain of axis
  /// \showinitializer
  util::id bdepth = 0;

  /// specify whether axis is periodic
  /// \showinitializer
  bool periodic = false;

  /// specify whether axis is extended for auxiliary
  /// \showinitializer
  bool auxiliary = false;
};

/*!
  This type is part of the coloring, and encapsulates the coloring information
  for a single index space (such as how many colors per axis the mesh needs to
  be partitioned into, if boundaries are periodic, etc), that is used by the
  coloring algorithm to create mesh partitions.
 */
struct index_definition {
  /// coloring information of each axis
  std::vector<axis_definition> axes;

  /// whether to include diagonally connected (i.e.,
  /// connected through vertex) as well as face
  /// connected entities during primary partitioning
  /// \showinitializer
  bool diagonals = false;

  /// whether to create a copy plan
  /// \showinitializer
  bool create_plan = true;

  /// whether to use include full ghost information for auxiliaries
  /// \showinitializer
  bool full_ghosts = true;

  /// Dimensions of index space
  Dimension dimensions() const {
    const Dimension dimensions = axes.size();
    flog_assert(dimensions < 17,
      "current implementation is limited to 16 dimensions (uint32_t)");
    return dimensions;
  }

  /// Number of colors in index space
  Color colors() const {
    Color nc = 1;
    for(const auto & ax : axes) {
      nc *= ax.colormap.size();
    } // for
    return nc;
  }

  /// Number of indices in index space
  util::gid indices() const {
    util::gid indices{1};
    for(const auto & ax : axes) {
      indices *= ax.colormap.total();
    } // for
    return indices;
  }

  /*!
   * Return a coloring for the current MPI rank on the given
   * communicator
   *
    @param comm MPI communicator
   */
  std::vector<index_color> process_coloring(
    MPI_Comm comm = MPI_COMM_WORLD) const {
    auto [rank, size] = util::mpi::info(comm);

    /*
      Create a color map for the total number of colors (product of axis
      colors) to the number of processes.
     */
    const util::equal_map cm(colors(), size);

    /*
      Create a coloring for each color on this process.
     */
    std::vector<index_color> coloring;
    for(const Color c : cm[rank]) {
      /*
        Get the indices representation of our color.
       */
      narray_impl::colors color_indices;
      {
        Color i = c;
        for(const auto & ax : axes) {
          auto & axcm = ax.colormap;
          color_indices.push_back(i % axcm.size());
          i /= axcm.size();
        }
      }

      /*
        Make the coloring information for our color.
       */
      coloring.emplace_back(make_color(color_indices));
    } // for
    return coloring;
  }

  /*!
    Create a coloring for the given color (as defined by color_indices).

    @param color_indices indices of the given color w.r.t the grid of colors in
    the decomposition.

    \return coloring for the given color
   */
  index_color make_color(const narray_impl::colors & color_indices) const {
    const Dimension dimension = dimensions();
    index_color idxco;

    for(Dimension axis = 0; axis < dimension; ++axis) {
      auto & ax = axes[axis];
      auto ci = color_indices[axis];
      const util::offsets & em = ax.colormap;
      const bool lo = (ci == 0);
      const util::gid ex = ax.auxiliary;

      // primary coloring
      util::gid offset_low = em(ci);
      util::gid offset_high = em(ci + 1);

      // modifications if auxiliary coloring
      if(ex) {
        offset_low += !lo;
        offset_high += 1;
      }

      idxco.axis_colors.push_back({em.size(),
        ci,
        em.total() + ex,
        ax.bdepth,
        full_ghosts ? ax.hdepth : 0,
        {offset_low, offset_high},
        ax.periodic,
        ax.auxiliary});
    } // for

    return idxco;
  } // make_color

  /*!
    Compute the ghost points and intervals for a given process coloring
    @param idxco index color
    \return ghost points and intervals for the given coloring
  */
  std::pair<index_color::points, index_color::intervals> ghosts(
    const index_color & idxco) const {
    const auto dimension = dimensions();
    index_color::points points;
    index_color::intervals intervals;

    /*
      Here, we compose the intervals from each sub-dimension to form
      the actual full-dimensional subregions. These define the coloring.
     */

    std::function<std::vector<std::pair<narray_impl::colors, hypercube>>(
      Dimension, Dimension)>
      expand = [&](Dimension dim, Dimension top) {
        std::vector<std::pair<narray_impl::colors, hypercube>> sregs;

        for(Dimension axis = 0; axis < dim; ++axis) {
          auto & axco = idxco.axis_colors[axis];
          std::optional<std::vector<std::pair<narray_impl::colors, hypercube>>>
            ssubs;

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
            auto log0 = axco.logical<0>();
            auto log1 = axco.logical<1>();
            for(size_t off{log0}; off < log1; ++off) {
              for(auto & s : subs) {
                s.first[axis] = axco.color_index;
                s.second[0][axis] = off;
                s.second[1][axis] = off + 1;
                sregs.emplace_back(s);
              } // for
            } // for
          } // if

          /*
            Add the subregions for this dimension.
           */
          for(auto & i : axco.ghost_intervals()) {
            narray_impl::colors co(top, 0);
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
              auto & acol = idxco.axis_colors[a];
              co[a] = acol.color_index;
              start[a] = acol.logical<0>();
              end[a] = acol.logical<1>();
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

    // Convenience functions to map between colors and indices.
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

    // Loop through the subregions and create the actual coloring.
    std::unordered_map<Color, index_color> idxmap;
    narray_impl::colors ax_colors;
    for(auto & axco : idxco.axis_colors) {
      ax_colors.push_back(axco.colors);
    }

    for(auto & s : subregions) {
      auto & color_indices = s.first;
      auto co = idx2co(color_indices, ax_colors);
      if(idxmap.find(co) == idxmap.end()) {
        idxmap[co] = make_color(color_indices);
      } // if

      /*
        The subregions are defined by hypercubes. These must be broken
        up into contiguous intervals. This lambda recurses each subregion
        to break up the volume into contiguous chunks.
       */
      std::function<void(Color, hypercube, Dimension)> make =
        [&](Color clr, hypercube const & subregion, Dimension axis) {
          if(axis == 0) {
            coord extents(dimension);
            for(Dimension a = 0; a < dimension; ++a) {
              extents[a] = idxco.axis_colors[a].extent();
            }

            // Compute the local memory interval.
            auto const start = idx2co(subregion[0], extents);
            auto const end = idx2co(op2cls(subregion[1]), extents);

            // The output intervals are closed on the start
            // and open on the end, i.e., [start, end)
            intervals.push_back({start, end + 1});

            /*
              Loop through the local interval sizes, and add the remote
              pointer offsets.
             */
            auto const gidx = idxco.global_index(subregion[0]);

            auto & ridxco = idxmap.at(clr);
            auto const ridx = ridxco.local_index(gidx);
            coord rextents(dimension);
            for(Dimension a = 0; a < dimension; ++a) {
              rextents[a] = ridxco.axis_colors[a].extent();
            }
            auto rmtoff = idx2co(ridx, rextents);

            for(std::size_t off{0}; off < (end + 1) - start; ++off) {
              points[clr].push_back({start + off, rmtoff + off});
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

      make(co, s.second, dimension - 1);
    } // for
    return make_pair(points, intervals);
  }
};

/*
 * Return the global communication graph over all colors
 */
template<Dimension D>
std::vector<std::vector<Color>>
peers(index_definition idef) {
  using A = std::array<int, D>;
  using nview = flecsi::topo::narray_impl::neighbors_view<D>;

  A color_bnd, color_strs, axes_bdepths, color_indices;
  std::array<bool, D> axes_periodic;
  for(Dimension k = 0; k < D; ++k) {
    color_bnd[k] = 0;
    color_strs[k] = idef.axes[k].colormap.size();
    axes_bdepths[k] = idef.axes[k].bdepth;
    axes_periodic[k] = idef.axes[k].periodic;
  }

  std::vector<std::vector<Color>> peer;

  // Loop over all colors
  for(auto && v : traverse<D>(color_bnd, color_strs)) {
    for(Dimension k = 0; k < D; ++k)
      color_indices[k] = v[k];

    std::vector<Color> ngb_colors;

    // Loop over neighbors of a color
    for(auto && nv : nview()) {
      A indices_mo;
      for(Dimension d = 0; d < D; ++d) {
        indices_mo[d] = color_indices[d] + nv[d];
        // if boundary depth, add the correct ngb color
        if(axes_periodic[d]) {
          if((indices_mo[d] == -1) && (color_indices[d] == 0) &&
             axes_bdepths[d])
            indices_mo[d] = color_strs[d] - 1;
          if((indices_mo[d] == color_strs[d]) &&
             (color_indices[d] == color_strs[d] - 1) && axes_bdepths[d])
            indices_mo[d] = 0;
        }
      }

      bool valid_ngb = true;
      for(Dimension k = 0; k < D; ++k) {
        if((indices_mo[k] == -1) || (indices_mo[k] == color_strs[k]))
          valid_ngb = false;
      }

      int lid;
      if(valid_ngb) {
        if(D == 1) {
          lid = indices_mo[D - 1];
        }
        else {
          lid = indices_mo[D - 1] * color_strs[D - 2] + indices_mo[D - 2];
          for(int k = D - 3; k >= 0; --k) {
            lid = lid * color_strs[k] + indices_mo[k];
          }
        }
        ngb_colors.push_back((Color)lid);
      }
    }

    std::sort(ngb_colors.begin(), ngb_colors.end());
    ngb_colors.erase(
      std::unique(ngb_colors.begin(), ngb_colors.end()), ngb_colors.end());
    peer.push_back(ngb_colors);
  }

  return peer;
}

inline std::ostream &
operator<<(std::ostream & stream, axis_color const & ac) {
  stream << "global_extent\n"
         << ac.global_extent << '\n'
         << "bdepth\n"
         << ac.bdepth << '\n'
         << "hdepth\n"
         << flog::container{ac.hdepth} << '\n'
         << "offsets\n"
         << flog::container{ac.offsets} << '\n'
         << "periodic\n"
         << (ac.periodic ? "true" : "false") << '\n'
         << "auxiliary\n"
         << (ac.auxiliary ? "true" : "false") << '\n';
  return stream;
} // operator<<

inline std::ostream &
operator<<(std::ostream & stream, index_color const & ic) {
  for(Dimension axis = 0; axis < ic.axis_colors.size(); ++axis) {
    stream << "axis " << axis << ":\n" << ic.axis_colors[axis] << '\n';
  }
  return stream;
} // operator<<

/// \}
} // namespace narray_impl

/// \addtogroup narray
/// \{

/// \if core
/// Specialization-independent definitions.
/// \endif
struct narray_base {
  using axis_color = narray_impl::axis_color;
  using index_color = narray_impl::index_color;
  using coord = narray_impl::coord;
  using gcoord = narray_impl::gcoord;
  using hypercube = narray_impl::hypercube;
  using colors = narray_impl::colors;
  using axis_definition = narray_impl::axis_definition;
  using index_definition = narray_impl::index_definition;

  /*!
   This domain enumeration provides a classification of the various
   types of partition entities that can be requested out of a topology
   specialization created using this type. The following describes what each
   of the domain enumeration means in a mesh part returned by the coloring
   algorithm. For the structured mesh partitioning, the partition info is
   specified per axis.

   These domains are used in many of the interface methods to provide
   information such as size, extents, offsets about them.
  */
  enum class domain : std::size_t {
    logical, ///<  the logical, i.e., the owned part of the axis
    extended, ///< the boundary padding along with the logical part
    all, ///< the ghost padding along with the logical part
    boundary_low, ///< the boundary padding on the lower bound of the axis
    boundary_high, ///< the boundary padding on the upper bound of the axis
    ghost_low, ///< the ghost padding on the lower bound of the axis
    ghost_high, ///< the ghost padding on the upper bound of the axis
    global ///< global info about the mesh, the meaning depends on what is being
           ///< queried
  };

  /// Coloring type.
  /// \ingroup narray
  struct coloring {
    MPI_Comm comm;

    std::vector</* over index spaces */
      index_definition>
      idx_colorings;

    Color colors() const {
      return idx_colorings[0].colors();
    }
  }; // struct _coloring

  static std::size_t idx_size(std::vector<std::size_t> vs, std::size_t c) {
    return vs[c];
  }
  /*!
   Method to compute the local ghost "intervals" and "points" which store map of
   local ghost offset to remote/shared offset. This method is called by the
   "make_copy_plan" method in the derived topology to create the copy plan
   objects.

   @param idef index definition
   @param[out] num_intervals vector of number of ghost intervals, over all
   colors, this vector is assumed to be sized correctly (all colors)
   @param[out] intervals  vector of local ghost intervals, over process colors
   @param[out] points vector of maps storing (local ghost offset, remote shared
   offset) for a shared color, over process colors
  */
  static void idx_itvls(index_definition const & idef,
    std::vector<std::size_t> & num_intervals,
    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> & intervals,
    std::vector<std::map<Color,
      std::vector<std::pair<std::size_t, std::size_t>>>> & points,
    MPI_Comm const & comm) {
    std::vector<std::size_t> local_itvls;
    for(const auto & idxco : idef.process_coloring(comm)) {
      auto [pts, itvls] = idef.ghosts(idxco);
      local_itvls.emplace_back(itvls.size());
      intervals.emplace_back(std::move(itvls));
      points.emplace_back(std::move(pts));
    }

    /*
      Gather global interval sizes.
     */

    auto global_itvls = util::mpi::all_gatherv(local_itvls, comm);

    auto it = num_intervals.begin();
    for(const auto & pv : global_itvls) {
      for(auto i : pv) {
        *it++ = i;
      }
    }
  } // idx_itvls

  // for make_copy_plan
  static void set_dests(
    data::multi<field<data::intervals::Value>::accessor<wo>> aa,
    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> const &
      intervals) {
    std::size_t ci = 0;
    for(auto [c, a] : aa.components()) {
      auto & iv = intervals[ci++];
      flog_assert(a.span().size() == iv.size(),
        "interval size mismatch a.span ("
          << a.span().size() << ") != intervals (" << iv.size() << ")");
      std::size_t i{0};
      for(auto & it : iv) {
        a[i++] = data::intervals::make({it.first, it.second}, c);
      } // for
    }
  }

  // for make_copy_plan
  template<PrivilegeCount N>
  static void set_ptrs(
    data::multi<field<data::points::Value>::accessor1<privilege_repeat<wo, N>>>
      aa,
    std::vector<std::map<Color,
      std::vector<std::pair<std::size_t, std::size_t>>>> const & points) {
    std::size_t ci = 0;
    for(auto & a : aa.accessors()) {
      for(auto const & si : points[ci++]) {
        for(auto p : si.second) {
          // si.first: owner
          // p.first: local ghost offset
          // p.second: remote shared offset
          a[p.first] = data::points::make(si.first, p.second);
        } // for
      } // for
    }
  }

}; // struct narray_base

/// \}
} // namespace topo

} // namespace flecsi

#endif
