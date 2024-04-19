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

using coord = std::vector<util::id>;
using gcoord = std::vector<util::gid>;
using hypercube = std::array<coord, 2>;
using interval = std::pair<std::size_t, std::size_t>;
using colors = std::vector<Color>;

/// \cond core

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

// Generate all selections from {-1,0,1} other than all 0s.
template<Dimension D>
struct neighbors_view {
  using S = short;
  using value_type = std::array<S, D>;

  struct iterator {
    using M = value_type;
    iterator(S f, S s) : iterator(f, s, std::make_index_sequence<D - 1>()) {}

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
    [[nodiscard]] iterator operator++(int) {
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
    iterator(S first, S last, std::index_sequence<II...>)
      : m{(void(II), first)..., last} {}
    M m;
  };

  iterator begin() const {
    return iterator(-1, -1);
  }
  iterator end() const {
    return iterator(-1, 2);
  }
};

// Generate multi-indices in a hyperrectangle.
// The first index varies fastest.
template<Dimension D, typename I = util::id>
struct traverse {
  using M = std::array<I, D>;
  M lbnds, ubnds;

  struct iterator {
    using M = std::array<I, D>;

    iterator(M lower_bnds, M upper_bnds, M bnds, I i)
      : iterator(lower_bnds, upper_bnds, bnds) {
      m[D - 1] = i;
    }
    iterator(M lower_bnds, M upper_bnds, M bnds)
      : lbnds(lower_bnds), ubnds(upper_bnds), m(bnds) {}

    const M & operator*() const {
      return m;
    }

    iterator & operator++() {
      for(Dimension d = 0; ++m[d] == ubnds[d] && ++d < D;
          m[d - 1] = lbnds[d - 1]) {
      }
      return *this;
    }

    [[nodiscard]] iterator operator++(int) {
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
  traverse(M lower_bnds, M upper_bnds) : lbnds(lower_bnds), ubnds(upper_bnds) {}
};

template<Dimension D, typename I = util::id>
struct linearize {
  using M = std::array<I, D>;
  M strs;

  I operator()(M indices) const {
    I lid = indices[D - 1];
    for(Dimension k = D - 1; k--;) {
      lid = lid * strs[k] + indices[k];
    }
    return lid;
  }
};

/*!
 The layout of one axis of one color.
 \gpu.
 */
struct axis_layout {
  FLECSI_INLINE_TARGET axis_layout(util::id bdepth,
    util::id log,
    util::id halo_up, // depth of points communicated upward
    util::id halo_down,
    util::id unused, // ghosts, because of diagonal position
    bool lo,
    bool hi)
    : bdy{lo ? bdepth : 0, hi ? bdepth : unused}, gh{lo ? 0 : halo_up,
                                                    hi ? 0 : halo_down},
      log(log), shr{lo ? 0 : halo_down, hi ? 0 : halo_up} {}

  /// The local extent of this color. This is the full size including
  /// boundary depth, and ghosts. The "extent" coordinate implicitly
  /// defines a range [0, extent[.
  FLECSI_INLINE_TARGET util::id extent() const {
    return both(bdy) + both(gh) + log;
  }

  /// The beginning or end index of the logical entities, i.e., the entities
  /// for this color without
  /// boundary padding or ghosts.
  /// \tparam E 0 or 1 for beginning or end
  template<short E>
  FLECSI_INLINE_TARGET util::id logical() const {
    static_assert(E == 0 || E == 1);
    return bdy[0] + gh[0] + E * log;
  }

  /// The beginning or end of the exclusive logical entities (_i.e.,_ that are
  /// not ghosts elsewhere).
  /// Really the end and beginning of the shared entities on each side; the
  /// end can come first if an entity is shared with both neighbors.
  /// \tparam E 0 or 1 for beginning or end
  template<short E>
  FLECSI_INLINE_TARGET util::id exclusive() const {
    static_assert(E == 0 || E == 1);
    return bdy[0] + gh[0] + (E ? log - shr[1] : shr[0]);
  }

  /// The beginning or end index of the domain entities, including logical and
  /// ghost entities.
  /// \tparam E 0 or 1 for beginning or end
  template<short E>
  FLECSI_INLINE_TARGET util::id ghost() const {
    static_assert(E == 0 || E == 1);
    return bdy[0] + E * (both(gh) + log);
  }

  /// The extended entities, i.e., the logical entities including boundary
  /// padding. The boundary depth can be computed like:\code
  ///   boundary_depth_low = logical<0>() - extended<0>();
  ///   boundary_depth_high = extended<1>() - logical<1>();\endcode
  /// The ghost depth can be computed like:\code
  ///   halo_depth_low = extended<0>();
  ///   halo_depth_high = extent() - extended<1>();\endcode
  /// \tparam E 0 or 1 for beginning or end
  template<short E>
  FLECSI_INLINE_TARGET util::id extended() const {
    static_assert(E == 0 || E == 1);
    return gh[0] + E * (both(bdy) + log);
  }

  void check_halo() const {
    if(std::max(shr[0], shr[1]) > log)
      throw std::invalid_argument("halo depth larger than logical color depth");
  }

private:
  FLECSI_INLINE_TARGET static util::id both(const util::id (&a)[2]) {
    return a[0] + a[1];
  }

  util::id bdy[2], gh[2], log, // these are all disjoint
    shr[2];
};

/// Information about the layout of an entire axis.
/// \note Unlike in the user interface, here the ghosts copied from the other
///   end of a \c periodic axis are considered to be a halo, not a boundary.
struct axis {
  Color colors; ///< The number of subdivisions for colors.
  util::gid extent; ///< The total number of index points.
  util::id bdepth, ///< The boundary padding depth at each end (0 if periodic).
    hdepth; ///< The depth of (primary) halos where they appear.
  bool periodic, ///< Whether the axis is periodic.
    auxiliary, ///< Whether the entities are delimiters (in this dimension).
    /// Whether an auxiliary ghost exists even if one of its adjacent primary
    /// entities is absent on a color.
    full_ghosts;
};
/// High-level information about one axis of one color.
struct axis_color {
  /// Color-independent information.
  narray_impl::axis axis;
  /// The color index along the axis.
  Color color;
  /// The number of logical index points.
  util::id logical;
  /// The position of the first logical index point on the global axis.
  util::gid offset;

  /// Whether the current color is at the low end of the axis.
  FLECSI_INLINE_TARGET bool low() const {
    return !color;
  }
  /// Whether the color is at the high end of the axis.
  FLECSI_INLINE_TARGET bool high() const {
    return color == axis.colors - 1;
  }

  Color color_step(Color d) const {
    return (color + axis.colors + d) % axis.colors;
  }

  /// The global index for a given local index.
  FLECSI_INLINE_TARGET util::gid global_id(util::id i) const {
    const auto al = (*this)();
    const util::id l0 = al.logical<0>(), l1 = al.logical<1>();
    return low() && i < l0     ? axis.extent - l0 + i
           : high() && i >= l1 ? i - l1
                               : offset + i - l0;
  }

  // skin indicates the communication with diagonal neighbor colors of
  // auxiliaries associated with non-diagonal primaries.
  FLECSI_INLINE_TARGET axis_layout operator()(bool skin = false) const {
    const util::id halo_down =
      axis.hdepth + (axis.auxiliary && axis.full_ghosts);
    assert(!skin || halo_down);
    return {axis.bdepth,
      logical,
      axis.hdepth ? axis.hdepth - (axis.auxiliary && !axis.full_ghosts) : 0,
      skin ? 1 : halo_down,
      skin ? halo_down - 1 : 0,
      low() && !axis.periodic,
      high() && !axis.periodic};
  }
};

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

  /// Number of boundary padding layers to be added at each end of the axis.
  /// \showinitializer
  util::id bdepth = 0;

  /// Whether the axis is periodic.
  /// The boundary index points for a periodic axis are copied as ghosts from
  /// the other end of the axis and must match the ghost points in number, but
  /// they are not categorized as ghost points.
  /// <!-- Internally they are ghosts, not boundary points. -->
  /// \showinitializer
  bool periodic = false;

  /// Whether the entity is a delimiter in this axis.
  /// In a structured mesh, a face is a delimiter in the one axis
  /// perpendicular to it; a vertex is a delimiter in all axes.
  ///
  /// Without considering \c auxiliary, the \c axis_definition objects for an
  /// index space describe a set of primary entities ("cells").  Each
  /// non-periodic, auxiliary axis is augmented with an additional index point
  /// at the high end, such that every primary has an auxiliary on each side.
  /// Each color owns the auxiliaries just before its first primaries.
  /// Ghost copies are performed for auxiliary entities associated with ghost
  /// layers of primaries in accordance with \c index_definition::full_ghosts.
  /// \note The primaries in terms of which auxiliaries are defined need not
  ///   exist as their own index space.
  /// \showinitializer
  bool auxiliary = false;

  bool extra() const {
    return auxiliary && !periodic;
  }

  axis_color operator()(Color c, bool full) const {
    const util::gid o = colormap(c);
    axis_color ret{{colormap.size(),
                     colormap.total() + extra(),
                     periodic ? 0 : bdepth, // bdepth meaningless when periodic
                     hdepth,
                     periodic,
                     auxiliary,
                     full},
      c,
      static_cast<util::id>(colormap(c + 1) - o) +
        (c == colormap.size() - 1 && extra()),
      o};
    ret().check_halo();
    return ret;
  }
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

  /// Unused.
  /// \deprecated Omit the initialization (and \e assign \c full_ghosts if
  ///   needed).
  bool create_plan = true;

  /// Whether to include full ghost information for auxiliaries.
  /// When true, all the auxiliaries surrounding needed primary entities
  /// (owned and ghost) are included. If false, only auxiliaries
  /// completely surrounded by needed primaries are included.
  /// For primary entities, this flag has no effect.
  /// \showinitializer
  bool full_ghosts = true;

  /// \cond core

  /// Number of colors in index space
  Color colors() const {
    Color nc = 1;
    for(const auto & ax : axes) {
      nc *= ax.colormap.size();
    } // for
    return nc;
  }

  std::vector<narray_impl::colors> process_colors(
    MPI_Comm comm = MPI_COMM_WORLD) const {
    // Check boundary and halo depth compatibility for periodic axes.
    for(const auto & axis : axes) {
      if(axis.periodic && axis.bdepth != axis.hdepth)
        flog_fatal("periodic boundary depth must match halo depth");
    }

    auto [rank, size] = util::mpi::info(comm);

    /*
      Create a color map for the total number of colors (product of axis
      colors) to the number of processes.
     */
    const util::equal_map cm(colors(), size);

    std::vector<narray_impl::colors> ret;
    for(const Color c : cm[rank]) {
      /*
        Get the indices representation of our color.
       */
      auto & color_indices = ret.emplace_back();
      {
        Color i = c;
        for(const auto & ax : axes) {
          auto & axcm = ax.colormap;
          color_indices.push_back(i % axcm.size());
          i /= axcm.size();
        }
      }
    } // for
    return ret;
  }

  axis_color make_axis(Dimension a, Color c) const {
    return axes[a](c, full_ghosts);
  }

  /// \endcond
};

/// \}
} // namespace narray_impl

/// \addtogroup narray
/// \{

/// Specialization-independent definitions.
/// Name as \c base in an \c narray specialization.
struct narray_base {
  using axis_color = narray_impl::axis_color;
  using axis_layout = narray_impl::axis_layout;
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
   information about an axis such as size, extents, and offsets.
   \image html narray-layout.svg "Layouts for each possible orientation." width=100%
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
  struct coloring {
    /// Coloring information for each index space.
    std::vector<index_definition> idx_colorings;

    Color colors() const {
      return idx_colorings[0].colors();
    }
  };

  /*!
    Create an axial color distribution for the given number of processes.

    @param np total number of colors
    @param indices number of entities per axis

    \return number of colors per axis after decomposition
   */
  static colors distribute(Color np, gcoord indices) {
    colors parts(indices.size(), 1);

    auto facs = narray_impl::factor(np);

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
    Create a vector of axis definitions with default settings (hdepth=0,
    bdepth=0, periodic=false, etc) for the given extents and number of colors.

    The method takes as input the distribution of colors over axes.
    Then, the end offsets for each color per axis is computed and used to
    initialize each axis's definition object.

    @param color_dist distribution of colors per axis
    @param indices number of entities per axis

    \return vector of axis definitions
   */
  static std::vector<axis_definition> make_axes(const colors & color_dist,
    const gcoord & indices) {
    std::vector<axis_definition> axes;
    for(std::size_t d = 0; d < indices.size(); d++) {
      flecsi::util::equal_map em{indices[d], color_dist[d]};
      axes.push_back({em});
    }
    return axes;
  } // make_axes

  /*!
    Choose a breakdown of colors per axis and construct axis definitions.
    \see make_axes(const narray_impl::colors & color_dist, const
    narray_impl::gcoord
    &), #distribute


    @param num_colors total number of colors
    @param indices number of entities per axis

    \return vector of axis definitions
   */
  static std::vector<axis_definition> make_axes(Color num_colors,
    const gcoord & indices) {
    const auto & colors = distribute(num_colors, indices);
    return make_axes(colors, indices);
  } // make_axes

  static std::size_t idx_size(std::vector<std::size_t> vs, std::size_t c) {
    return vs[c];
  }

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
