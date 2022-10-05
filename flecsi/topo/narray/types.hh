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

/// \cond core
namespace flecsi {
namespace topo {
namespace narray_impl {
/// \addtogroup narray
/// \{

enum masks : uint32_t { interior = 0b00, low = 0b01, high = 0b10 };

enum axes : Dimension { x_axis, y_axis, z_axis };

using coord = std::vector<std::size_t>;
using hypercube = std::array<coord, 2>;
using interval = std::pair<std::size_t, std::size_t>;
using colors = std::vector<Color>;

/*!
  This type is an input to the coloring method, and encapsulates
  information (such as how many colors per axis the mesh needs to
  be partitioned into, if boundaries are periodic, etc), that is
  used by the coloring algorithm to create mesh partitions.
 */
struct coloring_definition {
  colors axis_colors; ///< number of colors into which each axis will be divided
  coord axis_extents; ///< extents to be partitioned
  coord axis_hdepths; ///< halo depth (or number of ghost layers) per axis
  coord axis_bdepths; ///< number of boundary layers to be added to the domain
                      ///< per axis
  std::vector<bool> axis_periodic; ///< specify which axes are periodic
  bool diagonals = false; ///< whether to include diagonally connected (i.e.,
                          ///< connected through vertex) as well as face
                          ///< connected entities during primary partitioning
  bool create_plan = true; ///< whether to create a copy plan
};

/*!
 Type to store the coloring information for one color.
 */
struct index_coloring {
  /// Flags to indicate position within the overall domain.
  /// Each axis gets two bits starting from the least significant:
  /// the lower indicates the low edge, and the higher the high edge.
  std::uint32_t faces;

  ///  The global extents.
  coord global;

  ///  The local extents of this color. This is the full size including
  ///  boundary depth, and ghosts. The "extents" coordinate implicitly
  ///  defines a hypercube from {0, 0, ..., 0} to extents{...}.
  coord extents;

  ///  The global coordinate offset of the logical hypercube.
  ///  Local to global id translation can be computed with this.
  coord offset;

  ///  The logical entities, i.e., the entities for this color without
  ///  boundary padding or ghosts.
  hypercube logical;

  ///  The extended entities, i.e., the logical entities including boundary
  ///  padding. The boundary depth can be computed like:
  ///    boundary_depth_low[axis] = logical[0][axis] - extended[0][axis];
  ///    boundary_depth_high[axis] = extended[1][axis] - logical[1][axis];
  ///  The ghost depth can be computed like:
  ///    shared_depth_low[axis] = logical[0][axis];
  ///    shared_depth_high[axis] = extents[axis] - logical[1][axis];
  ///  (note: We use logical to compute the ghost depth because an edge
  ///    cannot have both boundary padding, and ghosts.)
  hypercube extended;

  /// Offsets on the remote color.
  std::map<Color,
    std::vector<std::pair</* local ghost offset, remote shared offset */
      std::size_t,
      std::size_t>>>
    points;

  /// Local ghost intervals.
  std::vector<std::pair<std::size_t, std::size_t>> intervals;
};

/// \}
} // namespace narray_impl

/// \addtogroup narray
/// \{

/// \if core
/// Specialization-independent definitions.
/// \endif
struct narray_base {
  using index_coloring = narray_impl::index_coloring;
  using coord = narray_impl::coord;
  using hypercube = narray_impl::hypercube;
  using colors = narray_impl::colors;
  using coloring_definition = narray_impl::coloring_definition;

  /*!
   This range enumeration provides a classification of the various
   types of partition entities that can be requested out of a topology
   specialization created using this type. The following describes what each
   of the range enumeration means in a mesh part returned by the coloring
   algorithm. For the structured mesh partitioning, the partition info is
   specified per axis.

   These ranges are used in many of the interface methods to provide
   information such as size, extents, offsets about them.
  */
  enum class range : std::size_t {
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
    Color colors;
    std::vector<index_coloring> idx_colorings;
  }; // struct coloring

  static std::size_t idx_size(index_coloring const & ic, std::size_t) {
    std::size_t allocation{1};
    for(auto e : ic.extents) {
      allocation *= e;
    }
    return allocation;
  }
  /*!
   Method to compute the local ghost "intervals" and "points" which store map of
   local ghost offset to remote/shared offset. This method is called by the
   "make_copy_plan" method in the derived topology to create the copy plan
   objects.

   @param vpc vector of process colors
   @param[out] num_intervals vector of number of ghost intervals, over all
   colors, this vector is assumed to be sized correctly (all colors)
   @param[out] intervals  vector of local ghost intervals, over process colors
   @param[out] points vector of maps storing (local ghost offset, remote shared
   offset) for a shared color, over process colors
  */
  static void idx_itvls(index_coloring const & ic,
    std::vector<std::size_t> & num_intervals,
    MPI_Comm const & comm) {
    num_intervals = util::mpi::all_gather(ic.intervals.size(), comm);
  }

  // for make_copy_plan
  static void set_dests(field<data::intervals::Value>::accessor<wo> a,
    std::vector<std::pair<std::size_t, std::size_t>> const & intervals,
    MPI_Comm const &) {
    flog_assert(a.span().size() == intervals.size(), "interval size mismatch");
    std::size_t i{0};
    for(auto it : intervals) {
      a[i++] = data::intervals::make({it.first, it.second}, process());
    } // for
  }

  // for make_copy_plan
  template<PrivilegeCount N>
  static void set_ptrs(
    field<data::points::Value>::accessor1<privilege_repeat<wo, N>> a,
    std::map<Color, std::vector<std::pair<std::size_t, std::size_t>>> const &
      shared_ptrs,
    MPI_Comm const &) {
    for(auto const & si : shared_ptrs) {
      for(auto p : si.second) {
        // si.first: owner
        // p.first: local ghost offset
        // p.second: remote shared offset
        a[p.first] = data::points::make(si.first, p.second);
      } // for
    } // for
  }
}; // struct narray_base

/// \}
} // namespace topo

/*----------------------------------------------------------------------------*
  Serialization Rules
 *----------------------------------------------------------------------------*/

template<>
struct util::serial::traits<topo::narray_impl::index_coloring> {
  using type = topo::narray_impl::index_coloring;
  template<class P>
  static void put(P & p, const type & s) {
    serial::put(p,
      s.faces,
      s.global,
      s.extents,
      s.offset,
      s.logical,
      s.extended,
      s.points,
      s.intervals);
  }
  static type get(const std::byte *& p) {
    const cast r{p};
    return type{r, r, r, r, r, r, r, r};
  }
};

} // namespace flecsi

/// \endcond

#endif
