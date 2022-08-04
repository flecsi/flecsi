// Copyright (c) 2016, Triad National Security, LLC
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

enum mask : uint32_t { interior = 0b00, low = 0b01, high = 0b10 };

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
struct process_color {
  /// Flags to indicate position within the overall domain.
  /// Each axis gets two bits starting from the least significant:
  /// the lower indicates the low edge, and the higher the high edge.
  std::uint32_t orientation;

  ///  The global extents.
  coord global;

  ///  The local extents of this color. This is the full size including
  ///  boundary depth, and ghosts. The "extents" coordinate implicitly
  ///  defines a hypercube from {0, 0, ..., 0} to extents{...}.
  coord extents;

  ///  The global coordinate offset of the local hypercube.
  ///  Local to global id translation can be computed with this.
  ///  The local hypercube includes ghosts but not boundary padding.
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
}; // struct process_color

inline std::ostream &
operator<<(std::ostream & stream, process_color const & ic) {
  stream << "global" << std::endl
         << flog::container{ic.global} << std::endl
         << "extents" << std::endl
         << flog::container{ic.extents} << std::endl
         << "offset" << std::endl
         << flog::container{ic.offset} << std::endl
         << "logical (low)" << std::endl
         << flog::container{ic.logical[0]} << std::endl
         << "logical (high)" << std::endl
         << flog::container{ic.logical[1]} << std::endl
         << "extended (low)" << std::endl
         << flog::container{ic.extended[0]} << std::endl
         << "extended (high)" << std::endl
         << flog::container{ic.extended[1]} << std::endl;
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
  using process_color = narray_impl::process_color;
  using coord = narray_impl::coord;
  using hypercube = narray_impl::hypercube;
  using colors = narray_impl::colors;
  using coloring_definition = narray_impl::coloring_definition;

  /// Coloring type.
  /// \ingroup narray
  struct coloring {
    MPI_Comm comm;
    Color colors;

    std::vector</* over index spaces */
      std::vector</* over global colors */
        std::size_t>>
      partitions;

    std::vector</* over index spaces */
      std::vector</* over process colors */
        process_color>>
      idx_colorings;
  }; // struct _coloring

  static std::size_t idx_size(std::vector<std::size_t> vs, std::size_t c) {
    return vs[c];
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
  static void idx_itvls(std::vector<process_color> const & vpc,
    std::vector<std::size_t> & num_intervals,
    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> & intervals,
    std::vector<std::map<Color,
      std::vector<std::pair<std::size_t, std::size_t>>>> & points,
    MPI_Comm const & comm) {
    auto [rank, size] = util::mpi::info(comm);

    std::vector<std::size_t> local_itvls;
    for(auto pc : vpc) {
      local_itvls.emplace_back(pc.intervals.size());
      intervals.emplace_back(std::move(pc.intervals));
      points.emplace_back(std::move(pc.points));
    }

    /*
      Gather global interval sizes.
     */

    auto global_itvls = util::mpi::all_gatherv(local_itvls, comm);

    std::size_t entities{1};
    for(auto a : vpc[0].global) {
      entities *= a;
    }
    util::color_map cm(size, num_intervals.size() /* colors */, entities);
    std::size_t p{0};
    for(auto pv : global_itvls) {
      std::size_t co{0};
      for(auto i : pv) {
        num_intervals[cm.color_id(p, co++)] = i;
      }
      ++p;
    }
  } // idx_itvls

  // for make_copy_plan
  static void set_dests(
    data::multi<field<data::intervals::Value>::accessor<wo>> aa,
    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> const &
      intervals,
    MPI_Comm const &) {
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
      std::vector<std::pair<std::size_t, std::size_t>>>> const & points,
    MPI_Comm const &) {
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

/*----------------------------------------------------------------------------*
  Serialization Rules
 *----------------------------------------------------------------------------*/

template<>
struct util::serial::traits<topo::narray_impl::process_color> {
  using type = topo::narray_impl::process_color;
  template<class P>
  static void put(P & p, const type & s) {
    serial::put(p,
      s.orientation,
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

#endif
