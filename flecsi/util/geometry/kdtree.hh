// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved

#ifndef FLECSI_UTIL_GEOMETRY_KDTREE_HH
#define FLECSI_UTIL_GEOMETRY_KDTREE_HH

#include "flecsi/util/common.hh"
#include "flecsi/util/geometry/point.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <stack>
#include <vector>

namespace flecsi {
namespace util {
/// \ingroup utils
/// \defgroup kdtree KDTree
/// KDTree based search utilities.
/// Possible use case is finding overlap between multiple meshes.
/// \{

/*!
 The bounding box type that is used by KDTree as input.
 */
template<Dimension DIM>
class BBox
{
public:
  /// Geometric point to represent coordinates
  using point_t = util::point<double, DIM>;

  /// Size of the box
  constexpr point_t size() const {
    return upper - lower;
  }

  /// Centroid
  constexpr point_t center() const {
    return (lower + upper) / 2;
  }

  /// Determine if the two bounding boxes intersect.
  constexpr bool intersects(const BBox & box) const {
    for(Dimension d = 0; d < DIM; ++d) {
      if(upper[d] < box.lower[d] || lower[d] > box.upper[d])
        return false;
    }
    return true;
  }

  /// Extend box as needed to contain a point.
  constexpr BBox & operator+=(const point_t & p) {
    minEqual(lower, p);
    maxEqual(upper, p);
    return *this;
  }

  /// Union rhs into this BBox.
  constexpr void operator+=(const BBox & rhs) {
    minEqual(lower, rhs.lower);
    maxEqual(upper, rhs.upper);
  }

  /// Return an empty BBox.
  constexpr static auto empty() {
    using nl = std::numeric_limits<double>;
    point_t lo, hi;
    for(Dimension d = 0; d < DIM; ++d) {
      lo[d] = nl::max();
      hi[d] = nl::lowest();
    }
    return BBox{lo, hi};
  }

  ///  The upper and lower corners of the box.
  point_t lower, upper;

private:
  constexpr static void minEqual(point_t & lhs, const point_t & rhs) {
    for(Dimension d = 0; d < DIM; ++d) {
      if(lhs[d] > rhs[d])
        lhs[d] = rhs[d];
    }
  }

  constexpr static void maxEqual(point_t & lhs, const point_t & rhs) {
    for(Dimension d = 0; d < DIM; ++d) {
      if(lhs[d] < rhs[d])
        lhs[d] = rhs[d];
    }
  }
};

// Find the index of a point with maximum component.
template<Dimension DIM>
int
max_component(util::point<double, DIM> & pnt) {
  return std::max_element(pnt.begin(), pnt.end()) - pnt.begin();
}

/*!
 A k-d tree for efficiently finding intersections between shapes.
 */
template<Dimension DIM>
struct KDTree {
  // To store node info
  struct TNode {
    // Current root
    int cur_root;
    // minimum index into the array of sorted keys
    int imin;
    // maximum index into the array of sorted keys
    int imax;
    // cutting direction at the current node
    int icut;
  };

  using point_t = util::point<double, DIM>;
  /// Type alias for a vector of Bounding Boxes
  using boxes = std::vector<BBox<DIM>>;
  /// The leaves of another tree that intersect each leaf of a given tree.
  using overlap = std::map<long, std::vector<long>>;

  KDTree(const boxes &);

  overlap intersect(const KDTree &) const;

private:
  boxes sbox; // including internal nodes
  std::vector<long> linkp;
};
/// \}

/*!
 Construct the tree from leaf bounding boxes.
 \param sboxp vector of safety boxes
*/

template<Dimension DIM>
KDTree<DIM>::KDTree(const boxes & sboxp)
  : sbox(2 * sboxp.size()), linkp(2 * sboxp.size()) {

  /* Compute the centers of the inputbounding boxes
   * and the root node of the k-D tree */
  std::vector<point_t> bbc(sboxp.size());
  int c = 0;
  for(auto & b : sboxp) {
    bbc[c++] = b.center();
    sbox[0] += b;
  }

  /* If there is only one safety box, the root node is a leaf.
     (Our convention is to set the link corresponding to a leaf
     equal to the negative of the unique item contained in
     that leaf.)  If the root is a leaf, our work is done. */

  if(sboxp.size() == 1)
    linkp[0] = 0;
  else {

    /* ipoly will contain a permutation of the integers
       {0,...,sboxp.size()-1}. This permutation will be altered as we
       create our balanced binary tree. nextp contains the
       address of the next available node to be filled. */

    std::vector<long> ipoly;
    for(int i = 0; i < (int)sboxp.size(); i++) {
      ipoly.push_back(i);
    }

    /* Use a stack to create the tree. Put the root node
       ``0'' on the top of the stack of TNodes. The array
       subset of ipoly (i.e., subset of boxes associated
       with this node) is recorded using imin and imax in TNode.

       We also record in the stack the appropriate ``cutting
       direction'' for bisecting the set of safety boxes
       corresponding to this node. This direction is either
       the x, y, or z directions, depending on which dimension
       of the bounding box is largest. */

    point_t cut_dim = sbox[0].size();
    std::stack<TNode> stk;
    stk.push({0, 0, (int)sboxp.size() - 1, max_component(cut_dim)});

    int nextp = 1;

    /* Pop nodes off stack, create children nodes and put them
       on stack. Continue until k-D tree has been created. */

    while(!stk.empty()) {

      /* Pop top node off stack. */
      auto node = stk.top();
      stk.pop();

      /* Make this node point to next available node location (nextp).
         This link represents the location of the FIRST CHILD
         of the node. The adjacent location (nextp+1)
         is implicitly taken to be the location of the SECOND
         child of the node. */

      linkp[node.cur_root] = nextp;
      int imn = node.imin;
      int imx = node.imax;
      int icut = node.icut;

      /* Partition safety box subset associated with this node.
         Using the appropriate cutting direction, use SELECT to
         reorder (ipoly) so that the safety box with median bounding
         box center coordinate is ipoly(imd), while the
         boxes {ipoly[i], i<imd} have SMALLER (or equal)
         bounding box coordinates, and the boxes with
         {ipoly[i], i>imd} have GREATER (or equal) bounding box
         coordinates. */

      int imd = (imn + imx) / 2;

      // Select the median bounding box using std::nth_element.
      auto comp = [&bbc, &icut](const long i, const long j) {
        return bbc[i][icut] < bbc[j][icut];
      };

      std::nth_element(
        ipoly.begin() + imn, ipoly.begin() + imd, ipoly.begin() + imx, comp);

      /* If the first child's subset of safety boxes is a singleton,
         the child is a leaf. Set the child's link to point to the
         negative of the box number. Set the child's bounding
         box to be equal to the safety box box. */

      if(imn == imd) {
        linkp[nextp] = -ipoly.at(imn);
        sbox[nextp] = sboxp[ipoly.at(imn)];
        nextp = nextp + 1;
      }
      else {

        /* In this case, the subset of safety boxes corres to the
           first child is more than one, and the child is
           not a leaf. Compute the bounding box of this child to
           be the smallest box containing all the associated safety'
           boxes. */

        sbox[nextp] = sboxp[ipoly.at(imn)];

        for(int i = imn + 1; i <= imd; i++)
          sbox[nextp] += sboxp[ipoly.at(i)];

        /* Put the first child onto the stack, noting the
           associated element subset in imin and imax, and
           putting the appropriate cutting direction in ict. */

        cut_dim = sbox[nextp].size();
        stk.push({nextp, imn, imd, max_component(cut_dim)});
        nextp++;
      }

      /* If the second child's subset of safety boxes is a singleton,
         the child is a leaf. Set the child's link to point to the
         negative of the sbox number. Set the child's bounding
         box to be equal to that of the safety box. */

      if((imd + 1) == imx) {
        linkp[nextp] = -ipoly.at(imx);
        sbox[nextp] = sboxp[ipoly.at(imx)];
        nextp = nextp + 1;
      }
      else {

        /* In this case, the subset of boxes corresponding to the
           second child is more than one safety box, and the child is
           not a leaf. Compute the bounding box of this child to
           be the smallest box containing all the associated safety
           boxes. */

        sbox[nextp] = sboxp[ipoly.at(imd + 1)];

        for(int i = imd + 2; i <= imx; i++)
          sbox[nextp] += sboxp[ipoly.at(i)];

        /* Put the second child onto the stack, noting the
           associated element subset in imin and imax, and
           putting the appropriate cutting direction in ict. */

        cut_dim = sbox[nextp].size();
        stk.push({nextp, imd + 1, imx, max_component(cut_dim)});
        nextp++;
      }

    } /* End of the while loop */
  }
}

/*!
 Find intersections between this tree and another.  A possible
 use case is finding the overlap between multiple distributed meshes .
 \return mapping from the this tree's leaves to vectors of intersecting leaves
   in \a k
 */
template<Dimension DIM>
typename KDTree<DIM>::overlap
KDTree<DIM>::intersect(const KDTree & k) const {
  overlap ret;

  const auto rec = [&](auto & f, int i1, int i2) -> void {
    if(sbox[i1].intersects(k.sbox[i2])) {
      // We don't want to test every leaf of one tree against some large
      // safety box of the other tree that might intersect them even though
      // most of its contents do not, so split both trees simultaneously.
      auto l1 = linkp[i1];
      auto l2 = k.linkp[i2];

      if(l1 <= 0) { // box 1 is a leaf
        if(l2 <= 0) // so is box 2
          ret[-l1].push_back(-l2);
        else {
          // split only the non-leaf:
          f(f, i1, l2);
          f(f, i1, l2 + 1);
        }
      }
      else {
        // split box 1
        for(long d = 0; d < 2; ++d)
          // and box 2 if possible
          if(l2 <= 0) {
            f(f, l1 + d, i2);
          }
          else {
            f(f, l1 + d, l2);
            f(f, l1 + d, l2 + 1);
          }
      }
    }
  };

  rec(rec, 0, 0);
  for(auto & [_, v] : ret)
    std::sort(v.begin(), v.end());

  return ret;
}
} // namespace util
} // namespace flecsi

#endif
