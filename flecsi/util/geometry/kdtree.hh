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

template<Dimension DIM>
class BBox
{
public:
  // Geometric point to represent coordinates
  using point_t = util::point<double, DIM>;

  // Size of the box
  constexpr point_t size() const {
    return upper - lower;
  }
  // Centroid
  constexpr point_t center() const {
    return (lower + upper) / 2;
  }

  // Determine if the two bounding boxes intersect.
  constexpr bool intersects(const BBox & box) const {
    for(Dimension d = 0; d < DIM; ++d) {
      if(upper[d] < box.lower[d] || lower[d] > box.upper[d])
        return false;
    }
    return true;
  }

  // Extend box as needed to contain a point.
  constexpr BBox & operator+=(const point_t & p) {
    minEqual(lower, p);
    maxEqual(upper, p);
    return *this;
  }

  // Union rhs into this BBox.
  constexpr void operator+=(const BBox & rhs) {
    minEqual(lower, rhs.lower);
    maxEqual(upper, rhs.upper);
  }

  //  These two vectors are the upper and lower corners of the box.
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

template<Dimension DIM>
int
max_component(util::point<double, DIM> & pnt) {
  return std::max_element(pnt.data()->begin(), pnt.data()->end()) -
         pnt.data()->begin();
}

template<Dimension DIM>
struct KDTree {
  // To store node info
  struct TNode {
    int cur_root;
    int imin;
    int imax;
    int icut;
  };

  using point_t = util::point<double, DIM>;
  using boxes = std::vector<BBox<DIM>>;
  KDTree(const boxes &);

  boxes sbox;
  std::vector<long> linkp;
};

/****************************************************************************/
/* Purpose        :KDTREE takes the set of Safety Boxes and                 */
/*                 produces a k-D tree that is stored in the array LINKP.   */
/*                 Leaf nodes in LINKP each coincide with exactly one       */
/*                 Safety Box.  For each node in the k-D tree,              */
/*                 there is a corresponding Safety Box which is just        */
/*                 big enough to contain all the Safety Boxes ``under''     */
/*                 the node.                                                */
/****************************************************************************/

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

using A2 = std::array<long, 2>;

template<Dimension DIM>
void
intersect(const KDTree<DIM> & k1,
  const KDTree<DIM> & k2,
  std::map<long, std::vector<long>> & candidates) {
  std::vector<A2> cnds;
  intersect(k1, k2, 0, 0, cnds);
  // add to map
  for(auto & c : cnds)
    candidates[c[1]].push_back(c[0]);

  // sort map entries
  for(auto & c : candidates)
    std::sort(c.second.begin(), c.second.end());
}

template<Dimension DIM>
void
intersect(const KDTree<DIM> & k1,
  const KDTree<DIM> & k2,
  std::vector<A2> & candidates) {
  intersect(k1, k2, 0, 0, candidates);
  std::sort(candidates.begin(), candidates.end(), [](A2 & a1, A2 & a2) {
    return a1[1] < a2[1];
  });
}

template<Dimension DIM>
void
intersect(const KDTree<DIM> & k1,
  const KDTree<DIM> & k2,
  int i1,
  int i2,
  std::vector<A2> & candidates) {
  std::vector<long> c1, c2;
  if(k1.sbox[i1].intersects(k2.sbox[i2])) {
    // We don't want to test every leaf of one tree against some large
    // safety box of the other tree that might intersect them even though
    // most of its contents do not, so split both trees simultaneously.
    auto l1 = k1.linkp[i1];
    auto l2 = k2.linkp[i2];

    if(l1 <= 0) { // box 1 is a leaf
      if(l2 <= 0) {
        candidates.push_back({-l1, -l2});
        return; // so is box 2
      }

      // can't split the leaf
      c1.push_back(i1);

      // but split the other
      c2.push_back(l2);
      c2.push_back(l2 + 1);
    }
    else {
      // split box 1
      c1.push_back(l1);
      c1.push_back(l1 + 1);

      // and box 2 if possible
      if(l2 <= 0) {
        c2.push_back(i2);
      }
      else {
        c2.push_back(l2);
        c2.push_back(l2 + 1);
      }
    }

    for(auto & ii : c1) {
      for(auto & jj : c2) {
        intersect(k1, k2, ii, jj, candidates);
      }
    }
  }
}
} // namespace util
} // namespace flecsi
#endif
