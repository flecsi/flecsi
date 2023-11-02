// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_SET_INTERSECTION_HH
#define FLECSI_UTIL_SET_INTERSECTION_HH

namespace flecsi {
namespace util {
/// \addtogroup utils
/// \{

//!
//! \brief  Detect intersections of sorted lists.
//! \remark This function has complexity O(n + m)
//! \deprecated Unused.
template<class InputIt1, class InputIt2>
[[deprecated]] bool
intersects(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2) {
  while(first1 != last1 && first2 != last2) {
    if(*first1 < *first2) {
      ++first1;
      continue;
    }
    if(*first2 < *first1) {
      ++first2;
      continue;
    }
    return true;
  }
  return false;
}

/// \}
} // namespace util
} // namespace flecsi

#endif
