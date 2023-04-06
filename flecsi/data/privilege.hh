// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_PRIVILEGE_HH
#define FLECSI_DATA_PRIVILEGE_HH

#include "flecsi/util/bitutils.hh"

#include <cstddef>
#include <utility>

namespace flecsi {
/// \addtogroup data
/// \{

using Privileges = unsigned;
using PrivilegeCount = unsigned short;

/*!
  Access privileges for data passed
  to FleCSI tasks.

  Each field must be initialized with \c wo privileges (perhaps combined with
  \c na).  Any use of such privileges produces default-initialized values.

  Ghost data is updated only when read access to it is requested.
  Writes to shared data are never propagated to ghost data for which the same
  task has write access, as it is assumed to have updated them.
 */

enum partition_privilege_t : Privileges {
  na = 0b00, ///< no access: defer consistency update
  ro = 0b01, ///< read-only
  wo = 0b10, ///< write-only: consistency updates discarded
  rw = 0b11 ///< read-write
}; // enum partition_privilege_t

/// \cond core

inline constexpr short privilege_bits = 2;

// The leading tag bit which indicates the number of privileges in a pack:
template<PrivilegeCount N>
inline constexpr Privileges privilege_empty = [] {
  const Privileges ret = 1 << privilege_bits * N;
  static_assert(bool(ret), "too many privileges");
  return ret;
}();

/*!
  Utility to allow general privilege components that will match the old
  style of specifying permissions, e.g., <EX, SH, GH> (The old approach was
  only valid for mesh type topologies, and didn't make sense for all topology
  types).

  \tparam PP privileges
 */
template<partition_privilege_t... PP>
inline constexpr Privileges privilege_pack = [] {
  static_assert(((PP < 1 << privilege_bits) && ...));
  Privileges ret = 0;
  ((ret <<= privilege_bits, ret |= PP), ...);
  return ret | privilege_empty<sizeof...(PP)>;
}();

/*!
  Return the number of privileges stored in a privilege pack.

  \param PACK a \c privilege_pack value
 */

constexpr PrivilegeCount
privilege_count(Privileges PACK) {
  return (util::bit_width(PACK) - 1) / privilege_bits;
} // privilege_count

/*!
  Get a privilege out of a pack for the specified id.

  \param i privilege index
  \param pack a \c privilege_pack value
 */

constexpr partition_privilege_t
get_privilege(PrivilegeCount i, Privileges pack) {
  return partition_privilege_t(
    pack >> (privilege_count(pack) - 1 - i) * privilege_bits &
    ((1 << privilege_bits) - 1));
} // get_privilege

// Return whether the privilege allows reading _without_ writing first.
constexpr bool
privilege_read(partition_privilege_t p) {
  return p & 1;
}
constexpr bool
privilege_write(partition_privilege_t p) {
  return p & 2;
}

constexpr bool
privilege_read(Privileges pack) noexcept {
  for(auto i = privilege_count(pack); i--;)
    if(privilege_read(get_privilege(i, pack)))
      return true;
  return false;
}
constexpr bool
privilege_write(Privileges pack) noexcept {
  for(auto i = privilege_count(pack); i--;)
    if(privilege_write(get_privilege(i, pack)))
      return true;
  return false;
}

// Return whether the privileges destroy any existing data.
constexpr bool
privilege_discard(Privileges pack) noexcept {
  // With privilege_pack<na,wo>, the non-ghost can be read later.  With
  // privilege_pack<wo,na>, the ghost data is invalidated either by a
  // subsequent wo or by a ghost copy.
  auto i = privilege_count(pack);
  bool ghost = i > 1;
  for(; i--; ghost = false)
    switch(get_privilege(i, pack)) {
      case wo:
        break;
      case na:
        if(ghost)
          break; // else fall through
      default:
        return false;
    }
  return true;
}

// privilege_pack<P,P,...> (N times)
template<partition_privilege_t P, PrivilegeCount N>
inline constexpr Privileges
  privilege_repeat = privilege_empty<N> |
                     (privilege_empty<N> - 1) / ((1 << privilege_bits) - 1) * P;
template<Privileges A, Privileges B>
inline constexpr Privileges privilege_cat = [] {
  // Check for overflow:
  (void)privilege_empty<privilege_count(A) + privilege_count(B)>;
  const auto e = privilege_empty<privilege_count(B)>;
  return A * e | B & e - 1;
}();

constexpr partition_privilege_t
privilege_merge(Privileges p) {
  return privilege_discard(p) ? wo
         : privilege_write(p) ? rw
         : privilege_read(p)  ? ro
                              : na;
}

/// \endcond
/// \}
} // namespace flecsi

#endif
