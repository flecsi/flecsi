// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

// Topology components used for storing sizes of other topologies.

#ifndef FLECSI_TOPO_SIZE_HH
#define FLECSI_TOPO_SIZE_HH

#include "flecsi/data/copy.hh"
#include "flecsi/exec/future.hh"
#include "flecsi/topo/color.hh"

#include <cmath> // pow

namespace flecsi::topo {
/// \addtogroup topology
/// \{

/// \if core
/// A subtopology for storing/updating row sizes of a partition.
/// \else
/// Types for resizing partitions.
/// \endif
struct resize : specialization<column, resize> {
  /// \link flecsi::field `field`\endlink for storing sizes of a type
  /// that is assignable from and convertible to an integer
  using Field = data::prefixes_base::Field;
  static const Field::definition<resize> field;
  template<privilege P>
  using accessor = data::accessor_member<field, privilege_pack<P>>;

  /// A heuristic for automatically resizing a partition.
  /// Each new size is derived from the current size and amount of it used.
  /// \note Reducing the size may reduce data movement, but does not by itself
  ///   release memory.
  struct policy {
    /// Specify a policy.  The defaults always maintain the current size.
    /// The floating-point parameters control geometric reallocation.
    ///
    /// \param m the minimum size to use
    /// \param e the minimum extra space to reserve
    /// \param l fill fraction threshold for shrinking the size
    /// \param h fill fraction threshold for increasing the size
    /// \param s hysteresis control on [0,1]: larger values reallocate more
    ///   frequently for monotonic size changes but less frequently for
    ///   oscillatory ones
    policy(std::size_t m = 0,
      std::size_t e = 0,
      float l = 0,
      float h = 1,
      float s = 0)
      : min(m), extra(e), lo(l), hi(h), hyst(l ? s : 1) {}

    data::prefixes_base::size_request operator()(std::size_t n,
      std::size_t cap) const {
      const auto apply_slow = [](float hyst, float a, float b) -> float {
        return hyst == 0   ? b
               : hyst == 1 ? a
                           : std::pow(a, hyst) * std::pow(b, 1 - hyst);
      };

      const auto div = [](size_t sz, float d) -> std::size_t {
        return std::nearbyint((sz + .5f) / d);
      };

      std::size_t s;
      bool req;
      if(n > hi * cap) {
        s = div(n, apply_slow(hyst, hi, lo));
        req = true;
      }
      else if(const auto lo_thr = lo * cap; n < lo_thr) {
        auto d = apply_slow(hyst, lo, hi);
        s = div(n, d);
        req = div(lo_thr, d) >= std::max(min, std::size_t(lo_thr) + extra);
      }
      else {
        s = cap * std::pow(n / (cap * std::sqrt(hi * lo)), 2 * (1 - hyst));
        req = false;
      }
      const std::size_t clamp = std::max(min, n + extra);
      return {std::max(s, clamp), cap < clamp || (req && s != cap)};
    }

  private:
    std::size_t min, extra;
    float lo, hi, hyst;
  };
};
// Now that resize is complete:
inline const resize::Field::definition<resize> resize::field;

/// Size information for a partition.
struct with_size { // separate to control initialization order
  explicit with_size(scheduler & s, Color n, const resize::policy & p = {})
    : sz(s, n), growth(p), rsz_required(make_future(false)) {}
  /// Access the sizes.
  /// \return field reference for \c resize::Field
  auto sizes() {
    return resize::field(sz);
  }
  void set_rsz_required(bool r) {
    rsz_required = make_future(r);
  }
  resize::topology sz;
  /// Automatic growth control.
  resize::policy growth;
  future<bool> rsz_required;
};

/// \}
} // namespace flecsi::topo

#endif
