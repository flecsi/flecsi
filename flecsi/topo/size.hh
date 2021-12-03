// Topology components used for storing sizes of other topologies.

#ifndef FLECSI_TOPO_SIZE_HH
#define FLECSI_TOPO_SIZE_HH

#include "flecsi/data/copy.hh"
#include "flecsi/topo/color.hh"

#include <cmath> // pow

namespace flecsi::topo {
/// \addtogroup topology
/// \{

// A subtopology for storing/updating row sizes of a partition.
struct resize : specialization<color, resize> {
  using Field = data::prefixes_base::Field;
  static const Field::definition<resize> field;
  template<partition_privilege_t P>
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
      : min(m), extra(e), lo(l), hi(h), slow(s ? std::pow(h / l, s) : 1) {}

    std::size_t operator()(std::size_t n, std::size_t cap) const {
      // n on [floor(lo*c),hi*c] preserves cap (although we have no mechanism
      // for a task to indicate to its caller the choice not to change it).
      const auto div = [n](float d) -> std::size_t {
        return std::nearbyint((n + .5f) / d);
      };
      return std::max(min,
        std::max(n + extra,
          n > hi * cap ? div(lo * slow)
                       : n >= std::size_t(lo * cap) ? cap : div(hi / slow)));
    }

  private:
    std::size_t min, extra;
    float lo, hi, slow;
  };
};
// Now that resize is complete:
inline const resize::Field::definition<resize> resize::field;

// To control initialization order:
struct with_size {
  explicit with_size(Color n, const resize::policy & p = {})
    : sz({n, 1}), growth(p) {}
  auto sizes() {
    return resize::field(sz);
  }
  resize::core sz;
  resize::policy growth;
};

/// \}
} // namespace flecsi::topo

#endif
