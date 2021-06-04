// High-level topology type implementation.

#ifndef FLECSI_DATA_LEG_COPY_HH
#define FLECSI_DATA_LEG_COPY_HH

#include "flecsi/data/backend.hh" // don't use policy.hh directly
#include "flecsi/topo/color.hh"

namespace flecsi::data {
namespace leg {
using rect = Legion::Rect<2>;

struct halves : topo::specialization<topo::color, halves> {
  using Field = flecsi::field<rect>;
  static const Field::definition<halves> field;
};
inline const halves::Field::definition<halves> halves::field;

// Produces pairs of abutting rectangles from single size inputs.
struct mirror {
  explicit mirror(size2 region);

  // Convert a prefixes_base::Field into a halves::Field.
  // Return the partition to use to get each rectangle separately.
  template<class F>
  const partition<> & convert(F f) {
    execute<extend>(f, halves::field(rects), width);
    return part;
  }

  static constexpr const field_id_t & fid = halves::field.fid;

private:
  // Used to fill in mirror::order with its completely predetermined pattern.
  static void fill(halves::Field::accessor<wo>);
  static void extend(prefixes_base::Field::accessor<ro>,
    halves::Field::accessor<wo>,
    std::size_t width);

  halves::core rects, // n rows, each with its left and right rectangles
    order; // 2n degenerate rectangles pointing to each half-row
  partition<> part;
  std::size_t width;
};
} // namespace leg

// Use inheritance to initialize mirror early:
struct prefixes : private leg::mirror, leg::partition<>, prefixes_base {
  template<class F>
  prefixes(region & reg, F f)
    : mirror(reg.size()), partition(reg, convert(std::move(f)), fid) {}

  template<class F>
  void update(F f) {
    partition::update(convert(std::move(f)), fid);
  }
};

struct intervals : leg::partition<> {
  using Value = leg::rect;

  static Value make(subrow n,
    std::size_t i = run::context::instance().color()) {
    const Legion::coord_t r = i;
    const Legion::coord_t ln = n.first;
    return {{r, ln}, {r, leg::upper(n.second)}};
  }

  using partition::partition;
};

struct points : leg::partition<false> {
  using Value = Legion::Point<2>;

  static auto make(std::size_t r, std::size_t i) {
    return Value(r, i);
  }

  using partition::partition;
};

struct copy_engine {
  copy_engine(const points & src, const intervals & dest, field_id_t f)
    : copy_engine(src, dest.logical_partition, f) {}

private:
  copy_engine(const points & src, Legion::LogicalPartition dest, field_id_t f)
    : copy_engine(src, leg::run().get_parent_logical_region(dest), dest, f) {}
  copy_engine(const points & src,
    Legion::LogicalRegion lreg,
    Legion::LogicalPartition dest,
    field_id_t ptr_fid)
    : cl_(src.get_color_space()),
      src(src.logical_partition, leg::def_proj, READ_ONLY, EXCLUSIVE, lreg),
      dest(dest, leg::def_proj, WRITE_ONLY, EXCLUSIVE, lreg) {
    Legion::RegionRequirement rr_pos(
      dest, leg::def_proj, READ_ONLY, EXCLUSIVE, lreg);
    cl_.add_src_indirect_field(ptr_fid, rr_pos);
  }

  Legion::IndexCopyLauncher cl_;
  Legion::RegionRequirement src, dest;

  void go(field_id_t f) && {
    src.add_field(f);
    dest.add_field(f);
    cl_.add_copy_requirements(src, dest);
    leg::run().issue_copy_operation(leg::ctx(), cl_);
  }

public:
  void operator()(field_id_t f) const {
    copy_engine(*this).go(f);
  }
};

} // namespace flecsi::data

#endif
