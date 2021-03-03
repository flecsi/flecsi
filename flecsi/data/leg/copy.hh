// High-level topology type implementation.

#ifndef FLECSI_DATA_LEG_COPY_HH
#define FLECSI_DATA_LEG_COPY_HH

#include "flecsi/data/backend.hh" // don't use policy.hh directly

namespace flecsi::data {

struct prefixes : leg::partition<> {
  using row = Legion::Rect<2>;

  static row make_row(std::size_t i, std::size_t n) {
    const Legion::coord_t r = i;
    return {{r, 0}, {r, leg::upper(n)}};
  }
  static std::size_t row_size(const row & r) {
    return r.hi[1] - r.lo[1] + 1;
  }

  using leg::partition<>::partition;

  using leg::partition<>::update;
};

struct intervals : leg::partition<> {
  using Value = prefixes::row;

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
