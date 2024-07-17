// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

// High-level topology type implementation.

#ifndef FLECSI_DATA_LEG_COPY_HH
#define FLECSI_DATA_LEG_COPY_HH

#include "flecsi/execution.hh"
#include "flecsi/topo/color.hh"

namespace flecsi::data {
namespace leg {
/// \addtogroup legion-data
/// \{

struct used : topo::specialization<topo::column, used> {
  using Field = flecsi::field<rect, single>;
  static const Field::definition<used> field;
};
inline const used::Field::definition<used> used::field;

// Produces rectangles from sizes.
struct with_used {
  explicit with_used(Color n) : rects(n) {}

  // Convert a prefixes_base::Field into a used::Field.
  template<class F>
  const data::partition & convert(F f) {
    execute<extend>(f, used::field(rects));
    return rects;
  }

  static constexpr const field_id_t & fid = used::field.fid;

private:
  static void extend(prefixes_base::Field::accessor<ro>,
    used::Field::accessor<wo>);

  used::core rects;
};
/// \}
} // namespace leg

// Use inheritance to initialize with_used early:
struct prefixes : private leg::with_used, leg::partition<>, prefixes_base {
  template<class F>
  prefixes(region & reg, F f)
    : with_used(reg.size().first), partition(reg, convert(std::move(f)), fid) {}

  template<class F>
  void update(F f) {
    static_cast<partition &>(*this) = remake(convert(std::move(f)), fid);
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
    : copy_engine(src, dest.root(), dest.logical_partition, f) {}

private:
  copy_engine(const points & src,
    Legion::LogicalRegion lreg,
    Legion::LogicalPartition dest,
    field_id_t ptr_fid)
    : cl_(src.get_color_space()), src(src.logical_partition,
                                    leg::def_proj,
                                    LEGION_READ_ONLY,
                                    LEGION_EXCLUSIVE,
                                    lreg),
      dest(dest, leg::def_proj, LEGION_WRITE_ONLY, LEGION_EXCLUSIVE, lreg) {
    Legion::RegionRequirement rr_pos(
      dest, leg::def_proj, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lreg);
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
  void operator()(field_id_t f) {
    copy_engine(*this).go(f);
  }
};

} // namespace flecsi::data

#endif
