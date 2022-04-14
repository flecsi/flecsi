// High-level topology type implementation.

#ifndef FLECSI_DATA_LEG_COPY_HH
#define FLECSI_DATA_LEG_COPY_HH

#include "flecsi/execution.hh"
#include "flecsi/topo/color.hh"

namespace flecsi::data {
namespace leg {
/// \addtogroup legion-data
/// \{

using rect = Legion::Rect<2>;

// Legion performs better with regions partitioned (completely) into disjoint
// subregions.  This internal topology stores two rectangles for each row;
// memory is allocated only for the first.
struct halves : topo::specialization<topo::color, halves> {
  using Field = flecsi::field<rect>;
  static const Field::definition<halves> field;
};
inline const halves::Field::definition<halves> halves::field;

// Produces pairs of abutting rectangles from single size inputs.
struct mirror {
  explicit mirror(size2 region);

  // Convert a prefixes_base::Field into a halves::Field.
  // Return the resulting two-subspace used/unused partition.
  template<class F>
  const partition<> & convert(F f) {
    execute<extend>(f, halves::field(rects), width);
    return part;
  }

  static constexpr const field_id_t & fid = halves::field.fid;

private:
  // Used to fill in mirror::columns with its completely predetermined pattern.
  static void fill(halves::Field::accessor<wo>, size_t c);
  static void extend(prefixes_base::Field::accessor<ro>,
    halves::Field::accessor<wo>,
    std::size_t width);

  halves::core rects, // n rows, each with its left and right rectangles
    columns; // 2x1 rectangles pointing to each column of rect
  partition<> part;
  std::size_t width;
};

struct with_partition {
  partition<> prt;
}; // for initialization order
/// \}
} // namespace leg

// Use inheritance to initialize mirror early:
struct prefixes : private leg::mirror,
                  private leg::with_partition,
                  leg::rows,
                  prefixes_base {
  template<class F>
  prefixes(region & reg, F f)
    : mirror(reg.size()),
      with_partition{{reg, convert(std::move(f)), fid, complete}},
      rows(
        [&] {
          auto r = get_first_subregion();
          const auto n =
            leg::name(reg.logical_region, "?") + std::string(1, '!');
          leg::run().attach_name(r, n.c_str());
          leg::run().attach_name(r.get_index_space(), n.c_str());
          return r;
        }(),
        reg.size()) {}

  template<class F>
  void update(F f) {
    auto p = prt.remake(convert(std::move(f)), fid, complete);
    rows::update(get_first_subregion(p));
    prt = std::move(p);
  }

  Legion::LogicalRegion get_first_subregion() const {
    return get_first_subregion(prt);
  }

private:
  static Legion::LogicalRegion get_first_subregion(
    const leg::partition<> & prt) {
    return leg::run().get_logical_subregion_by_color(prt.logical_partition, 0);
  }
};

// specialization of the partition constructor for prefixes being passed as a
// first argument
template<bool R>
leg::partition<R>::partition(prefixes & reg,
  const data::partition & src,
  field_id_t fid,
  completeness cpt)
  : partition(reg.get_first_subregion(),
      reg.get_first_subregion().get_index_space(),
      src,
      fid,
      cpt) {}

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
  void operator()(field_id_t f) const {
    copy_engine(*this).go(f);
  }
};

} // namespace flecsi::data

#endif
