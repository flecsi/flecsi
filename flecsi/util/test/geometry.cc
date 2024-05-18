#include "flecsi/util/geometry/filling_curve_key.hh"
#include "flecsi/util/geometry/kdtree.hh"
#include "flecsi/util/geometry/point.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;

// Point Tests
using point_1d_t = util::point<double, 1>;
using point_2d_t = util::point<double, 2>;
using point_3d_t = util::point<double, 3>;

enum class axis : Dimension { x = 0, y = 1, z = 2 };

int
point_sanity() {
  UNIT() {
    constexpr point_1d_t a1{-1.0};
    static_assert(-1.0 == a1[axis::x]);

    constexpr point_2d_t a2{3.0, 0.0};
    static_assert(3.0 == a2[axis::x]);
    static_assert(0.0 == a2[axis::y]);

    float v1 = 3, v2 = 0;
    point_2d_t b2(v1, v2);

    b2 -= a2;
    ASSERT_EQ(b2[axis::x], 0.0);
    ASSERT_EQ(b2[axis::y], 0.0);

    b2 -= 1.0;
    ASSERT_EQ(b2[axis::x], -1.0);
    ASSERT_EQ(b2[axis::y], -1.0);

    b2 += a2;
    ASSERT_EQ(b2[axis::x], 2.0);
    ASSERT_EQ(b2[axis::y], -1.0);

    b2 += 2;
    ASSERT_EQ(b2[axis::x], 4.0);
    ASSERT_EQ(b2[axis::y], 1.0);

    b2 *= a2;
    ASSERT_EQ(b2[axis::x], 12.0);
    ASSERT_EQ(b2[axis::y], 0.0);

    b2 *= 4;
    ASSERT_EQ(b2[axis::x], 48.0);
    ASSERT_EQ(b2[axis::y], 0.0);

    constexpr point_3d_t a3{3.0, 0.0, -1.0};
    static_assert(3.0 == a3[axis::x]);
    static_assert(0.0 == a3[axis::y]);
    static_assert(-1.0 == a3[axis::z]);
  };
} // point_sanity

util::unit::driver<point_sanity> point_sanity_driver;

int
point_distance() {
  UNIT() {
    point_1d_t a1{1.0};
    point_1d_t b1{4.0};
    double d = distance(a1, b1);
    EXPECT_EQ(3.0, d) << "Distance calculation failed";

    point_2d_t a2{1.0, 2.0};
    point_2d_t b2{4.0, 6.0};
    d = distance(a2, b2);
    EXPECT_EQ(5.0, d) << "Distance calculation failed";

    point_3d_t a3{1.0, 2.0, -1.0};
    point_3d_t b3{4.0, 6.0, -1.0 - std::sqrt(11.0)};
    d = distance(a3, b3);
    EXPECT_EQ(6.0, d) << "Distance calculation failed";
  };
} // point_distance

util::unit::driver<point_distance> point_distance_driver;

int
point_midpoint() {
  UNIT() {
    point_1d_t a1{1.0};
    point_1d_t b1{4.0};
    point_1d_t c1 = midpoint(a1, b1);
    EXPECT_EQ(2.5, c1[0]) << "Midpoint calculation failed";

    point_2d_t a2{1.0, 2.0};
    point_2d_t b2{4.0, 6.0};
    point_2d_t c2 = midpoint(a2, b2);
    EXPECT_EQ(2.5, c2[0]) << "Midpoint calculation failed";
    EXPECT_EQ(4.0, c2[1]) << "Midpoint calculation failed";

    point_3d_t a3{1.0, 2.0, -1.0};
    point_3d_t b3{4.0, 6.0, -4.0};
    point_3d_t c3 = midpoint(a3, b3);
    EXPECT_EQ(2.5, c3[0]) << "Midpoint calculation failed";
    EXPECT_EQ(4.0, c3[1]) << "Midpoint calculation failed";
    EXPECT_EQ(-2.5, c3[2]) << "Midpoint calculation failed";
  };
} // point_midpoint

util::unit::driver<point_midpoint> point_midpoint_driver;

// Filling Curve Tests
using point_t = util::point<double, 3>;
using range_t = std::array<point_t, 2>;
using hc = util::hilbert_key<3, uint64_t>;
using mc = util::morton_key<3, uint64_t>;

using point_2d = util::point<double, 2>;
using range_2d = std::array<point_2d, 2>;
using hc_2d = util::hilbert_key<2, uint64_t>;
using mc_2d = util::morton_key<2, uint64_t>;

int
hilbert_sanity() {
  UNIT() {
    using namespace flecsi;

    range_t range;
    range[0] = {0, 0, 0};
    range[1] = {1, 1, 1};
    point_t p1 = {0.25, 0.25, 0.25};

    flog(info) << "Hilbert TEST " << hc::max_depth() << std::endl;

    hc hc1;
    hc hc2(range, p1);
    hc hc3 = hc::min();
    hc hc4 = hc::max();
    hc hc5 = hc::root();
    flog(info) << "Default: " << hc1 << std::endl;
    flog(info) << "Pt:rg  : " << hc2 << std::endl;
    flog(info) << "Min    : " << hc3 << std::endl;
    flog(info) << "Max    : " << hc4 << std::endl;
    flog(info) << "root   : " << hc5 << std::endl;
    EXPECT_EQ(1, hc5.value());

    while(hc4 != hc5) {
      hc4.pop();
    }
    EXPECT_EQ(hc5, hc4);
  };
} // hilbert_sanity

util::unit::driver<hilbert_sanity> hilbert_driver;

int
hilbert_2d_rnd() {
  UNIT() {
    using namespace flecsi;
    // Test the generation 2D
    range_2d rge;
    rge[0] = {0, 0};
    rge[1] = {1, 1};
    std::array<point_2d, 4> pts = {point_2d{.25, .25},
      point_2d{.25, .5},
      point_2d{.5, .5},
      point_2d{.5, .25}};
    std::array<hc_2d, 4> hcs_2d;

    for(int i = 0; i < 4; ++i) {
      hcs_2d[i] = hc_2d(rge, pts[i]);
      point_2d inv = hcs_2d[i].coordinates(rge);
      double dist = distance(pts[i], inv);
      flog(info) << pts[i] << " " << hcs_2d[i] << " = " << inv << std::endl;
      EXPECT_LT(dist, 1.0e-4);
    }
  };
} // hilbert_2d_rnd

util::unit::driver<hilbert_2d_rnd> hilbert_2d_rnd_driver;

int
hilbert_3d_rnd() {
  UNIT() {
    using namespace flecsi;
    // Test the generation
    range_t range;

    range[0] = {0, 0, 0};
    range[1] = {1, 1, 1};
    std::array<point_t, 8> points = {point_t{.25, .25, .25},
      point_t{.25, .25, .5},
      point_t{.25, .5, .5},
      point_t{.25, .5, .25},
      point_t{.5, .5, .25},
      point_t{.5, .5, .5},
      point_t{.5, .25, .5},
      point_t{.5, .25, .25}};
    std::array<hc, 8> hcs;

    for(int i = 0; i < 8; ++i) {
      hcs[i] = hc(range, points[i]);
      point_t inv = hcs[i].coordinates(range);
      // double dist = distance(points[i], inv);
      flog(info) << points[i] << " " << hcs[i] << " = " << inv << std::endl;
      // ASSERT_TRUE(dist < 1.0e-3);
    }

    // rnd
    for(int i = 0; i < 20; ++i) {
      point_t pt((double)rand() / (double)RAND_MAX,
        (double)rand() / (double)RAND_MAX,
        (double)rand() / (double)RAND_MAX);
      hc h(range, pt);
      point_t inv = h.coordinates(range);
      // double dist = distance(pt, inv);
      flog(info) << pt << " = " << h << " = " << inv << std::endl;
      // ASSERT_TRUE(dist < 1.0e-4);
    }
  };
} // hilbert_3d_rnd

util::unit::driver<hilbert_3d_rnd> hilbert_3d_rnd_driver;

int
morton_sanity() {
  UNIT() {
    range_t range;
    range[0] = {-1, -1, -1};
    range[1] = {1, 1, 1};
    point_t p1 = {0, 0, 0};

    flog(info) << " Morton TEST " << hc::max_depth() << std::endl;

    mc hc1;
    mc hc2(range, p1);
    mc hc3 = mc::min();
    mc hc4 = mc::max();
    mc hc5 = mc::root();
    flog(info) << "Default: " << hc1 << std::endl;
    flog(info) << "Pt:rg  : " << hc2 << std::endl;
    flog(info) << "Min    : " << hc3 << std::endl;
    flog(info) << "Max    : " << hc4 << std::endl;
    flog(info) << "root   : " << hc5 << std::endl;
    EXPECT_EQ(1, hc5.value());

    while(hc4 != hc5) {
      hc4.pop();
    }
    EXPECT_EQ(hc5, hc4);
  };
}

util::unit::driver<morton_sanity> morton_driver;

int
morton_2d_rnd() {
  UNIT() {
    using namespace flecsi;
    // Test the generation 2d
    range_2d rge;
    rge[0] = {0, 0};
    rge[1] = {1, 1};
    std::array<point_2d, 4> pts = {point_2d{.25, .25},
      point_2d{.5, .25},
      point_2d{.25, .5},
      point_2d{.5, .5}};
    std::array<mc_2d, 4> mcs_2d;

    for(int i = 0; i < 4; ++i) {
      mcs_2d[i] = mc_2d(rge, pts[i]);
      point_2d inv = mcs_2d[i].coordinates(rge);
      double dist = distance(pts[i], inv);
      flog(info) << pts[i] << " " << mcs_2d[i] << " = " << inv << std::endl;
      EXPECT_LT(dist, 1.0e-4);
    }

    // rnd
    for(int i = 0; i < 20; ++i) {
      point_2d pt(
        (double)rand() / (double)RAND_MAX, (double)rand() / (double)RAND_MAX);
      mc_2d h(rge, pt);
      point_2d inv = h.coordinates(rge);
      double dist = distance(pt, inv);
      flog(info) << pt << " = " << h << " = " << inv << std::endl;
      EXPECT_LT(dist, 1.0e-4);
    }
  };
} // morton_2d_rnd

util::unit::driver<morton_2d_rnd> morton_2d_rnd_driver;

int
morton_3d_rnd() {
  UNIT() {
    using namespace flecsi;
    range_t range;

    // Test the generation
    range[0] = {0, 0, 0};
    range[1] = {1, 1, 1};
    std::array<point_t, 8> points = {point_t{.25, .25, .25},
      point_t{.5, .25, .25},
      point_t{.25, .5, .25},
      point_t{.5, .5, .25},
      point_t{.25, .25, .5},
      point_t{.5, .25, .5},
      point_t{.25, .5, .5},
      point_t{.5, .5, .5}};
    std::array<mc, 8> mcs;

    for(int i = 0; i < 8; ++i) {
      mcs[i] = mc(range, points[i]);
      point_t inv = mcs[i].coordinates(range);
      double dist = distance(points[i], inv);
      flog(info) << points[i] << " " << mcs[i] << " = " << inv << std::endl;
      EXPECT_LT(dist, 1.0e-4);
    }

    // rnd
    for(int i = 0; i < 20; ++i) {
      point_t pt((double)rand() / (double)RAND_MAX,
        (double)rand() / (double)RAND_MAX,
        (double)rand() / (double)RAND_MAX);
      mc h(range, pt);
      point_t inv = h.coordinates(range);
      double dist = distance(pt, inv);
      flog(info) << pt << " = " << h << " = " << inv << std::endl;
      EXPECT_LT(dist, 1.0e-4);
    }
  };
} // morton_3d_rnd

util::unit::driver<morton_3d_rnd> morton_3d_rnd_driver;

// KDTree tests
int
kdtree() {
  UNIT() {
    { // 2D
      int nsrc = 4, ntrg = 3;

      using point_t = util::point<double, 2>;

      auto get_boxes = [](int N) {
        std::vector<util::BBox<2>> boxes;
        point_t lo, hi;
        double h = 1.0 / N;
        for(int i = 0; i < N; ++i) {
          for(int j = 0; j < N; ++j) {
            lo[0] = i * h;
            lo[1] = j * h;
            hi[0] = (i + 1) * h;
            hi[1] = (j + 1) * h;
            boxes.push_back({lo, hi});
          }
        }
        return boxes;
      };

      const auto candidates_map =
        util::KDTree<2>(get_boxes(ntrg)).intersect(get_boxes(nsrc));

      // reference
      std::set<long> ref_candidates[9] = {{0, 1, 4, 5},
        {1, 2, 5, 6},
        {2, 3, 6, 7},
        {4, 5, 8, 9},
        {5, 6, 9, 10},
        {6, 7, 10, 11},
        {8, 9, 12, 13},
        {9, 10, 13, 14},
        {10, 11, 14, 15}};

      const auto s = [](auto && r) {
        return std::set<long>(r.begin(), r.end());
      };

      for(int i = 0; i < ntrg * ntrg; ++i)
        EXPECT_EQ(s(candidates_map.at(i)), ref_candidates[i]);

      // list the candidates
      std::cout << "Candidates Map :\n";
      for(auto & cells : candidates_map) {
        std::cout << "For target cell " << cells.first << ", source cells = [";
        for(auto & c : cells.second)
          std::cout << "  " << c;
        std::cout << "]\n";
      }
    }

    { // 3D
      int nsrc = 4, ntrg = 3;

      using point_t = util::point<double, 3>;

      auto get_boxes = [](int N) {
        std::vector<util::BBox<3>> boxes;
        point_t lo, hi;
        double h = 1.0 / N;
        for(int i = 0; i < N; ++i) {
          for(int j = 0; j < N; ++j) {
            for(int k = 0; k < N; ++k) {
              lo[0] = i * h;
              lo[1] = j * h;
              lo[2] = k * h;
              hi[0] = (i + 1) * h;
              hi[1] = (j + 1) * h;
              hi[2] = (k + 1) * h;
              boxes.push_back({lo, hi});
            }
          }
        }
        return boxes;
      };

      const auto candidates_map =
        util::KDTree<3>(get_boxes(ntrg)).intersect(get_boxes(nsrc));

      // reference
      std::set<long> ref_candidates[9] = {{0, 1, 4, 5, 16, 17, 20, 21},
        {1, 2, 5, 6, 17, 18, 21, 22},
        {2, 3, 6, 7, 18, 19, 22, 23},
        {4, 5, 8, 9, 20, 21, 24, 25},
        {5, 6, 9, 10, 21, 22, 25, 26},
        {6, 7, 10, 11, 22, 23, 26, 27},
        {8, 9, 12, 13, 24, 25, 28, 29},
        {9, 10, 13, 14, 25, 26, 29, 30},
        {10, 11, 14, 15, 26, 27, 30, 31}};

      const auto ref = [&ref_candidates](int cell) {
        int l = cell / 9;
        auto & rc = ref_candidates[cell % 9];
        std::set<long> vals;
        for(auto & r : rc)
          vals.insert(r + 16 * l);
        return vals;
      };

      const auto s = [](auto && r) {
        return std::set<long>(r.begin(), r.end());
      };

      for(int i = 0; i < ntrg * ntrg * ntrg; ++i) {
        EXPECT_EQ(s(candidates_map.at(i)), ref(i));
      }
    }
  };
} // kdtree

util::unit::driver<kdtree> driver;
