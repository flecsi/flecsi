#include "flecsi/data.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;
using namespace flecsi::data;

using mutator_t = typename field<std::size_t, ragged>::mutator<rw>;

int
ragged_mutator_driver() {
  UNIT() {
    // Declare backing storage and build the mutator
    std::vector<std::size_t> memory{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<std::size_t> endpoints{3, 9, 10, 10, 12};
    topo::resize::Field::value_type size = 1;
    mutator_t::TaskBuffer buffers;
    auto m = [&memory, &endpoints, &size, &buffers]() {
      // Declare the mutator
      field<std::size_t, ragged>::mutator<rw> m(mutator_t::base_type(0), {});
      // Bind the backing storage for data elements
      m.get_base().get_base().bind(util::span(memory));
      // Bind the backing storage for the endpoints
      m.get_base().get_offsets().bind(util::span(endpoints));
      // Bind the backing storage for the size
      m.get_size().get_base().get_base().bind(util::span(&size, 1));
      // Set up overflow storage and bind to mutator
      buffers.resize(endpoints.size());
      m.buffer(buffers);
      // Return the mutator
      return m;
    }();

    // Convenience for querying because I keep having off-by-one errors
    auto get_offset = [&endpoints](std::size_t irow) {
      return irow == 0 ? 0 : endpoints[irow - 1];
    };

    // Test size(), capacity()
    ASSERT_EQ(m.size(), endpoints.size());
    for(std::size_t irow{0}; irow < m.size(); ++irow) {
      auto row = m[irow];
      EXPECT_EQ(row.size(), get_offset(irow + 1) - get_offset(irow));
      EXPECT_EQ(row.capacity(), row.size());
    }

    // Test operator[]
    std::size_t n{0};
    for(std::size_t irow{0}; irow < m.size(); ++irow) {
      auto row = m[irow];
      for(std::size_t j{0}; j < row.size(); ++j) {
        EXPECT_EQ(&row[j], &memory[n]);
        ++n;
      }
    }

    // Test front() and back() (no buffer)
    for(std::size_t irow{0}; irow < m.size(); ++irow) {
      if(irow != 3) { // m[3] is empty so front() back() would fail
        auto const r = m[irow];
        EXPECT_EQ(&r.front(), &r[0]);
        EXPECT_EQ(&r.back(), &r[r.size() - 1]);
      }
    }

    // Test empty()
    for(std::size_t irow{0}; irow < m.size(); ++irow) {
      EXPECT_EQ(m[irow].empty(), irow == 3); // only m[3] empty
    }

    // Test clear()
    m[0].clear();
    EXPECT_EQ(m[0].size(), 0);

    // Test insert
    // -- insert(iterator pos, const T & value)
    std::size_t const v0 = 120;
    m[0].insert(m[0].begin(), v0);
    ASSERT_EQ(m[0].size(), 1);
    EXPECT_EQ(m[0].front(), 120);
    EXPECT_EQ(&m[0].front(), &memory[0]);
    // -- insert(iterator pos, T && value)
    std::size_t const v1 = 100;
    m[0].insert(m[0].begin(), std::move(v1));
    ASSERT_EQ(m[0].size(), 2);
    EXPECT_EQ(m[0].front(), 100);
    EXPECT_EQ(&m[0].front(), &memory[0]);
    EXPECT_EQ(&m[0].back(), &memory[1]);
    // -- insert(iterator pos, size_type count, const T & value)
    m[4].insert(std::next(m[4].begin()), 4, 120);
    ASSERT_EQ(m[4].size(), 6);
    std::vector<std::size_t> expected_m4{10, 120, 120, 120, 120, 11};
    EXPECT_TRUE(std::equal(m[4].begin(), m[4].end(), expected_m4.begin()));
    EXPECT_EQ(&m[4][0], &memory[10]);
    EXPECT_EQ(&m[4][1], &memory[11]);
    ASSERT_EQ(buffers[4].buffer.size(), 4);
    EXPECT_EQ(buffers[4].buffer, (std::vector<std::size_t>{120, 120, 120, 11}));
    // -- insert(iterator pos, I first, I last)
    std::vector<std::size_t> v({100, 101, 102});
    m[3].insert(m[3].begin(), v.begin(), v.end());
    ASSERT_EQ(m[3].size(), 3);
    // -- insert(iterator pos, std::initializer_list)
    m[3].insert(m[3].end(), {103, 104, 105});
    ASSERT_EQ(m[3].size(), 6);
    for(std::size_t n{0}; n < m[3].size(); ++n) {
      EXPECT_EQ(m[3][n], 100 + n);
      EXPECT_EQ(buffers[3].buffer[n], 100 + n);
    }

    // Test assign
    // -- assign(I first, I last)
    m[4].assign(v.begin(), v.end());
    ASSERT_EQ(m[4].size(), 3);
    EXPECT_EQ(m[4][0], 100);
    EXPECT_EQ(m[4][1], 101);
    EXPECT_EQ(m[4][2], 102);
    EXPECT_EQ(&m[4][0], &memory[10]);
    EXPECT_EQ(&m[4][1], &memory[11]);
    EXPECT_EQ(buffers[4].buffer[0], 102);
    // -- assign(std::initializer_list)
    m[2].assign({200, 201});
    ASSERT_EQ(m[2].size(), 2);
    EXPECT_EQ(m[2][0], 200);
    EXPECT_EQ(m[2][1], 201);
    EXPECT_EQ(&m[2][0], &memory[9]);
    EXPECT_EQ(buffers[2].buffer[0], 201);
    // -- assign(size_type count, const T & value)
    m[2].assign(3, 120);
    ASSERT_EQ(m[2].size(), 3);
    EXPECT_EQ(m[2][0], 120);
    EXPECT_EQ(m[2][1], 120);
    EXPECT_EQ(m[2][2], 120);
    EXPECT_EQ(&m[2][0], &memory[9]);
    ASSERT_EQ(buffers[2].buffer.size(), 2);
    EXPECT_EQ(buffers[2].buffer[0], 120);
    EXPECT_EQ(buffers[2].buffer[1], 120);

    // Test emplace
    m[0].emplace(std::next(m[0].begin()), 99);
    ASSERT_EQ(m[0].size(), 3);
    EXPECT_EQ(m[0][0], 100);
    EXPECT_EQ(m[0][1], 99);
    EXPECT_EQ(m[0][2], 120);
    EXPECT_EQ(&m[0][0], &memory[0]);
    EXPECT_EQ(&m[0][1], &memory[1]);
    EXPECT_EQ(&m[0][2], &memory[2]);

    // Test erase
    // -- erase(iterator pos)
    auto row_m1 = m[1];
    ASSERT_GT(row_m1.size(), 3);
    auto it = row_m1.begin() + 3;
    EXPECT_EQ(&*it, &row_m1[3]);
    EXPECT_EQ(&*it, &memory[6]);
    m[1].erase(it);
    std::vector<std::size_t> expected_values{3, 4, 5, 7, 8};
    auto check_m1 = [&]() {
      ASSERT_EQ(m[1].size(), expected_values.size());
      for(std::size_t n{0}; n < expected_values.size(); ++n) {
        EXPECT_EQ(m[1][n], expected_values[n]);
        EXPECT_EQ(&m[1][n], &memory[n + 3]);
      }
    };
    check_m1();
    // -- erase(iterator first, iterator last)
    m[1].erase(std::next(m[1].begin()), std::prev(m[1].end()));
    ASSERT_EQ(m[1].size(), 2);
    EXPECT_EQ(m[1][0], 3);
    EXPECT_EQ(m[1][1], 8);
    EXPECT_EQ(&m[1][0], &memory[3]);
    EXPECT_EQ(&m[1][1], &memory[4]);

    // Test push_back(), emplace_back(), pop_back()
    // -- push_back(const T & t)
    m[1].push_back(120);
    expected_values = {3, 8, 120};
    check_m1();
    // -- push_back(T && t)
    m[1].push_back(125);
    expected_values.push_back(125);
    check_m1();
    // -- emplace_back
    m[1].emplace_back(255);
    expected_values.emplace_back(255);
    check_m1();
    // -- pop_back
    m[1].pop_back();
    expected_values.pop_back();
    check_m1();

    // Test resize
    // -- resize(size_type count)
    m[2].resize(1);
    ASSERT_EQ(m[2].size(), 1);
    m[2].resize(2);
    ASSERT_EQ(m[2].size(), 2);
    EXPECT_EQ(m[2][0], 120); // resize does not overwrite existing values
    EXPECT_EQ(m[2][1], std::size_t());
    // -- resize(size_type count, const T & value)
    m[2].resize(3, 50);
    ASSERT_EQ(m[2].size(), 3);
    EXPECT_EQ(m[2][0], 120);
    EXPECT_EQ(m[2][1], std::size_t());
    EXPECT_EQ(m[2][2], 50);

    // Test commit
    // -- erase some elements so the new data fits in the available storage
    m[0].clear();
    m[3].erase(m[3].begin() + 3, m[3].end());
    m[4].erase(m[4].begin());
    std::vector<size_t> expected_sizes = {0, 4, 3, 3, 2};
    m.commit();
    for(std::size_t irow{0}; irow < endpoints.size(); ++irow) {
      ASSERT_EQ(m[irow].size(), expected_sizes[irow]);
    }
    EXPECT_EQ(memory,
      (std::vector<std::size_t>{
        3, 8, 120, 125, 120, std::size_t(), 50, 100, 101, 102, 101, 102}));
  };
}

util::unit::driver<ragged_mutator_driver> driver2;

#if FLECSI_BACKEND == FLECSI_BACKEND_mpi

int
storage_test() {
  UNIT() {
    mpi::detail::storage s;
    s.resize(sizeof(int)); // resizes toc and loc
    *reinterpret_cast<int *>(
      s.data<exec::task_processor_type_t::loc, partition_privilege_t::rw>()) =
      2;
    s.resize(sizeof(int) * 2); // resizes only loc

    // Invoke transfer_return where ret.size() < sync.size()
    s.data<exec::task_processor_type_t::toc, partition_privilege_t::rw>();

    s.resize(sizeof(int)); // resizes only toc
    // now size<toc> == 1 and size<loc> == 2,
    // so this invokes transfer_return where ret.size() > sync.size()
    EXPECT_EQ(
      *reinterpret_cast<int *>(
        s.data<exec::task_processor_type_t::loc, partition_privilege_t::rw>()),
      2);
  };
}

util::unit::driver<storage_test> driver3;

#endif
