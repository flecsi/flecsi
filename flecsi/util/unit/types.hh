/*
    @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
   /@@/////  /@@          @@////@@ @@////// /@@
   /@@       /@@  @@@@@  @@    // /@@       /@@
   /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
   /@@////   /@@/@@@@@@@/@@       ////////@@/@@
   /@@       /@@/@@//// //@@    @@       /@@/@@
   /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
   //       ///  //////   //////  ////////  //

   Copyright (c) 2016, Triad National Security, LLC
   All rights reserved.
                                                                              */
#pragma once

#include <iostream>
#include <sstream>
#include <string>

#include "flecsi/util/demangle.hh"
#include "flecsi/util/unit/output.hh"
#include <flecsi/flog.hh>

namespace flecsi {

inline log::devel_tag unit_tag("unit");

namespace util {
namespace unit {

namespace detail {
template<class T, class = void>
struct maybe_value {
  static const char * get(const T &, const char * what) {
    return what;
  }
};
template<class T>
struct maybe_value<T,
  decltype(void(std::declval<std::ostream &>() << std::declval<const T &>()))> {
  static const T & get(const T & t, const char *) {
    return t;
  }
};
template<class T>
decltype(auto)
stream(const T & t, const char * what) {
  return maybe_value<T>::get(t, what);
}

template<class T>
struct not_fn { // default-constructed std::not_fn
  template<class... AA>
  bool operator()(AA &&... aa) const {
    return !T()(std::forward<AA>(aa)...);
  }
};
} // namespace detail

struct state_t {

  state_t(std::string name) {
    name_ = name;
  } // initialize
  state_t(state_t &&) = delete;

  ~state_t() {
    log::devel_guard guard(unit_tag);

    if(result_) {
      std::stringstream stream;
      stream << FLOG_OUTPUT_LTRED("TEST FAILED " << name_) << std::endl;
      stream << error_stream_.str();
      flog(utility) << stream.str();
    }
    else {
      flog(utility) << FLOG_OUTPUT_LTGREEN("TEST PASSED " << name_)
                    << FLOG_COLOR_PLAIN << std::endl;
    } // if
  } // process

  const std::string & name() const {
    return name_;
  }

  std::stringstream & stringstream() {
    return error_stream_;
  } // stream

  template<bool A, class... CC>
  void fail(const char * file, int line, const CC &... cc) {
    result_ = 1;
    if(A)
      error_stream_ << FLOG_OUTPUT_LTRED("ASSERT FAILED") << ": '";
    else
      error_stream_ << FLOG_OUTPUT_YELLOW("EXPECT FAILED") << ": '";
    (error_stream_ << ... << cc)
      << "' at " << FLOG_COLOR_BROWN << file << ':' << line << ' ';
  }
  template<bool A>
  bool test(bool b, const char * what, const char * file, int line) {
    if(!b)
      fail<A>(file, line, what);
    return b;
  }
  template<class C, bool A, class T, class U>
  bool compare(const T & t,
    const U & u,
    const char * ts,
    const char * op,
    const char * us,
    const char * sfx,
    const char * file,
    int line) {
    const bool ret = C()(t, u);
    if(!ret)
      fail<A>(file,
        line,
        '(',
        detail::stream(t, ts),
        ')',
        op,
        '(',
        detail::stream(u, us),
        ')',
        sfx);
    return ret;
  }

  template<class F>
  int operator->*(F && f) { // highest binary precedence
    std::forward<F>(f)();
    return result_;
  }

  // Allows 'return' before <<:
  void operator>>=(const std::ostream &) {
    error_stream_ << FLOG_COLOR_PLAIN << std::endl;
  }

private:
  int result_ = 0;
  std::string name_;
  std::stringstream error_stream_;

}; // struct state_t

struct string_compare {
  using not_fn = detail::not_fn<string_compare>;
  bool operator()(const char * lhs, const char * rhs) const {
    if(lhs == nullptr) {
      return rhs == nullptr;
    }
    if(rhs == nullptr) {
      return false;
    }
    return strcmp(lhs, rhs) == 0;
  }
};

struct string_case_compare {
  using not_fn = detail::not_fn<string_case_compare>;
  bool operator()(const char * lhs, const char * rhs) const {
    if(lhs == nullptr) {
      return rhs == nullptr;
    }
    if(rhs == nullptr) {
      return false;
    }
    return strcasecmp(lhs, rhs) == 0;
  }
};

} // namespace unit
} // namespace util
} // namespace flecsi

#define UNIT                                                                   \
  ::flecsi::log::state::instance().config_stream().add_buffer(                 \
    "flog", std::clog, true);                                                  \
  ::flecsi::util::unit::state_t auto_unit_state(__func__);                     \
  return auto_unit_state->*[&]() -> void

#define UNIT_TYPE(name) ::flecsi::util::demangle((name))

#define UNIT_TTYPE(type) ::flecsi::util::demangle(typeid(type).name())

#define CHECK(ret, f, ...)                                                     \
  if(auto_unit_state.f(__VA_ARGS__, __FILE__, __LINE__))                       \
    ;                                                                          \
  else                                                                         \
    ret auto_unit_state >>= auto_unit_state.stringstream()

#define ASSERT_TRUE(c) CHECK(return, test<true>, c, #c)
#define EXPECT_TRUE(c) CHECK(, test<false>, c, #c)

#define ASSERT_FALSE(c) ASSERT_TRUE(!(c))
#define EXPECT_FALSE(c) EXPECT_TRUE(!(c))

#define COMMA , // relies on CHECK invoking no other macros
#define CHECK_CMP(ret, cmp, A, x, y, op, sfx)                                  \
  CHECK(ret, compare<cmp COMMA A>, x, y, #x, #op, #y, sfx)

#define ASSERT_CMP(x, y, cmp, op, sfx)                                         \
  CHECK_CMP(return, cmp, true, x, y, op, sfx)
#define EXPECT_CMP(x, y, cmp, op, sfx) CHECK_CMP(, cmp, false, x, y, op, sfx)

#define ASSERT_EQ(x, y) ASSERT_CMP(x, y, ::std::equal_to<>, ==, "")
#define EXPECT_EQ(x, y) EXPECT_CMP(x, y, ::std::equal_to<>, ==, "")
#define ASSERT_NE(x, y) ASSERT_CMP(x, y, ::std::not_equal_to<>, !=, "")
#define EXPECT_NE(x, y) EXPECT_CMP(x, y, ::std::not_equal_to<>, !=, "")
#define ASSERT_LT(x, y) ASSERT_CMP(x, y, ::std::less<>, <, "")
#define EXPECT_LT(x, y) EXPECT_CMP(x, y, ::std::less<>, <, "")
#define ASSERT_LE(x, y) ASSERT_CMP(x, y, ::std::less_equal<>, <=, "")
#define EXPECT_LE(x, y) EXPECT_CMP(x, y, ::std::less_equal<>, <=, "")
#define ASSERT_GT(x, y) ASSERT_CMP(x, y, ::std::greater<>, >, "")
#define EXPECT_GT(x, y) EXPECT_CMP(x, y, ::std::greater<>, >, "")
#define ASSERT_GE(x, y) ASSERT_CMP(x, y, ::std::greater_equal<>, >=, "")
#define EXPECT_GE(x, y) EXPECT_CMP(x, y, ::std::greater_equal<>, >=, "")
#define ASSERT_STREQ(x, y)                                                     \
  ASSERT_CMP(x, y, ::flecsi::util::unit::string_compare, ==, "")
#define EXPECT_STREQ(x, y)                                                     \
  EXPECT_CMP(x, y, ::flecsi::util::unit::string_compare, ==, "")
#define ASSERT_STRNE(x, y)                                                     \
  ASSERT_CMP(x, y, ::flecsi::util::unit::string_compare::not_fn, !=, "")
#define EXPECT_STRNE(x, y)                                                     \
  EXPECT_CMP(x, y, ::flecsi::util::unit::string_compare::not_fn, !=, "")
#define ASSERT_STRCASEEQ(x, y)                                                 \
  ASSERT_CMP(x,                                                                \
    y,                                                                         \
    ::flecsi::util::unit::string_case_compare,                                 \
    ==,                                                                        \
    " (case insensitive)")
#define EXPECT_STRCASEEQ(x, y)                                                 \
  EXPECT_CMP(x,                                                                \
    y,                                                                         \
    ::flecsi::util::unit::string_case_compare,                                 \
    ==,                                                                        \
    " (case insensitive)")
#define ASSERT_STRCASENE(x, y)                                                 \
  ASSERT_CMP(x,                                                                \
    y,                                                                         \
    ::flecsi::util::unit::string_case_compare::not_fn,                         \
    !=,                                                                        \
    " (case insensitive)")
#define EXPECT_STRCASENE(x, y)                                                 \
  EXPECT_CMP(x,                                                                \
    y,                                                                         \
    ::flecsi::util::unit::string_case_compare::not_fn,                         \
    !=,                                                                        \
    " (case insensitive)")

// Provide access to the output stream to allow user to capture output
#define UNIT_CAPTURE()                                                         \
  ::flecsi::util::unit::test_output_t::instance().get_stream()

// Return captured output as a std::string
#define UNIT_DUMP() ::flecsi::util::unit::test_output_t::instance().get_buffer()

// Compare captured output to a blessed file
#define UNIT_EQUAL_BLESSED(f)                                                  \
  ::flecsi::util::unit::test_output_t::instance().equal_blessed((f))

// Write captured output to file
#define UNIT_WRITE(f)                                                          \
  ::flecsi::util::unit::test_output_t::instance().to_file((f))

// Dump captured output on failure
#if !defined(_MSC_VER)
#define UNIT_ASSERT(ASSERTION, ...)                                            \
  ASSERT_##ASSERTION(__VA_ARGS__) << UNIT_DUMP()
#else
  // MSVC has a brain-dead preprocessor...
#define UNIT_ASSERT(ASSERTION, x, y) ASSERT_##ASSERTION(x, y) << UNIT_DUMP()
#endif

// Dump captured output on failure
#if !defined(_MSC_VER)
#define UNIT_EXPECT(EXPECTATION, ...)                                          \
  EXPECT_##EXPECTATION(__VA_ARGS__) << UNIT_DUMP()
#else
  // MSVC has a brain-dead preprocessor...
#define UNIT_EXPECT(EXPECTATION, x, y) EXPECT_##EXPECTATION(x, y) << UNIT_DUMP()
#endif

// compare collections with varying levels of assertions
#define UNIT_CHECK_EQUAL_COLLECTIONS(...)                                      \
  ::flecsi::util::unit::CheckEqualCollections(__VA_ARGS__)

#define UNIT_ASSERT_EQUAL_COLLECTIONS(...)                                     \
  ASSERT_TRUE(::flecsi::util::unit::CheckEqualCollections(__VA_ARGS__) << UNIT_DUMP()

#define UNIT_EXPECT_EQUAL_COLLECTIONS(...)                                     \
  EXPECT_TRUE(::flecsi::util::unit::CheckEqualCollections(__VA_ARGS__))        \
    << UNIT_DUMP()
