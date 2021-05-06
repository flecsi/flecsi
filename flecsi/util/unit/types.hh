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

struct assert_handler_t;

struct state_t {

  state_t(std::string name) {
    name_ = name;
  } // initialize

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

  int & result() {
    return result_;
  }

  const std::string & name() const {
    return name_;
  }

  std::stringstream & stringstream() {
    return error_stream_;
  } // stream

  template<class F>
  int operator->*(F f) { // highest binary precedence
    f();
    return result();
  }

  // Allows 'return' before <<:
  void operator>>=(const assert_handler_t &) const {}

private:
  int result_ = 0;
  std::string name_;
  std::stringstream error_stream_;

}; // struct state_t

struct assert_handler_t {

  assert_handler_t(const char * condition,
    const char * file,
    int line,
    state_t & runtime)
    : runtime_(runtime) {
    runtime_.result() = 1;
    runtime_.stringstream()
      << FLOG_OUTPUT_LTRED("ASSERT FAILED") << ": assertion '" << condition
      << "' failed in " << FLOG_OUTPUT_BROWN(file << ":" << line)
      << FLOG_COLOR_BROWN << " ";
  } // assert_handler_t

  ~assert_handler_t() {
    runtime_.stringstream() << FLOG_COLOR_PLAIN << std::endl;
  } // ~assert_handler_t

  template<typename T>
  assert_handler_t & operator<<(const T & value) {
    runtime_.stringstream() << value;
    return *this;
  } // operator <<

  assert_handler_t & operator<<(
    ::std::ostream & (*basic_manipulator)(::std::ostream & stream)) {
    runtime_.stringstream() << basic_manipulator;
    return *this;
  } // operator <<

private:
  state_t & runtime_;

}; // assert_handler_t

struct expect_handler_t {

  expect_handler_t(const char * condition,
    const char * file,
    int line,
    state_t & runtime)
    : runtime_(runtime) {
    runtime_.result() = 1;
    runtime_.stringstream()
      << FLOG_OUTPUT_YELLOW("EXPECT FAILED") << ": unexpected '" << condition
      << "' occurred in " << FLOG_OUTPUT_BROWN(file << ":" << line)
      << FLOG_COLOR_BROWN << " ";
  } // expect_handler_t

  ~expect_handler_t() {
    runtime_.stringstream() << FLOG_COLOR_PLAIN << std::endl;
  } // ~expect_handler_t

  template<typename T>
  expect_handler_t & operator<<(const T & value) {
    runtime_.stringstream() << value;
    return *this;
  } // operator <<

  expect_handler_t & operator<<(
    ::std::ostream & (*basic_manipulator)(::std::ostream & stream)) {
    runtime_.stringstream() << basic_manipulator;
    return *this;
  } // operator <<

private:
  state_t & runtime_;

}; // expect_handler_t

template<typename T1, typename T2>
inline bool
test_equal(const T1 & v1, const T2 & v2) {
  return v1 == v2;
}

template<typename T1, typename T2>
inline bool
test_less(const T1 & v1, const T2 & v2) {
  return v1 < v2;
}

template<typename T1, typename T2>
inline bool
test_less_equal(const T1 & v1, const T2 & v2) {
  return test_less(v1, v2) || test_equal(v1, v2);
}

template<typename T1, typename T2>
inline bool
test_greater(const T1 & v1, const T2 & v2) {
  return v1 > v2;
}

template<typename T1, typename T2>
inline bool
test_greater_equal(const T1 & v1, const T2 & v2) {
  return test_greater(v1, v2) || test_equal(v1, v2);
}

inline bool
string_compare(const char * lhs, const char * rhs) {
  if(lhs == nullptr) {
    return rhs == nullptr;
  }
  if(rhs == nullptr) {
    return false;
  }
  return strcmp(lhs, rhs) == 0;
} // string_compare

inline bool
string_case_compare(const char * lhs, const char * rhs) {
  if(lhs == nullptr) {
    return rhs == nullptr;
  }
  if(rhs == nullptr) {
    return false;
  }
  return strcasecmp(lhs, rhs) == 0;
} // string_case_compare

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

#define CHECK(ret, typ, condition, what)                                       \
  if(condition)                                                                \
    ;                                                                          \
  else                                                                         \
    ret ::flecsi::util::unit::typ##_handler_t(                                 \
      what, __FILE__, __LINE__, auto_unit_state)

#define ASSERT_TRUE(c) CHECK(return auto_unit_state >>=, assert, c, #c)
#define EXPECT_TRUE(c) CHECK(, expect, c, #c)

#define ASSERT_FALSE(c) ASSERT_TRUE(!(c))
#define EXPECT_FALSE(c) EXPECT_TRUE(!(c))

#define ASSERT_CMP(x, y, neg, f)                                               \
  ASSERT_TRUE(neg ::flecsi::util::unit::test_##f(x, y))
#define EXPECT_CMP(x, y, neg, f)                                               \
  EXPECT_TRUE(neg ::flecsi::util::unit::test_##f(x, y))

#define ASSERT_EQ(x, y) ASSERT_CMP(x, y, , equal)
#define EXPECT_EQ(x, y) EXPECT_CMP(x, y, , equal)
#define ASSERT_NE(x, y) ASSERT_CMP(x, y, !, equal)
#define EXPECT_NE(x, y) EXPECT_CMP(x, y, !, equal)
#define ASSERT_LT(x, y) ASSERT_CMP(x, y, , less)
#define EXPECT_LT(x, y) EXPECT_CMP(x, y, , less)
#define ASSERT_LE(x, y) ASSERT_CMP(x, y, , less_equal)
#define EXPECT_LE(x, y) EXPECT_CMP(x, y, , less_equal)
#define ASSERT_GT(x, y) ASSERT_CMP(x, y, , greater)
#define EXPECT_GT(x, y) EXPECT_CMP(x, y, , greater)
#define ASSERT_GE(x, y) ASSERT_CMP(x, y, , greater_equal)
#define EXPECT_GE(x, y) EXPECT_CMP(x, y, , greater_equal)

#define CHECK_STR(ret, typ, x, y, neg, f, op, sfx)                             \
  CHECK(ret, typ, neg ::flecsi::util::unit::f(x, y), a #op b sfx)

#define ASSERT_STR(x, y, neg, f, op, sfx)                                      \
  CHECK_STR(return auto_unit_state >>=, assert, neg, x, y, f, op, sfx)
#define EXPECT_STR(x, y, neg, f, op, sfx)                                      \
  CHECK_STR(, expect, neg, x, y, f, op, sfx)

#define ASSERT_STREQ(x, y) ASSERT_STR(x, y, , string_compare, ==, "")
#define EXPECT_STREQ(x, y) EXPECT_STR(x, y, , string_compare, ==, "")
#define ASSERT_STRNE(x, y) ASSERT_STR(x, y, !, string_compare, !=, "")
#define EXPECT_STRNE(x, y) EXPECT_STR(x, y, !, string_compare, !=, "")
#define ASSERT_STRCASEEQ(x, y)                                                 \
  ASSERT_STR(x, y, , string_case_compare, ==, " (case insensitive)")
#define EXPECT_STRCASEEQ(x, y)                                                 \
  EXPECT_STR(x, y, , string_case_compare, ==, " (case insensitive)")
#define ASSERT_STRCASENE(x, y)                                                 \
  ASSERT_STR(x, y, !, string_case_compare, !=, " (case insensitive)")
#define EXPECT_STRCASENE(x, y)                                                 \
  EXPECT_STR(x, y, !, string_case_compare, !=, " (case insensitive)")

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
