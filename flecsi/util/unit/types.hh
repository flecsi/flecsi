// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_UNIT_TYPES_HH
#define FLECSI_UTIL_UNIT_TYPES_HH

#include <ostream>
#include <sstream>
#include <string>

#include "flecsi/util/demangle.hh"
#include "flecsi/util/unit/output.hh"
#include <flecsi/flog.hh>

namespace flecsi {
namespace util {
namespace unit {
/// \addtogroup unit
/// \{

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
  FLECSI_TARGET bool operator()(AA &&... aa) const {
    return !T()(std::forward<AA>(aa)...);
  }
};
} // namespace detail

struct state_base {
  state_base() = default;
  state_base(state_base &&) = delete;

  template<class F>
  FLECSI_TARGET int operator->*(F && f) { // highest binary precedence
    std::forward<F>(f)();
    return result_;
  }

protected:
  int result_ = 0;
};

struct state_t : state_base {
  state_t(std::string name, std::string label) {
    name_ = name;
    label_ = label;
  } // initialize

  ~state_t() {
    if(result_) {
      std::stringstream stream;
      stream << (label_ == "TEST" ? FLOG_COLOR_LTRED : FLOG_COLOR_RED) << label_
             << " FAILED " << name_ << FLOG_COLOR_PLAIN << '\n';
      stream << error_stream_.str();
      flog(utility) << stream.str();
    }
    else
      flog(utility) << (label_ == "TEST" ? FLOG_COLOR_LTGREEN
                                         : FLOG_COLOR_GREEN)
                    << label_ << " PASSED " << name_ << FLOG_COLOR_PLAIN
                    << std::endl;
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

  // Allows 'return' before <<:
  void operator>>=(const std::ostream &) {
    error_stream_ << FLOG_COLOR_PLAIN << std::endl;
  }

private:
  std::string name_;
  std::string label_;
  std::stringstream error_stream_;

}; // struct state_t

struct string_compare {
  using not_fn = detail::not_fn<string_compare>;
  FLECSI_TARGET bool operator()(const char * lhs, const char * rhs) const {
    if(lhs == nullptr) {
      return rhs == nullptr;
    }
    if(rhs == nullptr) {
      return false;
    }
    return my_strcmp(lhs, rhs) == 0;
  }
  FLECSI_TARGET int my_strcmp(const char * lhs, const char * rhs) const {
    for(; *lhs == *rhs; ++lhs, ++rhs) {
      if(*lhs == '\0') {
        return 0;
      }
    }
    auto to_uc = [](const char * s) -> unsigned char { return *s; };
    return to_uc(lhs) < to_uc(rhs) ? -1 : 1;
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

// Provides the core of a GPU-capable unit test.  Intended to be used through
// the GPU_UNIT macro.  The state_t class (used through the UNIT macro) is
// preferred when running on the CPU, as it has more capabilities.  However,
// state_t/UNIT does not run on GPUs, so portable code needs to use
// gpu_state_t/GPU_UNIT instead.
struct gpu_state_t : state_base {
  FLECSI_TARGET gpu_state_t(const char * const name) : name_{name} {}
  FLECSI_TARGET ~gpu_state_t() {
    printf("TEST %s %s\n", (result_ == 0) ? "PASSED" : "FAILED", name_);
  }
  template<class Comparator, bool A, class T1, class T2>
  FLECSI_TARGET bool compare(const T1 & value1,
    const T2 & value2,
    const char * string1,
    const char * op,
    const char * string2,
    const char * suffix,
    const char * file,
    int line) {
    const bool ret = Comparator()(value1, value2);
    if(!ret) {
      printf("%s FAILED: '(%s)%s(%s)%s' at %s:%d\n",
        A ? "ASSERT" : "EXPECT",
        string1,
        op,
        string2,
        suffix,
        file,
        line);
      result_ = 1;
    }
    return ret;
  }
  template<bool A>
  FLECSI_TARGET bool
  test(bool value, const char * string, const char * file, int line) {
    if(!value) {
      printf("%s FAILED: '%s' at %s:%d\n",
        A ? "ASSERT" : "EXPECT",
        string,
        file,
        line);
      result_ = 1;
    }
    return value;
  }
  // For compatibility with state_t and ASSERT/EXPECT macros
  // -- Originally designed to allow the user to stream additional information
  //    into error messages.
  // -- Since std::ostream does not work on GPUs, for GPU_UNIT this returns the
  //    gpu_state_t instance itself (to help indicate where to look if the user
  //    tries to stream into this and gets a compiler error).
  FLECSI_TARGET const gpu_state_t & stringstream() {
    return *this;
  }
  FLECSI_TARGET void operator>>=(const gpu_state_t &) {}

private:
  const char * name_;
}; // gpu_state_t

/// \}
} // namespace unit
} // namespace util
} // namespace flecsi

/// \addtogroup unit
/// \{

inline std::string
label_default(std::string s) {
  return (s.empty() ? "TEST" : s);
}

/// Define a unit test function.  Should be followed by a compound statement,
/// which can use the other unit-testing macros, and a semicolon, and should
/// generally appear alone in a function that returns \c int.
/// Optionally, provide an expression convertible to \c std::string to label
/// the test results (along with \c __func__); the default is "TEST".
///
/// \note The `ASSERT`/`EXPECT` macros can be used in a lambda defined inside
///   the compound statement with `[&]`.
#define UNIT(...)                                                              \
  ::flecsi::util::unit::state_t auto_unit_state(                               \
    __func__, label_default({__VA_ARGS__}));                                   \
  return auto_unit_state->*[&]() -> void

/// \cond core
/// Alternative to UNIT that works on both CPU and GPU.  GPU_UNIT does not
/// support all features available through UNIT.
/// - GPU_UNIT() does not accept a custom label.
/// - GPU_UNIT does not work with the following macros:
///   - ASSERT_STRCASEEQ and EXPECT_STRCASEEQ
///   - ASSERT_STRCASENE and EXPECT_STRCASENE
///   - UNIT_CAPTURE
///   - UNIT_DUMP
///   - UNIT_BLESSED
///   - UNIT_WRITE
///   - UNIT_ASSERT
/// - You cannot stream additional information into the tests using << the way
///   you can with UNIT.
#define GPU_UNIT()                                                             \
  ::flecsi::util::unit::gpu_state_t auto_unit_state(__func__);                 \
  return auto_unit_state->*[&]() -> void
/// \endcond

#define CHECK(ret, f, ...)                                                     \
  if(auto_unit_state.f(__VA_ARGS__, __FILE__, __LINE__))                       \
    ;                                                                          \
  else                                                                         \
    ret auto_unit_state >>= auto_unit_state.stringstream()

/// \name Assertion macros
/// \{

#define ASSERT_TRUE(c) CHECK(return, test<true>, !!(c), #c)
#define EXPECT_TRUE(c) CHECK(, test<false>, !!(c), #c)

#define ASSERT_FALSE(c) ASSERT_TRUE(!(c))
#define EXPECT_FALSE(c) EXPECT_TRUE(!(c))

/// \}

#define COMMA , // relies on CHECK invoking no other macros
#define CHECK_CMP(ret, cmp, A, x, y, op, sfx)                                  \
  CHECK(ret, compare<cmp COMMA A>, x, y, #x, #op, #y, sfx)

#define ASSERT_CMP(x, y, cmp, op, sfx)                                         \
  CHECK_CMP(return, cmp, true, x, y, op, sfx)
#define EXPECT_CMP(x, y, cmp, op, sfx) CHECK_CMP(, cmp, false, x, y, op, sfx)

/// \name Assertion macros
/// Values to include in a failure message may be streamed into an
/// assertion:\code
/// EXPECT_GE(foo, 0) << foo << " should be non-negative";
/// \endcode
///
/// Macros that begin with \c ASSERT are identical to their \c EXPECT
/// counterparts except that they abandon the current \c UNIT (or innermost
/// lambda within it) on failure
/// (usually to avoid subsequent undefined behavior).
/// \{

/// Comparison.
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
/// Compare null-terminated strings, abandoning test on inequality.
#define ASSERT_STREQ(x, y)                                                     \
  ASSERT_CMP(x, y, ::flecsi::util::unit::string_compare, ==, "")
/// Check equality of null-terminated strings.
#define EXPECT_STREQ(x, y)                                                     \
  EXPECT_CMP(x, y, ::flecsi::util::unit::string_compare, ==, "")
/// Compare null-terminated strings, abandoning test on equality.
#define ASSERT_STRNE(x, y)                                                     \
  ASSERT_CMP(x, y, ::flecsi::util::unit::string_compare::not_fn, !=, "")
/// Check inequality of null-terminated strings.
#define EXPECT_STRNE(x, y)                                                     \
  EXPECT_CMP(x, y, ::flecsi::util::unit::string_compare::not_fn, !=, "")
/// Compare null-terminated strings, ignoring case and abandoning test on
/// inequality.
/// \deprecated The *_STRCASE* macros are deprecated.
#define ASSERT_STRCASEEQ(x, y)                                                 \
  ASSERT_CMP(x,                                                                \
    y,                                                                         \
    ::flecsi::util::unit::string_case_compare,                                 \
    ==,                                                                        \
    " (case insensitive)")
/// Check equality of null-terminated strings, ignoring case.
/// \deprecated The *_STRCASE* macros are deprecated.
#define EXPECT_STRCASEEQ(x, y)                                                 \
  EXPECT_CMP(x,                                                                \
    y,                                                                         \
    ::flecsi::util::unit::string_case_compare,                                 \
    ==,                                                                        \
    " (case insensitive)")
/// Compare null-terminated strings, ignoring case and abandoning test on
/// equality.
/// \deprecated The *_STRCASE* macros are deprecated.
#define ASSERT_STRCASENE(x, y)                                                 \
  ASSERT_CMP(x,                                                                \
    y,                                                                         \
    ::flecsi::util::unit::string_case_compare::not_fn,                         \
    !=,                                                                        \
    " (case insensitive)")
/// Check inequality of null-terminated strings, ignoring case.
/// \deprecated The *_STRCASE* macros are deprecated.
#define EXPECT_STRCASENE(x, y)                                                 \
  EXPECT_CMP(x,                                                                \
    y,                                                                         \
    ::flecsi::util::unit::string_case_compare::not_fn,                         \
    !=,                                                                        \
    " (case insensitive)")
/// \}

/// A stream that collects output for comparison.
#define UNIT_CAPTURE()                                                         \
  ::flecsi::util::unit::test_output_t::instance().get_stream()

/// Return captured output.
/// \return \c std::string
#define UNIT_DUMP() ::flecsi::util::unit::test_output_t::instance().get_buffer()

/// Compare captured output to a blessed file.
/// \return \c bool
#define UNIT_EQUAL_BLESSED(f)                                                  \
  ::flecsi::util::unit::test_output_t::instance().equal_blessed((f))

/// Write captured output to file.
#define UNIT_WRITE(f)                                                          \
  ::flecsi::util::unit::test_output_t::instance().to_file((f))

#if !defined(_MSC_VER)
/// Run an assertion and include captured output in any error message.
/// \param ASSERTION macro name (\c TRUE, \c EQ, \e etc.)
#define UNIT_ASSERT(ASSERTION, ...)                                            \
  ASSERT_##ASSERTION(__VA_ARGS__) << UNIT_DUMP()
#else
// MSVC has a brain-dead preprocessor...
#define UNIT_ASSERT(ASSERTION, x, y) ASSERT_##ASSERTION(x, y) << UNIT_DUMP()
#endif

/// \}

#endif
