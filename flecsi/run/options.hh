// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUN_OPTIONS_HH
#define FLECSI_RUN_OPTIONS_HH

#include "flecsi/run/context.hh"

#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/program_options.hpp>
#include <boost/smart_ptr/make_shared.hpp>

namespace flecsi {

/// \addtogroup runtime
/// \{

enum option_attribute : size_t {
  option_default,
  option_implicit,
  option_zero,
  option_multi
}; // enum

/*
  Auxilliary type for program options.
 */

using any = boost::any;

template<class>
struct program_option;

/*!
  Convert an option value into its underlying type.

  @tparam ValueType The option underlying value type.
  \deprecated Accept the value directly in the validation function.
 */

template<typename ValueType>
ValueType
option_value(any const & v) {
  return program_option<ValueType>::unwrap(v);
}

/*!
  The program_option type is a wrapper that implements a useful subset of
  Boost's Program Options utility. Creating an instance of this type at
  namespace scope will add a program option that can be queried after the
  \c initialize function is called.

  A validation function can have the signature
  `bool(ValueType,std::stringstream&)` or
  `bool(flecsi::any,std::stringstream&)`.  An error message may be written to
  the stream.  In the latter, \b deprecated case, use \link
  flecsi::option_value() `option_value`\endlink to obtain the \c ValueType.
 */

template<typename ValueType>
struct program_option {
  static const ValueType & unwrap(const boost::any & a) {
    return boost::any_cast<const boost::optional<ValueType> &>(a).value();
  }

private:
  static constexpr auto default_check = [](const ValueType &,
                                          std::stringstream &) { return true; };

  template<class F, class = void>
  struct wrapper {
    static auto get(F && f) {
      return [f = std::forward<F>(f)](const boost::any & a,
               std::stringstream & s) { return f(unwrap(a), s); };
    }
  };
  template<class F>
  struct wrapper<F,
    decltype(void(std::declval<F>()(std::declval<const boost::any &>(),
      std::declval<std::stringstream &>())))> {
    // for backward compatibility:
    // application should accept option value directly.
    static F && get(F && f) {
      return std::forward<F>(f);
    }
  };
  template<class F>
  static decltype(auto) wrap(F && f) {
    return wrapper<F>::get(std::forward<F>(f));
  }

  struct initializer_value
    : public std::pair<option_attribute, boost::optional<ValueType>> {

    initializer_value(option_attribute key) : initializer_value::pair(key, {}) {
      flog_assert(key == option_zero || key == option_multi,
        "invalid constructor for initializer_value: "
          << (key == option_default ? "option_default" : "option_implicit")
          << " must take a value");
    }
    initializer_value(option_attribute key, ValueType value)
      : initializer_value::pair(key, value) {
      flog_assert(key == option_default || key == option_implicit,
        "invalid constructor for initializer_value: "
          << (key == option_zero ? "option_zero" : "option_multi")
          << " does not take a value");
    }
  };

public:
  /*!
    Construct a program option.

    @param section The options description label.
    @param flag    The command-line option in long, short, or dual form,
                   e.g., "option,o" -> \em --option or \em -o.
    @param help    The help message for the option.
    @param values  Mechanism to set optional value attributes.  Supported keys
                   are \em option_default, \em option_implicit, \em
                   option_zero, and \em option_multi. If an \em option_default
                   value is specified, it will be used if the \em flag is not
                   passed to the command line. If an \em option_implicit value
                   is specified, it will be used if the flag is passed to the
                   command line without a value. If \em option_zero is passed,
                   the flag will not take any values, and must have an \em
                   option_implicit value specified. If \em option_multi is
                   passed, the flag will take multiple values.
    @param check   An optional, user-defined predicate to validate the option
                   passed by the user; see signature below.

    @code
      program_option<int> my_flag("My Section",
        "flag,f",
        "Specify the flag [0-9].",
        {
          {option_default, 1},
          {option_implicit, 0}
        },
        [](int value, std::stringstream &) {
          return value >= 0 && value < 10;
        });
    @endcode
   */
  template<class F = decltype((default_check))>
  program_option(const char * section,
    const char * flag,
    const char * help,
    std::initializer_list<initializer_value> values = {},
    F && check = default_check) {
    auto semantic_ = boost::program_options::value(&value_);

    bool zero{false}, implicit{false};
    for(auto const & v : values) {
      if(v.first == option_default) {
        semantic_->default_value(v.second);
      }
      else if(v.first == option_implicit) {
        semantic_->implicit_value(v.second);
        implicit = true;
      }
      else if(v.first == option_zero) {
        semantic_->zero_tokens();
        zero = true;
      }
      else if(v.first == option_multi) {
        semantic_->multitoken();
      }
      else {
        flog_fatal("invalid value type");
      } // if
    } // for

    if(zero) {
      flog_assert(implicit, "option_zero specified without option_implicit");
    } // if

    auto option =
      boost::make_shared<boost::program_options::option_description>(
        flag, semantic_, help);

    run::context::descriptions_map()
      .try_emplace(section, section)
      .first->second.add(option);

    std::string sflag(flag);
    sflag = sflag.substr(0, sflag.find(','));

    run::context::option_checks().try_emplace(
      sflag, false, wrap(std::forward<F>(check)));
  } // program_option

  /*!
    Construct a positional program option.

    @param name  The name for the positional option.
    @param help  The help message for the option.
    @param count The number of values to consume for this positional option. If
                 \em -1 is passed, this option will consume all remaining
                 values.
    @param check An optional, user-defined predicate to validate the option
                 passed by the user.
   */
  template<class F = decltype((default_check))>
  program_option(const char * name,
    const char * help,
    size_t count,
    F && check = default_check) {
    auto semantic_ = boost::program_options::value(&value_);
    semantic_->required();

    auto option =
      boost::make_shared<boost::program_options::option_description>(
        name, semantic_, help);

    using c = run::context;
    c::positional_description().add(name, count);
    c::positional_help().try_emplace(name, help);
    c::hidden_options().add(option);
    c::option_checks().try_emplace(name, true, wrap(std::forward<F>(check)));
  } // program_options

  /// Get the value, which must exist.
  ValueType value() const {
    return value_.value();
  }

  /// Get the value, which must exist.
  operator ValueType() const {
    return value();
  }

  /// Return whether the option was set.
  bool has_value() const {
    return value_.has_value();
  }

private:
  boost::optional<ValueType> value_{};

}; // struct program_option

/// Command-line parser.
/// The options defined by \c #initialize are not recognized.
struct getopt {
  /// Collect \c program_option objects.
  explicit getopt() : getopt({}) {}
  explicit getopt(run::config *); // lifetime-bound to argument

  /// Parse command-line arguments.
  /// Populate \c program_option objects with the results.
  /// \exception std::logic_error if parsing fails
  void operator()(int, const char * const *) const;
  auto parse(int, const char * const *) const;

  /// Return a string describing \c program_option objects.
  /// \param program name
  std::string usage(std::string_view p) const;

private:
  using desc = boost::program_options::options_description;
  desc basic{"Basic Options"}, all{"All Options"};
  std::optional<desc> flecsi;
};

/// \}

} // namespace flecsi

#endif
