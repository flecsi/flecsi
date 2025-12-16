// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_TAGS_HH
#define FLECSI_DATA_TAGS_HH

#include <cstddef>

namespace flecsi {
namespace data {
/// \addtogroup data
/// \{

/*!
  A type of logical data structure.
  The interface for each is provided by an \c accessor.
 */

enum layout : size_t {
  raw, ///< Uninitialized memory with no objects constructed or destroyed.
  single, ///< Access to the single element of an array.
  dense, ///< Ordinary array of objects.
  ragged, ///< Array of resizable arrays of objects.
  sparse, ///< Array of mappings from integers to objects.
  particle ///< Unordered elements are added/removed up to a maximum number.
};

/// \cond core

struct convert_tag {}; // must be recognized as a task argument

/// Task parameters of types that inherit from bind_tag must be specially
/// initialized by the backend after the task has been launched.
struct bind_tag {};

/// Classes that inherit from send_tag can decompose themselves into simpler
/// parameters via a send member function template.  This function template
/// accepts a callback that is used to process the subcomponents and which
/// itself accepts a callback that, on the caller side only, is used to
/// transform the task arguments.  Those task arguments may include
/// topo::borrow versions of the underlying topologies and field references
/// to such versions.
struct send_tag {};
/// \endcond

/// A class that inherits from params_tag is a composite task parameter.
/// The interface requires that the class provide a flecsi_params() member
/// function that returns a std::tie of all members. The class must be
/// constructible from the element types of the tuple returned by flecsi_params.
/// The corresponding argument is a tuple of task arguments for each element
/// type.
struct params_tag : bind_tag {};

/// A class that inherits from arg_tag is a custom task argument.
/// The interface requires that the class provide a flecsi_arg() member
/// function that returns a substitute task argument.
/// \warning \c flecsi_arg may be called multiple times on one task argument
///   as an rvalue or an lvalue (even after a call as an rvalue).
struct arg_tag : convert_tag {};

/// \}
} // namespace data

} // namespace flecsi

#endif
