// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPOLOGY_HH
#define FLECSI_TOPOLOGY_HH

/// \defgroup topology Topologies
/// Generic topology categories and tools for specializing them.
/// \code#include "flecsi/topology.hh"\endcode
///
/// This header provides the features in the following additional \b deprecated
/// headers:
/// - \ref narray "flecsi/topo/narray/interface.hh"
/// - flecsi/topo/narray/coloring_utils.hh
///
/// \note In a \c toc task, certain metadata provided by topology accessors is
///   \e host-accessible, which means that if the topology accessor is
///   read-only it can be accessed within or outside of a kernel in the task.
///   (The metadata might be used with another accessor that is not
///   read-only.)
///
/// \warning The material in this section and its subsections other than
///   \ref spec is of interest
///   only to developers of topology specializations.  Application developers
///   should consult the documentation for the specializations they are using,
///   which may refer back to this document (occasionally even to small,
///   specific parts of this section).

#include "flecsi/topo/global.hh"
#include "flecsi/topo/index.hh"
#include "flecsi/topo/narray/interface.hh"
#if defined(FLECSI_ENABLE_LEGION)
#include "flecsi/topo/ntree/interface.hh"
#endif
#include "flecsi/topo/set/interface.hh"
#include "flecsi/topo/unstructured/interface.hh"

#endif
