This document describes the FleCSI C++ API.
The [high-level documentation](../../index.html) provides an introduction to the design and usage of the library more suitable to the new reader or developer as well as full examples for context.

The application layer provides a portable interface for defining
and executing tasks and kernels, defining and accessing field data,
utilities for creating command-line options, a logging utility (flog),
an interface for performance analysis using
[Caliper](http://software.llnl.gov/Caliper), and an I/O interface for
analysis and checkpoint/restart.

The topology layer provides several core FleCSI topology
types, along with utilities designed to aid in the creation of
application specialization libraries.

\if core
\warning This version includes internal interfaces.
Specialization and application developers should use only the interfaces documented in the [user API reference](../user/index.html), as there is no guarantee that those that appear only here will remain stable, _e.g._, a type or interface may simply
be removed or changed.
\endif

<!-- vim: set tabstop=2 shiftwidth=2 expandtab fo=cqt tw=72 : -->
