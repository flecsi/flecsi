.. |br| raw:: html

   <br />

CMake Configuration Options
===========================

The following set of options are available to control how FleCSI is
built.

Basic CMake options
-------------------

* **BUILD_SHARED_LIBS [default: ON]** |br|
  Build shared library objects (as opposed to static).

* **CMAKE_BUILD_TYPE [default: Debug]** |br|
  Specify the build type (configuration) statically for this build tree.
  Possible choices are *Debug*, *Release*, *RelWithDebInfo*, and
  *MinSizeRel*.

* **CMAKE_CXX_COMPILER [default: g++]** |br|
  Specify the C++ compiler to use to build FleCSI.

* **CMAKE_CXX_FLAGS [default: ""]** |br|
  Specify the flags to pass to the C++ compiler when building FleCSI.

* **CMAKE_INSTALL_PREFIX [default: /usr/local]** |br|
  Specify the installation path to use when *make install* is invoked.

Flog (FleCSI logging utility) options
-------------------------------------

* **ENABLE_FLOG [default: OFF]** |br|
  This option enables support for the FleCSI logging utility (Flog).
  When *OFF*, functions in the ``flog`` namespace
  and the ``flog`` macros
  remain defined but produce no output.

* **FLOG_ENABLE_COLOR_OUTPUT [default: OFF]** |br|
  Produce colorized Flog output instead of using the output device's
  default colors.

* **FLOG_SERIALIZATION_INTERVAL [default: 100]** |br|
  The Flog serialization interval specifies the number of task
  executions after which FleCSI should check for buffered output to
  process.  It should be set to a value that balances output
  timeliness (lower = more timely output) against performance (higher
  = less overhead from the requisite global reduction).

* **FLOG_SERIALIZATION_THRESHOLD** |br|
  **This interface is deprecated and not used anymore.**

* **FLOG_STRIP_LEVEL [default: 0]** |br|
  Set the Flog strip level, which should an integer from 0 to 4.  Like
  tag groups, strip levels are a mechanism for the user to control the
  amount of output that Flog generates: the higher the strip level,
  the fewer Flog messages will be output.  There are five strip levels
  in Flog:

  =====  =====
  Level  Type
  =====  =====
  0      trace
  1      info
  2      warn
  3      error
  4      fatal
  =====  =====

  Each number represents the largest value of ``FLOG_STRIP_LEVEL``
  that will produce that type of output.  That is, if the strip level
  is 0, all message types will be output; if the strip level is 3,
  only *error* and *fatal* log messages will be output. Regardless of
  the strip level, Flog messages that are designated *fatal* will
  generate a runtime error and will invoke ``std::exit``.

Parallelization options
-----------------------

* **ENABLE_KOKKOS [default: OFF]** |br|
  If enabled, support the use of `Kokkos <https://kokkos.org/>`_ for
  thread-level parallelism and GPU support.

* **ENABLE_MPI_CXX_BINDINGS [default: OFF]** |br|
  This option is a fall-back for codes that actually require the MPI C++
  bindings. **This interface is deprecated and should only be used if it
  is impossible to get rid of the dependency.**

* **ENABLE_OPENMP [default: OFF]** |br|
  Enable `OpenMP <https://www.openmp.org/>`_ pragmas for thread-level
  parallelism.  The appropriate flags will be passed to the C++
  compiler to enable language support for OpenMP.

* **FLECSI_BACKEND [default: legion]** |br|
  Specify the backend to use. Currently, *legion* and *mpi* are
  the only valid options.

Documentation options
---------------------

* **ENABLE_DOCUMENTATION [default: OFF]** |br|
  This option controls whether or not the FleCSI user- and
  developer-guide documentation is built. If enabled, CMake will
  generate these guides as PDFs in the *doc* subdirectory of the
  build.  To build the documentation, run

.. code-block:: console

  $ make doc

* **ENABLE_DOXYGEN [default: OFF]** |br|

  If enabled, CMake will verify that a suitable *doxygen* binary is
  available on the system and will add a target for generating
  Doxygen-style interface documentation from the FleCSI source code
  (``make doxygen``, which becomes a dependency of ``make doc``).

* **ENABLE_DOXYGEN_WARN [default: OFF]** |br|
  Normal Doxygen output produces many pages worth of warnings. These are
  distracting and overly verbose. As such, they are disabled by default.
  This options allows the user to turn them back on.

Miscellaneous options
---------------------

* **ENABLE_COVERAGE_BUILD [default: OFF]** |br|
  Enable build mode to determine the code coverage of the current set of
  unit tests. This is useful for continuous integration (CI) test analysis.

* **ENABLE_GRAPHVIZ [default: OFF]** |br|
  If enabled, support the use of `Graphviz <https://graphviz.org/>`_
  to produce graphical visualizations of a FleCSI program's control
  points and actions.

* **ENABLE_HDF5 [default: OFF]** |br|
  If enabled, support the use of `HDF5 <https://www.hdfgroup.org/>`_
  for checkpointing program state.

* **ENABLE_UNIT_TESTS [default: OFF]** |br|
  Enable FleCSI unit tests. If enabled, the unit test suite can be run
  by invoking:

.. code-block:: console

  $ make test

* **FLECSI_ID_TYPE [default: std::uint32_t]** |br|
  Specify the C++ type for FleCSI topologies to use for entity IDs.
