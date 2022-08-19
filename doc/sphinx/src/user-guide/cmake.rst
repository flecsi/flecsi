CMake for FleCSI client applications
####################################

The FleCSI installation provides mutliple CMake files to support
writing new software.  While the ``FleCSI`` CMake package is used to
build software on top of FleCSI, the ``FleCSICMake`` package files
provide common CMake code and macros for adding dependencies, building
documentation, unit-testing and other utilities that might be useful
for clients.

Both CMake packages are installed either when you build and install
FleCSI manually via CMake to ``CMAKE_INSTALL_PREFIX`` or when using a
Spack installation.

If you installed FleCSI in a Spack environment the necessary
environment variables will be automatically set.

After manual compilation and installation of FleCSI via CMake, you will
have to prepend the installation prefix to the ``CMAKE_PREFIX_PATH``
environment variable to make it available.

.. code-block:: bash

   export CMAKE_PREFIX_PATH=/path/to/your/flecsi/install:$CMAKE_PREFIX_PATH

---------------------------------------------------------------------------

FleCSI CMake files
==================

FleCSI-based applications often need common CMake code to enable
features and/or add dependencies. The ``FleCSICMake`` package exposes
utility CMake files that are meant to be included in new projects. It
does so by adding its installation parent folder to the CMake's search
path when you add the package:

.. code-block:: cmake

   find_package(FleCSICMake REQUIRED)

Once added, you can ``include`` the CMake files provided by this
package.

For example, if you want to use the provided ``documentation.cmake``,
you can add it as follows:

.. code-block:: cmake

   include(FleCSICMake/documentation)

.. note::

   These utilities themselves depend on CMake files provided by the
   `lanl-cmake-modules <https://github.com/lanl/cmake-modules>`_ project, which
   is a Spack dependency of FleCSI.

Documentation
-------------

FleCSI uses both Sphinx and Doxygen for its documentation system. The
following CMake files can be included to utilize the same
documentation system for your own projects.

``documentation``
   Adds the ``ENABLE_DOCUMENTATION`` CMake option, which is used to
   conditionally enable the Sphinx and Doxygen options if included. It
   also defines a ``doc`` target, which will collect all documentation
   targets. ``make doc`` will build all defined sphinx and doxygen
   targets.

   It also adds the advanced CMake option ``GITHUB_PAGES_REPO``. If
   ``GITHUB_PAGES_REPO`` is set, this adds the ``deploy-docs``
   target. This will checkout the ``gh-pages`` branch from the Git
   repository defined in ``GITHUB_PAGES_REPO``.  The checkout will be
   cleared and the result of the ``doc`` target is copied into
   it. Files are only added, but **not** commited or pushed. These are
   left as manual steps.

``sphinx``
  Adds the ``ENABLE_SPHINX`` CMake option, which is only shown if
  ``ENABLE_DOCUMENTATION`` is ``ON``. If enabled, it will add a
  ``sphinx`` target, which collects all defined sphinx targets. The
  ``sphinx`` target is added as dependency of the ``doc`` target.

  Sphinx builds are then defined with the ``add_sphinx_target``
  function. It takes three parameters: the target name, ``CONFIG``
  directory, and ``OUTPUT`` directory. The provided ``CONFIG``
  directory must contain a ``conf.py.in`` file. This template will be
  processed with ``configure_file`` to produce the final Sphinx
  configuration file ``conf.py``. This configuration file as well as
  the ``_static`` and ``_templates`` directories will be copied to the
  ``OUTPUT`` directory, where the target will be built.

  Usage:

  .. code-block:: cmake

     include(documentation)
     include(sphinx)

     add_sphinx_target(main                     # custom target name
       CONFIG ${CMAKE_SOURCE_DIR}/doc/sphinx    # folder containing conf.py.in
       OUTPUT ${CMAKE_BINARY_DIR}/doc           # output directory
     )

``doxygen``
  Adds the ``ENABLE_DOXYGEN`` and ``ENABLE_DOXYGEN_WARN`` CMake
  options. Both are only shown if ``ENABLE_DOCUMENTATION`` is
  ``ON``. If enabled, it will add a ``doxygen`` target, which collects
  all defined doxygen targets. The ``doxygen`` target is added as
  dependency of the ``doc`` target.

  Doxygen builds are defined with the ``add_doxygen_target`` function.
  It takes two parameters: the target name, which will be prefixed
  with ``doxygen-``, and one or more file names in ``CONFIGS``. Each
  of the configuration files will be processed with
  ``configure_file``. The final output location is controlled by what
  is defined in these configuration files. E.g. by defining
  ``OUTPUT_DIRECTORY`` relative to the ``CMAKE_BINARY_DIR``:

  .. code-block:: ini

     OUTPUT_DIRECTORY = @CMAKE_BINARY_DIR@/doc/api

  See ``doc/doxygen/conf.in`` in the FleCSI source repository as an
  example configuration.

Coverage and Unit Testing
-------------------------

FleCSI uses its own unit-testing framework and installs the necessary
CMake files to allow using it in your own applications.

``coverage``
  Adds the ``ENABLE_COVERAGE_BUILD`` CMake option. If enabled, this
  will add compiler and linker flags to enable capturing coverage
  information.

``unit``
  Adds the ``ENABLE_UNIT_TESTS`` CMake option. If enabled, it turns on
  CMake's testing capabilities through ``ctest`` and defines a
  ``test`` target.

  While you can define your own test executables manually with `add_test
  <https://cmake.org/cmake/help/latest/command/add_test.html>`_, this
  CMake file also defines its own ``add_unit`` function for
  writing tests based on FleCSI Unit Test framework.

  .. code-block:: cmake

     add_unit(test-name                           # name of target
              SOURCES src1 src2 ... srcN          # list of source files
	      INPUTS in1 in2 ... inN              # list of input files
	      LIBRARIES lib1 lib2 ... libN        # libraries linked to test target
	      DEFINES define1 define2 ... defineN # defines added to test target
	      ARGUMENTS  arg1 arg2 ... argN       # command arguments
	      TESTLABELS label1 label2 ... labelN # labels added to test target
	      PROCS nprocs1 nprocs2 ... nprocsN   # number(s) of MPI processes
	     )

  ``add_unit`` will take the sources files in ``SOURCES`` and compile
  them together with a predefined ``main()`` function. It will also
  link to any ``LIBRARIES`` and add ``DEFINES`` as compile
  definitions.

  If the test uses input files, they can be specified as
  ``INPUTS``. This ensures they are copied to the execution folder.

  Command-line arguments are passed via the ``ARGUMENTS`` option. You
  can also control the number of MPI processes with ``PROCS``. If you
  provide more than one value in ``PROCS``, this will define one
  target per value with a name ``<target-name>_<value>``.

  .. note::

     If FleCSI was compiled with Kokkos, Legion and CUDA support,
     ``add_unit`` will append ``--backend-args="-ll:gpu 1"`` to the
     arguments passed to your test executable.


  ``TESTLABELS`` can be added to your test to allow filtering based on
  label when using ``ctest``.

  **Usage:**

  Here is a minimal unit test file ``mytest.cc``:

  .. code-block:: cpp

     #include <flecsi/util/unit.hh>

     int mytest_driver() {
       UNIT() {
         ASSERT_TRUE(true);
       };
     } // mytest_driver

     flecsi::unit::driver<mytest_driver> driver;

  Which can be compiled with the following ``CMakeLists.txt``:

  .. code-block:: cmake

     cmake_minimum_required(VERSION 3.20)
     project(myproject LANGUAGES CXX C)

     set(CXX_STANDARD_REQUIRED ON)
     set(CMAKE_CXX_STANDARD 17)

     find_package(FleCSI REQUIRED)
     find_package(FleCSICMake REQUIRED)

     include(unit)

     add_unit(mytest
              SOURCES mytest.cc)

  To configure and compile:

  .. code-block:: console

     mkdir build
     cd build
     cmake -D ENABLE_UNIT_TESTS=on ..
     make

  Once compiled, you can run the tests with:

  .. code-block:: console

     make test
     # OR
     ctest


Code Formatting
---------------

``format``

  Add the ``ENABLE_FORMAT`` and ``ClangFormat_VERSION`` CMake options.
  If ``ENABLE_FORMAT`` is ``ON``, you can use ``ClangFormat_VERSION``
  to specify which version of ``clang-format`` should be used for
  formatting.

  When enabled, it adds a ``format`` target that depends on both
  ``git`` and ``clang-format`` to be present. It also requires the
  source tree to be a Git checkout. Running this target will find all
  ``.hh`` and ``.cc`` files and apply the style defined in the
  project's ``.clang-format``.


Dependencies
------------

Some projects might want to explicitly link to dependencies that
FleCSI uses itself. All external libraries used by FleCSI are included
as their own CMake file.

The general structure in these files is that they add a
``ENABLE_<PACKAGE>`` CMake option and, if necessary, more advanced
options for customization. If enabled, the package defines, include
folders and libraries will be appended to the globals ``TPL_DEFINES``,
``TPL_INCLUDES`` and ``TPL_LIBRARIES``.

* ``hdf5``
* ``hpx``
* ``kokkos``
* ``legion``
* ``mpi``
* ``openmp``
* ``parmetis``
* ``boost``
* ``caliper``

.. note::

   ``caliper`` does **not** define a ``ENABLE_CALIPER`` option, but instead a
   ``CALIPER_DETAIL`` option with possible values of: ``none``, ``low``,
   ``medium``, ``high``. The library is only added if the value is not
   ``none``.


Other files
-----------

``colors``
  Defines several ASCII color codes for colored console output.

  .. hlist::
     :columns: 3

     * ``FLECSI_ColorReset``
     * ``FLECSI_ColorBold``
     * ``FLECSI_Red``
     * ``FLECSI_Green``
     * ``FLECSI_Yellow``
     * ``FLECSI_Brown``
     * ``FLECSI_Blue``
     * ``FLECSI_Magenta``
     * ``FLECSI_Cyan``
     * ``FLECSI_White``
     * ``FLECSI_BoldGrey``
     * ``FLECSI_BoldRed``
     * ``FLECSI_BoldGreen``
     * ``FLECSI_BoldYellow``
     * ``FLECSI_BoldBlue``
     * ``FLECSI_BoldMagenta``
     * ``FLECSI_BoldCyan``
     * ``FLECSI_BoldWhite``

``summary``
  Defines multiple macros to generate a (colored) configuration
  summary. Each of these macros appends to the global ``_summary``.
  At the end of your CMake file you can then print this summary using
  ``message(STATUS ${_summary})``.

  ``summary_header`` will add a header.

  ``summary_info(name info allow_split)`` will take a given ``name``
  and add its value ``info`` next to it. If ``info`` is a
  space-separated list of values, ``allow_split`` controls if each
  value should be in its own line.

  ``summary_option(name state extra)`` is used for adding Boolean
  values to the summary. If ``state`` evaluates to ``TRUE`` the option
  state will be shown in a bright green color, followed by what is in
  ``extra``. Otherwise, the ``state`` will be shown in gray.
