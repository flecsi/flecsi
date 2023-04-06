.. |br| raw:: html

   <br />

Tutorial
********

This tutorial attempts to give a basic introduction to the design
philosophy and structure of the FleCSI programming system. Each
subdirectory contains example codes that can be compiled with CMake.
Details on how to build the tutorials are given below.

.. attention::

  The tutorial is primarily designed as an introduction for application
  developers: i.e., we do not go into the details of designing or
  implementing the specialization layer, and the discussion of core
  FleCSI features is limited to the high-level execution and data
  interfaces.

  Because a specialization layer is necessary to use FleCSI, we have
  provided a simple mesh interface as part of the tutorial. Users who
  are interested in the basic structure of a mesh topology
  specialization are encouraged to examine the source code in the
  *specialization* subdirectory of this tutorial.

----

Requirements
++++++++++++

The tutorial assumes that you have successfully installed FleCSI
somewhere on your system. Instructions for different install methods are
available here: :ref:`build`.

Once you have built and installed FleCSI, add the path to your
installation to your *CMAKE_PREFIX_PATH*:

.. code-block:: console

  $ export CMAKE_PREFIX_PATH=/path/to/flecsi/install/dir:$CMAKE_PREFIX_PATH

----

Building the Examples
+++++++++++++++++++++

The tutorial uses a standard CMake build system. To configure and build
the examples, execute the following steps from the *tutorial*
subdirectory of the FleCSI source:

.. code-block:: console

  $ mkdir build && cd build
  $ cmake ..
  $ make

----

Tutorial Examples
+++++++++++++++++

The tutorial examples are designed to guide the reader through basic to
more advanced FleCSI concepts. 

.. toctree::

  tutorial/runtime
  tutorial/control
  tutorial/execution
  tutorial/data
  tutorial/parallel
  tutorial/flaxpy
  tutorial/poisson

.. vim: set tabstop=2 shiftwidth=2 expandtab fo=cqt tw=72 :
