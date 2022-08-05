.. include:: <isopub.txt>

.. _news:

Release Notes
*************

``flecsi::`` qualification is omitted throughout.

.. Each release has as sections some subsequence of Removed, Other
   incompatibilities, Possible incompatibilities, Deprecated, New features,
   and Fixed.

   Each such section discusses some subsequence of Build, Runtime, Data,
   Execution, Topologies, Legion backend, MPI backend, On-node parallelism,
   Utilities, and Logging.

Changes in v2.2.0
+++++++++++++++++

Deprecated
^^^^^^^^^^
* Runtime

  * Using a control policy type that does not inherit from ``control_base`` |mdash| derive from it with no other changes
  * ``control::state`` |mdash| use ``control::policy``

* Data

  * ``global_topology`` and ``process_topology`` |mdash| create instances of ``topo::global`` and ``topo::index`` as needed

* Topologies

  * In ``narray::access``, ``extents`` and ``range`` |mdash| renamed to ``range`` and ``domain`` respectively

* Logging

  * The namespace ``flecsi::log`` |mdash| use ``flecsi::flog`` instead

New features
^^^^^^^^^^^^
* Build

  * The HIP compiler is supported.

* Runtime

  * Control policies may inherit from ``control_base`` to enable throwing ``control_base::exception`` and to limit their lifetime to that of ``control::execute``.
  * ``control::policy`` replaces ``control::state``.
  * ``meta_point`` defines special control points for a specialization's use via ``control::meta``.
  * New option ``--Xbackend`` to pass single backend arguments. Can be used multiple times.

* Data

  * ``launch::mapping`` allows field data to be accessed from other colors (using ``multi`` task parameters) with the Legion backend.
  * ``reduction_accessor`` allows ``dense`` fields on a ``global`` topology to be updated collectively.
  * The ``particle`` layout supports efficient creation and destruction of unordered field elements.
  * Field definitions are copyable and may be destroyed after use.
  * Fields may be of non-portable types so long as they are used only by MPI tasks.
  * ``single`` accessors support ``->`` to access the members of a field value of class type.

* Execution

  * Tasks may be executed as ``omp``, allowing efficient use of OpenMP.
  * ``exec::trace`` improves performance of loops when used with the Legion backend.

* Topologies

  * ``unstructured`` represents unstructured meshes.
  * ``narray`` represents hyperrectangular arrays, including structured meshes.
  * ``global`` topology instances may be created with sizes other than 1.
    (All elements are shared together among all point tasks in an index launch.)

* MPI backend

  * The ``toc`` processor type is supported.

* On-node parallelism

  * ``parallel_for``, ``forall``, ``parallel_reduce``, and ``reduceall`` may be used without Kokkos enabled, in which case they run serially.
  * ``mdiota_view`` provides support for multi dimensional ranges. Supports ``full_range``, ``prefix_range`` and ``sub_range`` options.

* Utilities

  * ``unit.hh`` provides macros, based on Google Test, for writing unit tests in or outside of tasks.
  * ``serial`` provides a general-purpose serialization interface.
    It may be extended to allow additional types to be used as task parameters or with MPI communication.
  * ``mdcolex`` accesses a multi-dimensional array using Fortran-like syntax.
  * ``mpi::one_to_alli`` scatters generated and/or serialized data with bounded memory usage.
  * MPI communication functions compute values to send in rank order and support mutable functors.
  * ``substring_view`` represents part of another range.
  * ``crs`` provides a Compressed Row Storage implementation.
  * ``equal_map`` and ``offsets`` represent partitions of integer ranges.

* Logging

  * ``flog_fatal`` suppresses backtraces (already unavailable if ``NDEBUG`` is defined) unless ``FLECSI_BACKTRACE`` is set in the environment.
  * ``flog`` is a new name that avoids collisions with ``::log`` in code lacking proper namespace qualifications.

Changes in v2.1.1
+++++++++++++++++

Possible incompatibilities
^^^^^^^^^^^^^^^^^^^^^^^^^^
* Build

  * ``FLECSI_RUNTIME_MODEL`` is renamed to ``FLECSI_BACKEND`` (it never referred to :ref:`runtime`).

* Utilties

  * ``util::mdspan::operator()`` is removed (it had an inconsistent argument order).

Deprecated
^^^^^^^^^^
* Utilities

  * in ``util::annotation``, ``begin`` and ``end`` |mdash| use ``guard`` or ``rguard``

Fixed
^^^^^
* Build

  * HDF5 is supported by the CMake configuration.
  * Caliper is supported by the MPI backend.
  * Building with the MPI backend properly links to the threading library.
  * Added ``FLECSI_VERSION`` define to record version

* Runtime

  * Certain control-flow graphs compile with Graphviz support and are drawn correctly.
  * ``--backend-args`` can be specified multiple times. The collected arguments are passed to the backend.
  * MPI and Kokkos are initialized with no arguments (so that they cannot misinterpret arguments not meant for them).

* Data

  * Mutators support write-only privileges (which, as with accessors, are necessary to properly initialize fields).
  * Mutators work on fields over zero index points.

* Topologies

  * Topology accessors that provide metadata for direct access in a ``toc`` task work properly when writable (although without such direct access in that case).

* Legion backend

  * Certain user-level fields are allocated properly.
  * MPI tasks with reference parameters support argument conversions correctly.
  * Errors at process termination in certain configurations are avoided.
  * Accessors with both ``wo`` and ``na`` privileges are processed correctly.

* MPI backend

  * ``exec::fold::min`` and ``exec::fold::max`` work with ``bool`` (but may be slower than using another type like ``int``).
  * Index futures work properly when declared ``const``.
  * Index futures work with ``bool`` return type.

* On-node parallelism

  * More accessor and topology accessor functions are available on a device.
  * Several uses of "iterator" in documentation have been corrected to use "range".

* Logging

  * Tags work correctly in tasks executed in parallel.
  * Messages are sorted by timestamp correctly.
  * Exiting the process before ``flecsi::finalize`` does not crash.

Changes in v2.1.0 (April 16 2021)
+++++++++++++++++++++++++++++++++

New features
^^^^^^^^^^^^
* Topologies

  * ``topo::help`` is a convenient base class for specialization class templates that defines the non-dependent names in ``specialization``.

Changes in v2.0.0 (March 30 2021)
+++++++++++++++++++++++++++++++++

The changes between versions 1 and 2 are extensive and so are merely summarized here.
The broadest change is that macros are used only for on-node parallelism constructs and for logging.
Some macro arguments become template arguments.

Removed
^^^^^^^
* Data

  * ``flecsi_get_client_handle`` |mdash| use the topology slot directly
  * ``flecsi_get_handles``, ``flecsi_get_handles_all``, ``flecsi_is_at``, ``flecsi_has_attribute_at``, ``flecsi_has_attribute`` |mdash| will be provided differently by ``io``

* Execution

  * ``flecsi_register_task``, ``flecsi_register_task_simple``, ``flecsi_register_mpi_task``, ``flecsi_register_mpi_task_simple`` |mdash| unneeded
  * ``specialization_tlt_init``, ``specialization_spmd_init``, ``execution::driver`` |mdash| the application provides ``main``
  * ``flecsi_execute_task_simple`` |mdash| namespaces require no special treatment
  * ``flecsi_register_program`` |mdash| unneeded
  * ``flecsi_register_global_object``, ``flecsi_set_global_object``, ``flecsi_initialize_global_object``, ``flecsi_get_global_object`` |mdash| use non-local variables directly, with the same restrictions
  * ``flecsi_register_reduction_operation`` |mdash| unneeded

* Topologies

  * Topology element ``id()`` |mdash| iteration produces indices
  * Unstructured mesh |mdash| will be reimplemented

Other incompatibilities
^^^^^^^^^^^^^^^^^^^^^^^
* Data

  * ``flecsi_register_data_client`` |mdash| now topology slots
  * ``flecsi_register_field``, ``flecsi_register_global``, ``flecsi_register_color`` |mdash| now ``field::definition``, or a container of same to implement multiple versions
  * ``flecsi_get_handle``, ``flecsi_get_global``, ``flecsi_get_color``, ``flecsi_get_mutator`` |mdash| now ``definition::operator()``
  * ``data_client_handle_u<T,P>`` |mdash| now ``T::accessor<P>``
  * ``dense_accessor``, ``ragged_accessor``, ``ragged_mutator``, ``sparse_accessor``, ``sparse_mutator`` |mdash| now ``definition::accessor`` or ``definition::mutator``

    * ``ragged`` fields have a ``std::vector``-like interface at each index point
    * ``sparse`` fields have a ``std::map``-like interface at each index point

* Execution

  * ``flecsi_execute_task`` |mdash| now ``execute``
  * ``flecsi_execute_mpi_task``, ``flecsi_execute_mpi_task_simple`` |mdash| pass ``mpi`` to ``execute``
  * ``flecsi_execute_reduction_task`` |mdash| now ``reduce``
  * ``flecsi_color()``, ``flecsi_colors()`` |mdash| now ``color()`` and ``colors()``
  * ``flecsi_register_function``, ``flecsi_execute_function``, ``flecsi_function_handle``, ``flecsi_define_function_type`` |mdash| now ``exec::make_partial``
  * ``execution::flecsi_future`` |mdash| now ``future``, with a simpler interface

* Topologies

  * Specializations inherit from ``topo::specialization`` rather than from a core topology type directly.

New features
^^^^^^^^^^^^
* Runtime

  * Application command-line options can be specified using the ``program_option`` class template and associated helper functions.
  * The ``control`` class template is used to define an extensible control model for an application.

* Data

  * The ``raw`` and ``single`` layouts support special cases of field usage.

* Execution

  * Tasks can be (specializations of) function templates.
  * Tasks can accept parameters of dynamic size (*e.g.*, ``std::vector<double>``), although passing large objects is inefficient.
  * Index launches can be of any number of point tasks (determined by the task's arguments, including ``exec::launch_domain``).
  * ``test`` conveniently executes tasks for unit testing.

* Topologies

  * Multiple user-defined topology instances may exist sequentially or simultaneously.
