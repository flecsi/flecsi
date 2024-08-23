.. include:: <isopub.txt>

Release Notes
*************

``flecsi::`` qualification is omitted throughout.

.. Each release has as sections some subsequence of Removed, Other
   incompatibilities, Possible incompatibilities, Deprecated, New features,
   and Fixed.

   Each such section discusses some subsequence of Build, Runtime, Data,
   Execution, Topologies, Legion backend, MPI backend, On-node parallelism,
   Utilities, and Logging.

Changes in v2.3.1
+++++++++++++++++

Possible incompatibilities
^^^^^^^^^^^^^^^^^^^^^^^^^^
* Legion backend

  * The error of destroying a topology after ``runtime::control`` returns (perhaps when ``main`` returns) is more likely to produce a crash.

Fixed
^^^^^
* Legion backend

  * The speed of ghost copies has been significantly improved (without losing the decreased memory usage).

* Utilities

  * ``dimensioned_array`` construction works reliably in constant expressions.

Deprecated
^^^^^^^^^^
* Utilities

  * The ``EXPECT_STRCASEEQ``, ``EXPECT_STRCASENE``, ``ASSERT_STRCASEEQ``, ``ASSERT_STRCASENE`` unit
    test macros will be removed in a future release |mdash| use ``*_STREQ`` or ``*_STRNE`` for
    case-sensitive comparison, or write a case-insensitive comparison function and use ``*_TRUE``
    or ``*_FALSE`` to check its output.

Changes in v2.3.0 (June 20 2024)
++++++++++++++++++++++++++++++++

Possible incompatibilities
^^^^^^^^^^^^^^^^^^^^^^^^^^
* Topologies

  * ``narray`` auxiliaries between two colors are owned by the upper color.

Deprecated
^^^^^^^^^^
* Build

  * ``ENABLE_DOXYGEN_WARN`` |mdash| ignored since 2.2.0
  * ``FLOG_ENABLE_COLOR_OUTPUT``, ``FLOG_SERIALIZATION_INTERVAL``, and
    ``FLOG_STRIP_LEVEL`` |mdash| use ``flog::config``

* Runtime

  * ``initialize``, ``start``, ``finalize``, ``run::status``, and ``control::check_status`` |mdash| use ``runtime`` and, optionally, ``getopt``
  * ``program`` |mdash| use ``argv`` directly
  * Use of a control policy's ``node_policy`` type |mdash| remove it
  * ``control::execute`` |mdash| use ``control::invoke`` or ``runtime::control`` with ``run::control_base`` and not ``node_policy``
  * ``option_value`` and needing one in a ``program_option`` validation function |mdash| accept the option type instead
  * ``flecsi/run/control.hh`` |mdash| use ``flecsi/runtime.hh``

* Data

  * ``coloring_slot`` |mdash| use ``specialization::mpi_coloring``

* Execution

  * Constructing a ``trace`` with an id |mdash| use the default constructor

* Topologies

  * ``specialization::cslot`` |mdash| use ``mpi_coloring``
  * ``flecsi/topo/narray/interface.hh`` |mdash| use ``flecsi/topology.hh``
  * ``narray_impl::index_definition::create_plan`` |mdash| always ignored
  * ``narray_base::domain`` and ``is_low``, ``is_high``, ``is_interior``, ``is_degenerate``, ``global_id``, ``size``, ``range``, and ``offset`` in ``narray::access`` |mdash| use ``axis``

* Utilities

  * ``util::dag``, ``util::reorder``, ``util::reorder_destructive``, ``util::intersects``, ``util::set_intersection``, ``util::set_union``, ``util::set_difference`` |mdash| superfluous
  * Passing binary functors to ``mpi::one_to_allv``, ``mpi::one_to_alli``, and ``mpi::all_to_allv`` |mdash| remove second parameter or use ranges
  * ``flecsi/util/annotation.hh``, ``flecsi/util/array_ref.hh``,
    ``flecsi/util/color_map.hh``, ``flecsi/util/common.hh``,
    ``flecsi/util/dag.hh``, ``flecsi/util/demangle.hh``,
    ``flecsi/util/dimensioned_array.hh``, ``flecsi/util/mpi.hh``,
    ``flecsi/util/reorder.hh``, ``flecsi/util/serialize.hh``,
    ``flecsi/util/set_intersection.hh``, ``flecsi/util/set_utils.hh``,
    ``flecsi/util/unit.hh`` |mdash| use ``flecsi/utilities.hh``

New features
^^^^^^^^^^^^
* Build

  * ``flecsi/config.hh`` provides information about the build of FleCSI in use.
  * A :doc:`CMake utility library <user-guide/cmake>` is provided.  (This was added without a release note in 2.2.0.)
  * ``flecsi_add_target_test`` is a CMake function to define tests using existing targets.
  * ``flecsi_add_test`` can use a launcher command to wrap the test execution.

* Runtime

  * ``runtime.hh`` provides access specifically to the runtime model.
  * ``runtime`` represents FleCSI initialization as an object.
  * ``getopt`` parses user-defined command-line options.
  * ``run::dependencies_guard`` allows for application control over initialization of FleCSI dependencies.
  * ``control::invoke`` executes a control model with a policy object constructed from arguments provided.
  * ``run::call`` is a trivial predefined control model.
  * ``program_option`` validation functions can accept the option value directly.
  * ``task_names`` returns a mapping of shortened registered FleCSI task names to their full signature.
  * ``initialize`` accepts ``const char * const *`` for better compatibility.

* Data

  * ``topology_slot`` is now movable; empty slots may be detected with ``topology_slot::is_allocated``.

* Execution

  * ``test`` convenience function launches unit test tasks.
  * ``trace`` objects can be move-assigned.

* Topologies

  * ``specialization::mpi_coloring`` creates a coloring eagerly.
  * ``topo::make_ids<S>(r)`` is a convenience function to convert a range ``r`` of ordinary integers into a range of ``id<S>`` objects.
  * ``unstructured_base::bounding_box`` computes the axis-aligned bounding box of a mesh.
  * ``unstructured::special_field`` is the field definition for special-entity lists.
  * ``unstructured::get_special_entities`` allows access to individual special-entity lists.
  * ``narray_base::distribute`` and ``narray_base::make_axes`` help construct ``coloring`` objects.
  * The names ``axis_definition`` and ``index_definition`` are available in ``narray_base``.
  * ``axis``, ``axis_color``, ``axis_layout``, and ``axis_info`` in ``narray_base`` describe axes in topology accessors.
  * ``ntree`` is a topology that tracks interacting particles. It only supports Legion as a backend and does not support ragged or sparse fields.

* Legion backend

  * Task names are now shortened for better usability in Legion profiling tools. See :doc:`user-guide/profiling` for details.

* Utilities

  * ``transform`` applies a transformation functor to a range.
  * ``partition_point`` and ``binary_index`` find values in sorted ranges.
  * ``permutation_view`` accesses a subset of a range.
  * ``mpi::one_to_allv``, ``mpi::one_to_alli``, and ``mpi::all_to_allv`` additionally accept ranges and unary functors.
  * ``KDTree`` efficiently finds intersections between shapes.
  * Values may be included in expectation/assertion failure messages.
  * ``FLECSI_DEVICE_CODE`` is a macro defined during the device compilation for either CUDA or HIP.

* Logging

  * ``flog::config`` is the collection of FLOG options that can be changed at runtime.
  * ``flog::tags`` returns the names of all defined tags.

Fixed
^^^^^
* Build

  * ``flecsi_add_sphinx_target`` doesn't use ``-n`` (which can break for references into other libraries).

* Data

  * The size of one color of an index space can exceed :math:`2^{32}` (if FLECSI_ID_TYPE is configured approprately).

* Execution

  * Tasks may be declared ``noexcept``.
    (This was fixed but not documented in 2.2.1.)

* Topologies

  * ``narray``

    * ``index_definition::diagonals`` being ``false`` works.
      (This was fixed but not documented in 2.2.1.)
    * Periodic axes require equal boundary and halo depths (differing values never worked reliably).
    * Auxiliary axes support ``full_ghosts`` and ``periodic`` properly.

  * ``unstructured`` supports ``num_intervals`` of 0.

* Legion backend

  * ``omp`` tasks now work in builds with GPU support.
  * MPI tasks properly wait on previously launched tasks.
  * Memory usage for implementing ghost copies has been significantly reduced.
  * Processes with multiple CPUs schedule tasks more efficiently.
  * Launches of zero point tasks succeed (vacuously).
  * The ``--Xbackend -dm:memoize`` option is no longer required to enable tracing.

* On-node parallelism

  * Custom reductions and reduced types work with Kokkos.

* Utilities

  * ``one_to_alli`` no longer triggers a race condition.

* Logging

  * Special options like ``--control-model`` and ``--help`` work reliably.
    (This was documented but not actually implemented in 2.2.1.)
  * FLOG messages are now variable size and are no longer truncated.

Changes in v2.2.1 (July 12 2023)
++++++++++++++++++++++++++++++++

Fixed
^^^^^
* Runtime

  * Control policy objects are value-initialized by ``run::control::execute``.
  * Unrecognized options are properly rejected along with unrecognized positional arguments.
  * The same exit status is used for all erroneous command lines.
  * Control-model graphs are labeled with the program name.
  * Control model output strips parameters, return values, wrappers, and common
    namespaces from actions for better readability.

* Data

  * ``ragged`` accessors with ``wo`` privileges work for GPU tasks. (The field type must still be trivially default-constructible.)

* Utilities

  * ``transform_view`` supports pointers to members (though not during constant evaluation).

Changes in v2.2.0 (April 14 2023)
+++++++++++++++++++++++++++++++++

Possible incompatibilities
^^^^^^^^^^^^^^^^^^^^^^^^^^
* Build

  * ``FLECSI_RUNTIME_MODEL`` is renamed to ``FLECSI_BACKEND`` (it never referred to :doc:`tutorial/runtime`).

* Utilities

  * ``util::mdspan::operator()`` is removed (it had an inconsistent argument order).

Deprecated
^^^^^^^^^^
* Runtime

  * ``control::state`` |mdash| derive the control policy type from ``control_base`` and define actions accepting a reference to it
  * ``run::control_point`` |mdash| use ``control_base::point``
  * ``run::cycle`` |mdash| use ``control_base::cycle``

* Data

  * ``global_topology`` and ``process_topology`` |mdash| create instances of ``topo::global`` and ``topo::index`` as needed

* Utilities

  * in ``util::annotation``, ``begin`` and ``end`` |mdash| use ``guard`` or ``rguard``

* Logging

  * The namespace ``flecsi::log`` |mdash| use ``flecsi::flog`` instead
  * The CMAKE option ``FLOG_SERIALIZATION_THRESHOLD`` is not used anymore 

New features
^^^^^^^^^^^^
* Build

  * The HIP compiler is supported.

* Runtime

  * Control policies may inherit from ``control_base`` to be provided to actions, to allow those actions to throw ``control_base::exception``, and to limit their lifetime to that of ``control::execute``.
  * ``control_base::meta`` defines special control points for a specialization's use via ``control::meta``.
  * New option ``--Xbackend`` to pass single backend arguments. Can be used multiple times.

* Data

  * ``field_reference::get_elements`` and associated types allow control of the storage for elements of ragged/sparse fields.
  * ``launch::mapping`` allows field data to be accessed from other colors (using ``multi`` task parameters) with the Legion backend.
  * ``topology_slot::colors`` reports on the number of colors for an allocated topology.
  * ``reduction_accessor`` allows ``dense`` fields on a ``global`` topology to be updated collectively.
  * The ``particle`` layout supports efficient creation and destruction of unordered field elements.
  * Field definitions are copyable and may be destroyed after use.
  * Fields may be of non-portable types so long as they are used only by MPI tasks.
  * ``single`` accessors support ``->`` to access the members of a field value of class type.

* Execution

  * Tasks may be executed as ``omp``, allowing efficient use of OpenMP.
  * ``exec::trace`` improves performance of loops when used with the Legion backend.
  * ``task_local`` objects define global variables with task-local values.

* Topologies

  * ``unstructured`` represents unstructured meshes.
  * ``narray`` represents hyperrectangular arrays, including structured meshes.
  * ``global`` topology instances may be created with sizes other than 1.
    (All elements are shared together among all point tasks in an index launch.)
  * ``util::id`` and ``util::gid`` store topology-entity IDs.

* MPI backend

  * The ``toc`` processor type is supported.

* On-node parallelism

  * ``exec::parallel_for``, ``forall``, ``exec::parallel_reduce``, and ``reduceall`` may be used without Kokkos enabled, in which case they run serially.
  * ``exec::mdiota_view`` provides support for multi dimensional ranges, with ``exec::full_range``, ``exec::prefix_range`` and ``exec::sub_range`` options.

* Utilities

  * ``unit.hh`` provides macros, based on Google Test, for writing unit tests in or outside of tasks.
  * ``serial`` provides a general-purpose serialization interface.
    It may be extended to allow additional types to be used as task parameters or with MPI communication.
  * ``mdcolex`` accesses a multi-dimensional array using Fortran-like syntax.
  * ``mpi::one_to_alli`` scatters generated and/or serialized data with bounded memory usage.
  * MPI communication functions compute values to send in rank order and support mutable functors.
  * ``substring_view`` represents part of another range.
  * ``equal_map`` and ``offsets`` represent partitions of integer ranges.

* Logging

  * ``flog_fatal`` suppresses backtraces (already unavailable if ``NDEBUG`` is defined) unless ``FLECSI_BACKTRACE`` is set in the environment.
  * ``flog`` is a new name that avoids collisions with ``::log`` in code lacking proper namespace qualifications.

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
