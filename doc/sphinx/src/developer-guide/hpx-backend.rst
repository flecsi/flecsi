HPX Backend
***********

`HPX <https://hpx.stellar-group.org>`_ is an efficient user-level threading implementation that also extends the `C++ concurrency support library <https://en.cppreference.com/w/cpp/thread>`_ to operate across multiple processes.

Overview
++++++++

The HPX backend implements data management similarly to FleCSI's MPI backend while task execution management is implemented similarly to FleCSI's Legion backend.
Consequently, the HPX backend supports only one ``color`` per process.
However, it supports concurrent execution of FleCSI tasks whenever possible.

The HPX backend extracts execution dependencies among FleCSI tasks by analyzing the task parameters (accessor privileges) and arguments (which fields and topologies are used).
Each of the FleCSI tasks is then scheduled as an HPX task such that it will run only after all FleCSI fields to which it has write access have been relinquished by all tasks having read and/or write access to those fields and all FleCSI fields to which it has read access have been relinquished by all tasks having write access to those fields.
This ensures that all tasks run as early as possible and with as much concurrency as possible.

Dependency tracking is implemented via a set of ``hold`` objects, which track the HPX futures and active communicators that are associated with each field.
When a task has finished execution, the corresponding future is marked as "ready".

As communicator creation is expensive, the HPX backend strives to reuse existing communicators.
Whether or not a task requires a communicator—as one of its dynamic descendants might—it begins by walking the dependency DAG to prune nodes (``comms`` objects—one per task) that no longer are needed, thereby simplifying the graph for subsequent traversals.
The task then adds a single node to the DAG, unless it has nowhere to store it.
This node points to all direct predecessors, "pointing through" any that lack communicators to the nearest indirect predecessor that does not.
Finally, the task stores a shared pointer to the node in appropriate fields.

As a result of these graph operations, communicators effectively are moved down the DAG to the deepest point where they still can be discovered by potential users of those communicators.
A communicator held by a task with no dependents cannot be migrated.
Rather, the communicator remains inaccessible until it is destroyed along with the regions used by the task graph when the computation performed by that graph completes.

Unique features of the HPX backend's implementations of :ref:`run <hpx-run-mod>`, :ref:`exec <hpx-exec-mod>`, and :ref:`data <hpx-data-mod>` are described below, followed by a discussion of how the HPX backend manages :ref:`task dependencies <hpx-task-deps>`.

.. _hpx-run-mod:

``run`` module
++++++++++++++

``flecsi::run::context_t``
^^^^^^^^^^^^^^^^^^^^^^^^^^

``flecsi::run::context_t`` begins executing the control model by passing a task to ``hpx::init`` that finishes the initialization of the FleCSI environment and launches the FleCSI startup action that was passed to ``context_t::init``.
Once FleCSI has finished running, ``hpx::finalize``, which is a non-blocking operation, signals to the HPX runtime that it should exit once all scheduled operations have completed.

``flecsi::run::context_t`` manages communicators (an ``hpx::collectives::communicator`` wrapped in a ``flecsi::run::communicator``).
Its ``p2p_comm`` method returns a (singleton) communicator for HPX peer-to-peer communication operations, and its ``world_comm`` method allocates and returns a new communicator for HPX collective operations.
The latter maintains a generation number, incremented via ``communicator::gen``, that is used to ensure proper sequencing of communication operations invoked on the same communicator instance.

``flecsi::run::context_t`` provides the ability to drain all currently scheduled FleCSI tasks (i.e., wait for them to finish running).
The member function ``context_t::termination_detection`` is used by the HPX backend to create synchronization barriers for FleCSI ``mpi`` tasks.

``flecsi::task_local``
^^^^^^^^^^^^^^^^^^^^^^

The HPX backend implements ``flecsi::task_local`` in terms of HPX's "thread data", which follows an HPX task even if it is suspended and later resumed on a different kernel thread.
More specifically, each HPX thread defines a single object of type ``task_local_data`` that backs all objects of the various ``flecsi::task_local<T>`` types.
That is, ``task_local_data`` is a per-task, type-erased map from ``task_local*``\ s to ``T*``\ s.

.. _hpx-exec-mod:

``exec`` module
+++++++++++++++

``flecsi::exec::task_prologue_base``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This class is responsible for analyzing the access rights specified for FleCSI task arguments and generating the corresponding execution dependencies among those tasks.
The prolog traverses the arguments of the scheduled task to perform the following operations:

  * For each FleCSI task parameter/argument,

    * Prepare the necessary data for binding the task accessor to its underlying memory.
      (Binding occurs during task execution when executing the ``bind_accessors`` constructor.)

    * Schedule any required ghost-copy operations for the field and tie those into the dependency graph as additional steps that have to finish before the scheduled task can run.

    * Derive all dependencies (expressed as ``hpx::future``\ s) on the field associated with the current parameter.
      These dependencies are defined by the field's access rights and the operations on the same field that have to finish before the current FleCSI task is allowed to run.

  * Invoke ``hpx::dataflow`` to run the current task with the futures derived above as arguments.
    This returns a new future for the current task.

  * For each FleCSI task parameter/argument,

    * Update the corresponding field to depend on this new future for subsequent read and/or write accesses as discussed in :ref:`Managing Task Dependencies <hpx-task-deps>` below.

    This procedure extends the dependency DAG, ensuring that the current task will block on the completion of all tasks with conflicting access to the fields the current task declares it will access.


A noteworthy aspect of ``flecsi::exec::task_prologue_base``\ 's implementation is that a task can run *concurrently* with the installation of its future on the fields.
While this ordering may seem unsafe, it is legitimized by the fact that task launches are serialized.
As a result, the vulnerable state between dependencies being derived and the task's future being installed is in fact unobservable by a FleCSI program.

``flecsi::exec::task_prologue_base`` is exposed via a template, ``task_prologue``, but the template argument (the processor type) is ignored because the HPX backend does not distinguish processor types.

``flecsi::exec::fold::wrap``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This class defines overloads of ``operator()`` that perform HPX data serialization.
``hpx::serialization::serialize_buffer<R>`` is a special zero-copy-enable serialization type integrated with the HPX serialization infrastructure.
It enables wrapping arrays of any type ``R`` to prevent copy operations from being performed on those arrays during serialization.


.. _hpx-data-mod:

``data`` module
+++++++++++++++

In this module, the HPX backend provides an implementation of ``flecsi::data::copy_engine`` and ``flecsi::data::backend_storage``.
Although both implementations perform operations that are unique to HPX, large parts of the code are shared with the MPI backend.

``flecsi::data::copy_engine``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The HPX-specific code for ``flecsi::data::copy_engine`` customizes the operations that correspond to APIs exposed by HPX such as collective operations and peer-to-peer communication between processes.

``flecsi::data::backend_storage``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This type holds the HPX backend-specific data items needed to manage the execution dependencies among FleCSI tasks (described in :ref:`Managing Task Dependencies <hpx-task-deps>` below).
The relevant data types are as follows.
``flecsi::data::backend_storage``, which is instantiated for each FleCSI field, comprises a single ``hold`` for the most recent write to the field and a vector of ``hold``\ s for the reads since.
A ``hold``, which may be empty, comprises a pointer into the HPX communicator graph and a ``fate``.
A ``fate`` is a future that can be shared by multiple ``hold``\ s.
(It wraps a ``hpx::shared_future<void>``.)

.. _hpx-task-deps:

Managing Task Dependencies
++++++++++++++++++++++++++

The HPX backend establishes explicit dependencies among HPX tasks to impose a partial ordering of FleCSI tasks based on launch order and the access rights specified for the relevant field accessors.
However, the backend additionally adds internal HPX tasks to the task-dependency graph.
These internal HPX tasks manage operations related to copy engines and field reductions.

A dependency is represented by an HPX ``hpx::shared_future<void>`` that is associated with the corresponding FleCSI field data.
Each task stores for each field a pointer to a single empty future to be used by subsequent dependent tasks.
When the HPX task is launched, it replaces the empty future with a concrete future from ``hpx::dataflow``.

Each read of a field depends on completing only the most recent write of that field.
(This is why a single write ``hold`` suffices.)
Each write of a field depends on completing *all* of the most recent reads of that field.
(This is why a vector of read ``hold`` is required.)
In the case of write-after-write (i.e., ``wo`` or ``rw`` accesses with no intervening ``ro`` accesses), the write of the field depends only on the most recent write to that field.
Thus, the HPX backend maintains the frontier of the task graph, which suffices for dynamically adding new dependencies.
