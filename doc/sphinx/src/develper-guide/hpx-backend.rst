FleCSI's HPX Backend
********************

This section of the documentation is intended for developers of the core FleCSI library.
HPX is the `C++ Standard Library for Concurrency and Parallelism <https://github.com/STEllAR-GROUP/hpx>`_. 
For an extensive description of HPX and its API, please see the `HPX Documentation <https://hpx-docs.stellar-group.org/latest/html/index.html>`_.

At its core, HPX is a highly efficient user-level threading implementation.
On top of this core functionalities, HPX implements:

* All of the facilities related to concurrency and parallelism as defined by the C++ Standard, such as ``hpx::thread``, ``hpx::future``, or ``hpx::async``.

* The full set of the C++17/C++20/C++23 parallel algorithms; in fact its is one of the first full openly available implementations of those.

  HPX adds certain extensions to the algorithms, such like asynchronous execution, parallel range based algorithms, and explicit vectorization execution policies ``simd`` and ``par_simd``.

* Full set of the sender-receiver proposal (`std::execution <https://wg21.link/p2300>`_) implemented using C++17.

* An uniform integration of your Kokkos, CUDA, HIP, and SYCL (oneAPI) kernels enabling their asynchronous invocation.

* Fully distributed operation enabling remote management of distributed C++ objects and the invocation of global C++ functions and C++ member functions on those objects. 

  HPX extends the standard parallelism and concurrency interfaces for use on tightly coupled clusters.
  This functionalities build on top of a global virtual address space that enables load balancing and an uniform API for local and remote operations.

The goal of HPX is to create a high quality, freely available, open source implementation of a new programming model for conventional systems, such as classic Linux based Beowulf clusters or multi-socket highly parallel SMP nodes.
At the same time, HPX exposes a very modular and well designed runtime system architecture that allows porting it implementation to new computer system architectures.
Real-world applications drive the development of the runtime system, which ensures a stable API providing a smooth migration path for developers.

The API exposed by HPX is modeled after the interfaces defined by the C++11/14/17/20 ISO standard. 
The library also adheres to the programming guidelines used by the Boost collection of C++ libraries. 
HPX aims to improve the scalability of today's applications and to expose new levels of parallelism that are necessary to take advantage of the exascale systems of the future.

-----

High Level Picture
++++++++++++++++++

FleCSI's modular structure enables retargetting it to different back-end implementations. 
Those provide features that are needed for its pr oper functioning. 
In particular, HPX specializes the following backend modules: :ref:`HPX's ``run`` module specialization`, :ref:`HPX's ``exec`` module specialization`, and :ref:`HPX's ``data`` module specialization`.

From a higher level perspective, the HPX backend extracts execution dependencies between FleCSI tasks by analyzing the annotations (accessors) provided to the task's arguments.
Each of the FleCSI tasks is then scheduled as a HPX tasks such that it will run only after all FleCSI fields it may depend on have been updated as spefied by the access rights defined for the corresponding argument of that task.
This ensures that all tasks run as early as possible and with as much concurrency as possible. 
In this case, independent FleCSI tasks may run concurrently if sufficient compute resources are available.

This dependency tracking is implemented by storing an instance of a ``hpx::shared_future<void>`` with each FleCSI field. 
This future becomes ready whenever the operation has finished running that is specified by the access rights on the corresponding FleCSI task argument.

Conceptually, the HPX backend implements the data management similarily to FleCSI's MPI backend, while the task execution management is similar to FleCSI's Legion backend.
Consequently, the HPX backend will currently support only one ``color`` per rank. 
It however supports concurrent execution of the scheduled FleCSI tasks whenever possible.

The HPX backend is enabled at CMake configuration time by specifying the ``-DFLECSI_BACKEND=hpx`` option.
This will result in the macro ``FLECSI_BACKEND`` being defined as ``FLECSI_BACKEND_hpx`` (``3``) at compile time.

Benefits of Using the HPX Backend for FleCSI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here are the main benefits of using HPX as a FleCSI backend:

* HPX is a C++ library and for this reason it integrates well with the FleCSI code base.

  HPX exposes an API that is fully conforming to the C++ standard, thus its use is easy to understand and straightforward to maintain.
  The library itself is highly portable, which means that porting FlCSI to new hardware architecture and compute platforms will require little effort./

* The HPX backend is automatically discovering and adding task asynchrony and concurrent execution of tasks at runtime with minimal overheads.

  This discovery is based on the the analysis of the data dependencies that are implicitly defined by the user through the access rights associated with the argument types for a particular FleCSI task.
  As a result, any task that can run concurrently (independently of other tasks) will do so as long as sufficient compute resources are available.

* The use of the HPX library as the runtime platform enables portability in code and performance.

  The built-in facilities in the runtime ensure load balancing of local tasks.
  Runtime adaptivity mechanism implemented in the runtime to improve overall performance are available to every FleCSI application.

* The existing built-in tracing and performance analysis capabilities are availbe to FleCSI users.

  This enables the rapid visualization of traces, which helps with performance analysis and optimization.

* The HPX backend exposes more parallelism than the MPI backend, thus allowing to break the lock-step between ranks.

* The HPX backend exposes better performance compared to the Legion backend.

  The code base of the HPX backend is much smaller and simpler compared to the Legion backend.

Task Dependencies managed by the HPX Backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The HPX backend manages several types of task dependencies. 
A created dependency causes the dependent task to be guaranteed not to start running before the task it depends on is finished executing.
A task A is possibly used as a required dependency for task B if both tasks operate on the same accessor argument (field). 
The type of the dependency is determined by the access rights associated with this field on task A and task B. 
The sequencing of the tasks is determined by the order in which the tasks are being scheduled (executed) by the user program. 

* *Read-after-read*: task A and task B both have read-only access rights to a given field.

  In this case no dependency is imposed and both tasks are allowed to run concurrently.

* *Write-after-write*: task A and task B both have write-only or read-write access rights to a field.

  While such a situation might look pointless as the second write operation would simply overwrite the first write operation, such a dependency is still allowed as it may simplify user code.
  The created dependency makes sure that task A does not overwrite the result produced by the task B.
  In effect, task B starts running only after task A has finished running.

* *Read-after-write*: task A has write-only or read-write access rights, while task B has read-only access rights to a field.

  In this case a dependency is being created that ensures that task B starts runing only after task A is finished executing.
  This ensures that task B always sees the value written by task A.

* *Write-after-read*: task A has read-only access rights, while task B has write-only or read-write access rights to a field.

  A dependency is being created that ensures that task B always waits for task to finish reading the value of a field before overwriting it with a new value.
  Note that this type of dependency tracking also applies to the case of a write after multiple reads. 

These rules are applied to any combination of fields various tasks may need to access, each of those fields may depend on a separate distinct task. 
As a result a possibly complicated dependency graph is created that represents the data dependency structure of the user's code.

Note also, that in addition to the deduced dependencies as described above, the HPX backend additionally adds internal FleCSI tasks to this dependency graph, such as the operations related to the copy engines or field reduction operations.

Each of the dependencies is represented by an HPX ``hpx::shared_future<void>`` that is associated with the corresponding FleCSI field data. 
Each task that creates a dependency initializes the corresponding future just before the task starts running and makes the future 'ready' after the task has finished executing.
Each task that depends on a result is delayed until all futures it depends on have been marked 'ready'.
The HPX runtime makes sure the tasks are then scheduled and run in the correct order.


HPX's ``run`` module specialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this module, the HPX backend specializes the following types:

* ``flecsi::run::context_t``: a type representing the HPX backend instance that manages the initialization and shutdown of the HPX runtime system.

  The type ``context_t`` is responsible for initializing and launching the HPX runtime system.
  It schedules a function as the first HPX task to run that is passes to ``hpx::init``, which finishes the initialization of the FleCSI environment, and launching FleCSI's startup action that was passed to ``context_t::init``.
  Once FleCSI is finished running, ``hpx::finalize`` (which is a non-blocking operation) signals to the HPX runtime that it should exit once all scheduled operations have ceased.
  HPX is initialized with the following additional configuration options:

  * ``hpx.force_min_os_threads=2``: instruct the HPX runtime to occupy at least two cores for scheduling FleCSI tasks.
    This setting has to be taken into account when running more than one HPX locality (rank) on the same node. 
    Any single node should not run more than ``N`` localities, where ``N == num_cores / 2``.

  * ``hpx.handle_signals=0``: disables installing HPX signal handlers as those may interfere with FleCSI's error reporting.

  * ``hpx.run_hpx_main=1``: instructs HPX to actually schedule the initial task (the lambda passed to ``hpx::init``) on all localities.

  Additionally, the `context_t`` type manages the communicator cache(s), one for the communicators needed for HPX's collective operations and one for HPX's peer-to-peer communication operations.
  These caches also maintain the generational numbers needed to ensure proper sequencing of communication operations invoked on the same communicator instance.

  The ``context_t`` type also exposes functionalities that allow to drain all currently scheduled FleCSI tasks (i.e. wait for them finish running).
  The member function ``context_t::termination_detection`` is used by the HPX backend to create synchronization barriers for FleCSI ``mpi`` tasks.

* ``flecsi::run::config``: a type storing the HPX specific command line arguments provided by the user. 

  This type is used by FleCSI's testing infrastructure to supply additional configuration options specific to FleCSI's tests.
  Currently, two additional configuration options are supplied:

  * ``hpx.ignore_batch_env=1``: instructs HPX not to rely on environmental information passed to the application by SLURM (or any other) batch environment.
    This setting is necessay, as FleCSI runs all tests using a single batch environment, even if different tests involve, e.g. different numbers of ranks. 
    If this setting is not used, conflicting configuration information is being passed to HPX, causing possible hangs during the execution of the tests.

  * ``hpx.os_threads=4``: instructs HPX to limit the number of cors to use for scheduling FleCSI tasks to four.
    This configuration setting is applied to avoid oversubscription of the test environment as all ranks are run on the same node by the CIs.

* It also provides an HPX specific task-local storage implementation ``flecsi::task_local<T>`` that FleCSI uses to store an instance of an arbitrary type ``T`` with each generated HPX task.

  HPX 'threads' (HPX tasks) represent execution agents in the HPX runtime that are managed by a user-level scheduling system. 
  This scheduling system binds a kernel thread (pthread) to each of the utilized cores on a node, i.e. the affinities for that kernel thread are defined such that the OS will srun it on a particular core only.
  HPX tasks can move from core to core, i.e. can be re-scheduled to run on a different core after suspension.
  For this reason, the use of conventional thread-local storage is not possible for associating data to each of those 'threads'.
  The type ``flecsi::task_local<T>`` binds the existing HPX APIs that manage HPX user-level 'thread'-local storage to the interface that is expected by FleCSI when accessing those data items.

  Note that HPX 'threads' can be scheduled to execute 'inline' (i.e. ''nested').
  In this case, the nested HPX 'thread' is run directly in the context of its parent 'thread'.
  Thread local storage in this case will always refer to a data item that is stored in association with the outermost HPX 'thread'.

HPX's ``exec`` module specialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this module, the HPX backend specializes the following types and functions:

* ``flecsi::future``: a type that represents dependencies between data items and execution results of tasks in FleCSI.

  This type binds an underlying ``hpx::shared_future<void>`` to the ``flecsi::future`` API expected by the rest of FleCSI's infrastructure.
  The type ``hpx::shared_future<T>`` at its core is very similar to ``std::shared_future<T>``.
  It however adds a couple of additional functionalities that are being used by the HPX backend to chain tasks (define task dependencies).
  The most important of those functionalities are ``hpx::shared_future<T>::then()``, the ``hpx::dataflow()``, and ``hpx::wait_all()`` APIs that are applied to orchestrate the asynchronous execution flow determining the sequence of executing the scheduled FleCSI tasks.

* ``flecsi::exec::reduce_internal``: this function is responsible for scheduling a FleCSI task.

  The function traverses all arguments supplied to the task that is to be scheduled (see: ``bind_accessors``).
  Depending on the access rights associated with those arguments, the function derives the dependencies between the task to be scheduled and tasks that were previously scheduled.
  Please see :ref:`Task Dependencies managed by the HPX Backend` for more information about the types of dependencies created.
  Note that the FleCSI bind operation that ensures that the field memory is available is delayed such that it runs only after all dependencies for the scheduled task have been satisfied.
  This also possibly schedules additional reduction operations to run after the scheduled task is finished executing but before all dependent task are triggered.

* ``flecsi::exec::task_prologue<task_processor_type_t>``: the template responsible for analysing the access rights specified for FlecSI task arguments and generating the corresponding execution dependencies between those tasks.

  The prolog traverses the arguments of the scheduled task to:
  
  * Prepare the necessary data that will allow binding the accessors to their corresponding underlying memory later (during executing the constructor of the `bind_accessors`` facility.

  * Derive all dependencies on fields that are defined by the access rights associated with a particular argument and the operations on the same field that have to finish before the current task is allowed to run.

    Each future that represents a field operation performed by a task that the current task has to depend on is collected during this traversal operation (see the data member: ``task_prolog::dependencies``).
    These futures are being used to produce a new future that will become ready once all of the initial futures have become ready (this is achieved using the HPX ``hpx::dataflow`` API).
    The scheduled task itself is passed to ``hpx::dataflow`` as the function to run once all the arguments to this API have become ready.
    The new future (the one returned fro ``hpx::dataflow`` (i.e. the one representing the factthat the scheduled task has finished running) is then stored with the fields the arguments of the scheduled task, replacing the futures that were originally stored with those fields before the traversal. 

  * Schedule ghost copy operations for the fields involved as necessary and tie those into the dependency graph as additional steps that have to finish before the scheduled task can run.

* ``flecsi::exec::bind_accessors<task_processor_type_t>``: the template responsible for binding all accessors to the appropriate underlying field data items.

  This type traverses all arguments supplied to the FleCSI task in order to bind the accessor arguments to their corresponding underlying memory.
  The corresponding operation is run right before the actual FleCSI task executes, only after all fields the task depends on have been updated.

* ``flecsi::exec::fold::wrap<hpx::serialization::serialize_buffer<R>>``: a template specialization for an internal HPX type needed for proper HPX serialization of types.

  The type ``hpx::serialization::serialize_buffer<R>`` is s special zero-copy-enable serialization type well integrated with the HPX serialization infrastructure.
  It allows to wrap arrays of any type ``R`` such that no copy operations are being performed on those arrays during serialization.

HPX's ``data`` module specialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this module, the HPX backend specializes the following types and functions:

* ``flecsi::data::backend_storage``: a type that holds the HPX backend specific data items needed to manage the execution dependencies between FleCSI tasks.

  This specialized type for every FleCSI field holds two additional data items: a ``hpx::shared_future<void>`` representing whether the data in the corresponding field is valid and a state variable used to specify whether the future represents a previous read or write operation to the field. 
  The future is made ready whever the FleCSI task that is currently operating on the field has finished running.
  The future will have continuations attached that trigger (run) FleCSI tasks that depend on this field (see the description of ``flecsi::exec::task_prologue<task_processor_type_t>`` above).

* ``flecsi::data::copy_engine``: a type that manages the copy operations required for managing the data exchange between the distributed FleCSI processes for a given field.

  This type manages the tasks that are related to FleCSI's implicit broadcasting and reduction operations for distributed fields.

Large parts of the code are shared with the MPI backend. 
The specialized types customize the functionalities that bind to the corresponding APIs exposed by HPX, such as collective operations and peer-to-peer communication between processes.

