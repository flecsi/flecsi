Execution Model
***************

This section describes the FleCSI execution model.
FleCSI expresses parallelism via tasks, which are coarse-grained and can be distributed, and kernels, which are fine-grained and utilize shared memory.

Schedulers
++++++++++

A FleCSI `scheduler` is an object that manages the launch of tasks.
Where and when to execute the task is determined based on the task's parameter types and arguments.

Schedulers are obtained from the control-model object provided to an action:

.. code-block:: c++

   void action(control_policy &cp) {
     flecsi::scheduler s = cp.scheduler();
     s.execute<task>(...);
   }

Tasks
+++++

A task in FleCSI serves as the bridge between two core parts of the execution model.
On the caller side, each `process` executes its own copy of the ``main`` function and the associated control model.

After all processes collectively launch a task, its function then executes, typically several times concurrently.
Each of these executions is referred to as a `point task`; their number does not necessarily match the number of processes, and they can run on any process.
This allows FleCSI to support flexible and scalable task-parallel execution.

.. figure:: tikz/flecsi_execution_model.svg

   The different levels of execution in FleCSI.

Callee Side
^^^^^^^^^^^

A task is a function that can accept certain special parameter types.
For instance, the task ``foo`` may be defined as:

.. code-block:: c++

   void foo(flecsi::exec::cpu e,
     flecsi::field<double>::accessor<ro, ro> f1,
     flecsi::field<std::size_t>::accessor<rw, ro> f2) noexcept { /* ... */ }

Tasks must be declared ``noexcept``: they are called asynchronously without any means of handling an exception.
Here the first parameter, ``e``, specifies the execution space where the task runs.
Available options include ``cpu``, ``omp``, ``gpu``, or ``accelerator``.
The ``accelerator`` space is flexible and will use the backend selected when compiling Kokkos (e.g., CUDA, OpenMP, or serial), allowing the same task code to use different kinds of hardware.
If there is no execution space parameter, the task will default to CPU execution.

:ref:`field-accessors` are automatically converted from field references provided during the task launch.
The field data they describe is placed in the correct memory space; ghost elements are updated if required by the privileges.

Tasks can also handle collections of fields, using either ``std::vector`` or ``std::tuple``.
For example, if all fields share the same type, a ``std::vector`` can be used to group them together:

.. code-block:: c++

   void bar(flecsi::exec::cpu e,
     std::vector<flecsi::field<double>::accessor<ro, ro>> fv1) noexcept { /* ... */ }

The corresponding invocation might look like:

.. code-block:: c++

   s.execute<bar>(flecsi::exec::on, std::vector{fr1, fr2, fr3});

Caller Side
^^^^^^^^^^^

Tasks are launched by ``scheduler::execute``:

.. code-block:: c++

   s.execute<foo>(flecsi::exec::on, fr1, fr2);

Here, the task ``foo`` from above is specified as a template parameter.
The first argument, ``flecsi::exec::on``, represents the execution space (when one is specified by the task).

The subsequent arguments, ``fr1`` and ``fr2``, are field references passed into the task.
These field references act as logical handles to the underlying data.

.. _future:

Futures and Reductions
++++++++++++++++++++++

In FleCSI, tasks can return values through `futures`.
A future represents the result of a task that might not yet be completed.
The value is available only after the task is actually executed.
This asynchronous behavior allows for flexible execution ordering.

.. code-block:: c++

   int toto() noexcept { return 10; }
   // ...
   auto future = s.execute<toto>();
   int value = future.get();

The ``get`` function blocks until the associated task completes and returns the resolved value of the future.
Another function, ``wait``, can be used to pause execution until the future is resolved.

.. warning::
   The incorrect usage of futures can create some :doc:`performance` bottlenecks.

FleCSI supports reductions through the ``reduce`` function, which combines results from multiple point tasks into a single value. For instance:

.. code-block:: c++

   auto result = s.reduce<exec::fold::sum>( /* ... */ );

.. _reduction-folds:

The future provides the sum of the values returned by all point tasks.
FleCSI provides several built-in reduction folds, including `min`, `max`, `sum`, and `product`.
Users can define custom folds by implementing `a structure <../../api/user/structflecsi_1_1exec_1_1fold_1_1reduce.html>`_ with ``combine`` and ``identity`` methods.
The reduction types can use a specific type or provide a function template.

.. _portable_tasks:

Portable Tasks
++++++++++++++

A portable task comprises *variants* of the task for different execution spaces.
These are expressed as the various specializations of a static member function template.
(Note that for technical reasons they cannot be overloads or non-member function templates.)
FleCSI chooses one of the variants when the task is launched.

The different execution-space types supply the same interface, so the task can be written just once as the primary template.
Alternatively, optimized implementations for each space can be provided as explicit specializations.
Either kind of definition can be deleted to prevent using it.

The example below shows the static member function template for a task:

.. code-block:: cpp

   struct task_variants {
     template<class S>
     static void task(S, /* ... */ ) noexcept {
       /* ... */
     }
   };

The following example defines a portable task with variants for ``exec::cpu`` and ``exec::omp``.
The primary function template is deleted to indicate that no default implementation exists.
This ensures that tasks cannot be launched on the architectures that are not specified.

.. code-block:: cpp

   struct task_variants {
     template<class S>
     static void task(S, /* ... */ ) noexcept = delete;
   };

   // CPU variant
   template<>
   void task_variants::task(exec::cpu c, /* ... */ ) noexcept {
     /* CPU-specific implementation */
   }

   // OpenMP variant
   template<>
   void task_variants::task(exec::omp c, /* ... */ ) noexcept {
     /* OpenMP-specific implementation */
   }

Portable tasks are launched through the scheduler in the same way as other tasks.
FleCSI automatically uses the variant for the execution space where the task runs.

.. code-block:: cpp

   flecsi::scheduler & s = cp.scheduler();
   s.execute<task_variants>(exec::on, /* ... */ );

Given the previous declaration, this call to ``execute`` will not attempt to use ``exec::gpu``.

MPI Tasks
+++++++++

FleCSI also supports a special class of tasks known as `MPI tasks`.
Exactly one point task for an MPI task runs on each process, like the control-model action that launched it.

MPI tasks are invoked as follows:

.. code-block:: c++

   flecsi::execute<qux, flecsi::mpi>(flecsi::exec::on, fr1, fr2);

.. note::
   Unlike standard tasks, MPI tasks are invoked using the ``flecsi::`` namespace rather than a scheduler object with ``s.execute``.
   In future versions, MPI tasks may be launched through the scheduler interface for consistency with other task types.

MPI tasks can use :ref:`multi-accessors` to access fields whose number of colors does not match the number of processes.

.. _tracing:

Tracing
+++++++

The Legion backend uses a feature called `tracing` to improve performance for repeated execution patterns, particularly in critical loops.

When tracing is enabled, Legion records task launches, data movement, and communication patterns during the first execution.
On subsequent iterations, it reuses this recorded information instead of redoing scheduling and analysis, significantly reducing overhead.

Here is an example of tracing in use:

.. code-block:: c++

   static exec::trace t;
   t.skip(); // optional: skips tracing for the first iteration

   for(std::size_t i{0}; i < size; ++i) {
     auto g = t.make_guard();
     s.execute<task1>();
     s.execute<task2>();
   }

In this example, the `trace` object ``t`` can be used on several regions.
The call to ``make_guard()`` creates a scope in which tasks and data movement will be recorded.
On subsequent executions of the loop, Legion reuses this trace to optimize performance.

It is important to note that during the first iteration of a loop, the communication and task execution patterns may differ from later iterations because of ghost copies.
This will lead to an error at runtime, due to an inconsistent trace.
To address this, FleCSI provides a ``skip()`` function, which tells the tracing mechanism to ignore the first iteration and begin tracing on the second.
This ensures that the recorded pattern reflects the steady-state behavior of the loop.
