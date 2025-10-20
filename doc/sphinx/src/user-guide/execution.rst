Execution Model
***************

This section describes the FleCSI execution model, which is based on the concepts of tasks and kernels.
These abstractions allow users to express parallelism at different levels, from task scheduling to fine-grained data access.

Schedulers
++++++++++

A FleCSI *scheduler* is an object that manages the launch of tasks.
Where and when to execute the task is determined based on the task's parameter.
They select the appropriate execution space based on the task and platform.

Schedulers are obtained from a *control_point* when writing an action inside the control model:

.. code-block:: c++

   void action(control_point &cp) {
     flecsi::scheduler s = cp.scheduler();
     s.execute<task>(...);
   }

Tasks
+++++

A task in FleCSI serves as the bridge between two core parts of the execution model.
On the caller side, each ``process``, also referred to as a ``rank`` in the MPI terminology, executes its own copy of the ``main`` function and the associated control model.

Each process executes its own instance of ``main`` and the control model, and so all processes collectively launch and schedule a task based on its :ref:`field privileges <field-accessors>`.
The function for a task then executes, typically several times concurrently.
Each of these executions is referred to as a *point task*; their number does not necessarily match the number of processes, and they can run on any process.
This allows FleCSI to support flexible and scalable task-parallel execution.

.. figure:: tikz/flecsi_execution_model.svg

   The different levels of abstraction in FleCSI.

Callee Side
^^^^^^^^^^^

A task is a function that can accept certain special parameter types.
For instance, the task ``foo`` may be defined as:

.. code-block:: c++

   void foo(flecsi::exec::cpu e,
     flecsi::field<double>::accessor<ro, ro> f1,
     flecsi::field<std::size_t>::accessor<rw, ro> f2) noexcept { /* ... */ }

Tasks must be declared ``noexcept`` because they are scheduled and executed indirectly and cannot throw exceptions.
The first parameter, ``e``, specifies the execution space where the task runs.
Available options include `cpu`, `omp`, `gpu`, or `accelerator`.
The `accelerator` space is flexible and will use the backend selected when compiling Kokkos (e.g., `Serial`, `OpenMP`, `CUDA`, etc.).
This allows users to target different hardware execution spaces without modifying the task code.
If the execution space is not explicitly specified in the callee declaration, the task will default to CPU execution.

The field references provided during the task launch are automatically converted into :ref:`field-accessors`.
These accessor types grant access to the real data, potentially after hidden memory copies or device transfers, depending on the privileges set by the user.

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

Tasks are launched by ``scheduler::execute``.
This function launches tasks into the execution space specified by the user.
For example, consider a task ``baz``:

.. code-block:: c++

   s.execute<baz>(flecsi::exec::on, fr1, fr2);

Here, the task ``baz`` is specified as a template parameter.
The first argument, ``flecsi::exec::on``, represents the execution space (when one is specified by the task).
If no execution space is specified in the callee declaration, the task defaults to CPU execution.

The subsequent arguments, ``fr1`` and ``fr2``, are :ref:`fields` references passed into the task.
These field references act as logical handles to the underlying data.

.. _future:

Futures and Reductions
++++++++++++++++++++++

In FleCSI, tasks can return values through *futures*.
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

This performs a sum across all point tasks.
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

FleCSI also supports a special class of tasks, known as *MPI tasks*.
These tasks ensure that the number of index points matches the number of MPI processes.
As a result, each task instance runs in the same memory space as its corresponding process.

MPI tasks are invoked as follows:

.. code-block:: c++

   flecsi::execute<qux, flecsi::mpi>(flecsi::exec::on, fr1, fr2);

.. note::
   Unlike standard tasks, MPI tasks are currently invoked directly via the ``flecsi::`` namespace rather than using the scheduler object with ``s.execute``.
   Eventually, MPI tasks may be launched through the scheduler interface for consistency with other task types.

MPI tasks can use :ref:`multi-accessors` to access fields whose number of colors does not match the number of processes.

.. _tracing:

Tracing
+++++++

The Legion backend uses a feature called *tracing* to improve performance for repeated execution patterns, particularly in critical loops.

When tracing is enabled, Legion records task launches, data movement, and communication patterns during the first execution.
On subsequent iterations, it reuses this recorded information instead of redoing scheduling and analysis, significantly reducing overhead.

Here is an example of tracing in use:

.. code-block:: c++

   static exec::trace t;
   t.skip(); // optional: skips tracing for the first iteration

   auto g = t.make_guard();
   for(std::size_t i{0}; i < size; ++i) {
     execute<task1>();
     execute<task2>();
   }

In this example, the `trace` object ``t`` can be used on several regions.
The call to ``make_guard()`` creates a scope in which tasks and data movement will be recorded.
On subsequent executions of the loop, Legion reuses this trace to optimize performance.

It is important to note that during the first iteration of a loop, the communication and task execution patterns may differ slightly from later iterations.
This will lead to an error at runtime, due to an inconsistent trace.
To address this, FleCSI provides a ``skip()`` function, which tells the tracing mechanism to ignore the first iteration and begin tracing on the second.
This ensures that the recorded pattern more accurately reflects the steady-state behavior of the loop.
