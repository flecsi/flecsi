Execution Model
***************
FleCSI expresses parallelism via tasks, which are coarse-grained and can be distributed, and kernels, which are fine-grained and utilize shared memory.
These are the lower levels of execution in the figure already seen:

.. figure:: tikz/flecsi_execution_model.svg

   The different levels of execution in FleCSI.

Schedulers
++++++++++

A FleCSI `scheduler` is an object that manages the launch of tasks.
Where and when to execute the task is determined based on the task's parameter types and arguments.

Schedulers are obtained from the control-model object provided to an action:

.. code-block:: c++

   void action(control_policy &cp) {
     flecsi::scheduler &s = cp.scheduler();
     s.execute<task>(/* ... */);
   }

Tasks
+++++

A task in FleCSI serves as the bridge between two core parts of the execution model.
On the caller side, each `process` executes its own copy of the ``main`` function and the associated control model.

After all processes collectively launch a task, its function is then called in a number of `task instances`.
In certain special cases (determined by the task argument types), a `single launch` occurs with just one task instance (total, not per process).
In the typical case, several task instances execute concurrently as an `index launch`.
Each of these executions is referred to as a `point task`; their number does not necessarily match the number of processes, and they can run on any process.
This allows FleCSI to support flexible and scalable task-parallel execution.

Callee Side
^^^^^^^^^^^

A task is a function that can accept certain special parameter types.
For instance, the task ``foo`` may be defined as

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

Similarly, ``std::optional`` can be used to provide zero or one field, and ``std::variant`` may be used to accept a runtime choice from a set of field types.

.. warning::

   A task with a ``std::vector``, ``std::optional``, or ``std::variant`` argument that can contain a relevant FleCSI type but does not do so (because the vector or optional is empty or the variant holds some other type) will execute zero times unless some other argument implies a launch size.

Caller Side
^^^^^^^^^^^

Tasks are launched by ``scheduler::execute``:

.. code-block:: c++

   s.execute<foo>(flecsi::exec::on, fr1, fr2);

Here, the task ``foo`` from above is specified as a template parameter.
The first argument, ``flecsi::exec::on``, represents the execution space (when one is specified by the task).

The subsequent arguments, ``fr1`` and ``fr2``, are field references passed into the task.
These field references act as logical handles to the underlying data.

Compound parameters use corresponding compound arguments:

.. code-block:: c++

   s.execute<bar>(flecsi::exec::on, std::vector{fr1, fr2, fr3});

.. _future:

Futures and Reductions
++++++++++++++++++++++

In FleCSI, tasks can return values through `futures`.
A future represents the result of a task that might not yet have completed.
The value is available only after the task is actually executed.
This asynchronous behavior allows for flexible execution ordering.

.. code-block:: c++

   int toto() noexcept { return 10; }
   // ...
   flecsi::future<int> future = s.execute<toto>();
   int value = future.get();

The ``get`` function blocks until the associated task completes and returns the resolved value of the future.
Another function, ``wait``, can be used to pause execution until the future is resolved.
Note that these can involve communication: ``toto`` is launched as a single task, but its return value is (and must be) accessed by the caller for every processes.

Having an action wait on a future's value can cause :doc:`performance` problems; the alternative is declare a task parameter as a future.
The task then runs only when the future provided as an argument is ready (so ``get`` but not ``wait`` is useful inside a task).
An index launch produces an `index future` which can also be passed to a task that accepts a normal future, in which case each point task receives the value returned by one point task in the prior launch.

Many tasks return ``void`` because their purpose is to compute new field values.
Their futures can simply be discarded: tasks that use those field values will automatically be scheduled to run only afterwards.
It can be useful for the control model to occasionally wait on all outstanding tasks to finish: to change a global variable, for instance.
In that case, ``scheduler::wait`` can be used to wait on all tasks that have been launched: it is equivalent to waiting on every future so far produced.

FleCSI supports reductions through the ``reduce`` function, which combines results from multiple point tasks into a single value. For instance:

.. code-block:: c++

   auto future1 = s.reduce<task, exec::fold::sum>( /* ... */ );

.. _reduction-folds:

The future provides the sum of the values returned by all point tasks.
FleCSI provides several built-in reduction operators, including `min`, `max`, `sum`, and `product`.
Users can define custom folds by implementing `a structure <../api/user/structflecsi_1_1exec_1_1fold_1_1reduce.html>`_ with ``combine`` and ``identity`` methods.
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
     static void task(auto, /* ... */ ) noexcept {
       /* ... */
     }
   };

The following example defines a portable task with variants for ``exec::cpu`` and ``exec::omp``.
The primary function template is deleted to indicate that no default implementation exists.
This ensures that tasks cannot be launched on the architectures that are not specified.

.. code-block:: cpp

   struct task_variants {
     static void task(auto, /* ... */ ) noexcept = delete;
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

.. _strong:

Strong Tasks
++++++++++++

In some cases, the transparent parallelism of tasks is too limiting.
Tasks can be executed with several stronger semantics, usually with a performance cost.
For example, when a task is passed a ``comm`` argument, it is executed once on each process (like the control model) to allow it to use MPI correctly.
(It can use :ref:`multi-accessors` to access fields whose number of colors does not match the number of processes.)

A ``mapping`` argument that is passed to a task also specifies which process runs each point task, without the restriction that exactly one point task runs on each process.
(Only the Legion backend can make use of that relaxation.)
Such a task can access field data in conjunction with non-field data owned by the process chosen by the ``mapping`` for that color.
It can also (carefully) modify shared data on each process via pointers or references or use only some of the ranks of a ``comm``.

If a set of index tasks that are not already ordered utilize some external resource that should not be accessed concurrently (by multiple launches), they can accept a ``point_mutex`` argument.
The name indicates that it is the corresponding point tasks in each launch that are serialized; it is still possible for point task 0 of one launch to run concurrently with point task 1 of another (which might be mitigated with appropriate synchronization within the task).

In one case, additional semantics can mitigate part of a cost: in the case where an action must wait on a task immediately after launching it, the task (class) can be declared ``synchronous`` for more flexible and efficient argument processing.
(The interface is not otherwise affected: such a task must still be declared ``noexcept`` and executing it still returns a ``future``.)

A task's parameter types both select semantics for it and are restricted by those semantics.
Given cv-unqualified object types ``X``, movable ``M``, and copyable ``C`` and a function type ``F``, the types that can be used are as follows:

all tasks
  ``C``, ``const M&``, ``const X*``, ``F*``
synchronous tasks
  ``const X&``
per-process tasks
  ``M``, ``X*``
synchronous, per-process tasks
  ``X&``, ``X&&``, ``F&``, ``F&&``

For example, a task could use a scratch file to communicate among its point tasks:

.. code-block:: c++

   void scratch(std::fstream f, Status *s,
                flecsi::field<double>::accessor<flecsi::rw> a,
                flecsi::exec::group::concurrent,
                flecsi::exec::point_mutex::lease) noexcept { /* ... */ }

The non-copyable ``f`` and the writable pointer ``s`` are allowed because the ``concurrent`` parameter makes the task run once per process.
That parameter also guarantees that the point tasks will not run sequentially (so that they can synchronize); passing the same ``point_mutex`` to multiple launches of this task guarantees that their point tasks will execute in launch order (even if ``a`` refers to a different field in each).
Passing the stream object by value rather than by pointer avoids needing to wait on the task in the calling action before destroying it.

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

In this example, the ``trace`` object ``t`` can be used on several regions.
The call ``make_guard()`` identifies a scope in which tasks and data movement will be recorded.
On subsequent executions of the loop, Legion reuses this trace to optimize performance.

It is important to note that during the first iteration of a loop, the communication and task execution patterns may differ from later iterations because of ghost copies.
This will lead to an error at runtime due to an inconsistent trace.
To address this, FleCSI provides the ``trace::skip`` function, which tells the tracing mechanism to ignore the first iteration and begin tracing on the second.
This ensures that the recorded pattern reflects the steady-state behavior of the loop.
