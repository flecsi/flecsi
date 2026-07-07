Programming Model
*****************
Task-based programming differs from other models of computation in a number of respects.
Application code is subject to corresponding restrictions of which the programmer must be aware.
Some backends do not take advantage of every optimization opportunity afforded the implementation by these restrictions, but portable applications must not rely on the resulting behavior.

Data
++++
Between any two tasks, field data may be relocated to another memory space.
Any pointer (or reference) that is or points to (any subobject of) a field element is thereby invalidated.
A field might instead contain an index to select a value from a data structure pointed to by a task parameter.

Even an ``exec::gpu`` task executes on the host, but its accessors are references to field data stored on the device.
Those accessors are copied into the kernels launched by the task (with ``forall`` or similar), which do run on the nominal execution space and can thus use them.
Certain topology information useful for launching the kernels is copied to the host automatically.

Parallelism
+++++++++++
Whether or not the MPI backend is in use, a FleCSI application is an MPI program, perhaps running many times in parallel (although there is no requirement in general that that number be the same as the number of colors in any particular topology).
The control model actions run serially on each process and must perform the same sequence of collective calls into FleCSI with the same arguments.
(In certain cases, it is the identity rather than the value of the arguments that matters; for example, a mesh coloring might be distributed (rather than replicated) over multiple processes, but that distributed object is the same object for the purpose of initializing a topology.)
However, it is unspecified on which processes task instances execute, so it is not generally meaningful to modify data outside the task.
Moreover, tasks are normally asynchronous: ``scheduler::execute`` may return before they complete and task instances from multiple task launches may run out of order or in parallel.
To avoid dangling references, arguments passed to a task are immediately used to initialize objects of the task's decayed parameter types.
(``std::thread`` does something similar with the decayed *argument* types.)

The threads necessary to implement this impose the ordinary responsibility of thread safety among tasks as well as between them and the actions.
Because the threads may be pooled, they provide only the `parallel forward progress guarantee <https://en.cppreference.com/w/cpp/language/memory_model#Parallel_forward_progress>`_ (invalidating certain collective operations).
Because they may be implemented in user space, blocking operations provided by FleCSI (*e.g.*, ``flecsi::future::get``) may degrade the caller to be weakly parallel, with forward progress delegation provided only by other such blocking operations.
Furthermore, thread-local storage (whose utility is already limited by the pooling) may be invalidated by such an operation.

The ``flecsi::exec::executor`` operations (including the ``forall`` and ``reduceall`` macros) are asynchronous, but each waits on the previous such that values written by one may be read by another in the same task.

Optional Semantics
++++++++++++++++++
Stronger semantics for tasks are available at a performance cost.

Synchronous Tasks
^^^^^^^^^^^^^^^^^
The caller of any task can immediately call ``wait`` on its future (with known performance costs), but if it is known statically that the caller will wait it is possible to initialize task parameters directly from task arguments (even though they might be temporary objects).
Certain copies can thus be avoided without the syntactic overhead of using pointers.
Furthermore, a synchronous task can accept references to non-movable types, but only if they bind directly to the arguments (without a user-defined conversion, in the current implementation).

MPI Groups
^^^^^^^^^^
If a task accepts a ``group::match`` as (part of) a parameter, its point tasks are assigned to processes (according to MPI rank).
This control is necessary for interprocess communication but also has ancillary effects:

#. Additional memory movement and scheduling latency may be required by the processor assignment.
#. Tasks can accept pointers to non-const objects (without aliasing) and non-copyable objects by value.
#. The various calling processes can provide different argument values for non-FleCSI types.
#. Since no data relocation can be needed between two such tasks, fields used *only* by such tasks in one memory space may use non-trivial data types.

If the task is also synchronous, such that argument lifetimes are not a concern, its parameters can further be any kind of reference.
If such a task immediately follows a call to ``scheduler::wait``, no other point tasks can run concurrently, so global data can be modified without race conditions.

Concurrent Tasks
^^^^^^^^^^^^^^^^
If a task accepts a ``group::concurrent`` or ``comm::ref`` as (part of) a parameter, the concurrent forward progress guarantee is also applied to it.
(It is known that the HPX backend does not implement the guarantee perfectly and may produce a deadlock in certain situations with numerous concurrent tasks using communicators.)
This guarantee makes more parallel operations correct in such a task, but it can also impair parallelism among task launches.

The corresponding point tasks in each task to which a given ``comm`` is passed as an argument execute in launch order to avoid incorrectly ordered calls to MPI collectives on the communicator.
This ordering can also reduce parallelism if the tasks use different fields; tasks that are entirely independent should use separate ``comm`` objects.
