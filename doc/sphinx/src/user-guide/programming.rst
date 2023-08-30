Programming Model
*****************
Task-based programming differs from other models of computation in a number of respects.
Application code is subject to corresponding restrictions of which the programmer must be aware.
Some backends do not take advantage of every optimization opportunity afforded the implementation by these restrictions, but portable applications must not rely on the resulting behavior.

Data
++++
Between any two tasks, field data may be relocated to another memory space.
Any pointer (or reference) that is or points to (any subobject of) a field element is thereby invalidated.

A task need not execute in the same memory space as its caller, so task arguments must be serialized and cannot contain external pointers or references.

Data that is neither stored a field nor suitable for serialization (perhaps because it is large) must therefore typically be stored in global variables.
A field might contain an index to select a value from such a data structure instead of a pointer.

Even a ``toc`` task executes on the host, but its accessors are references to field data stored on the device.
Those accessors are copied into the kernels launched by the task (with ``forall`` or similar), which run on the device and can thus use them.
Certain topology information useful for launching the kernels is copied to the host automatically.

Parallelism
+++++++++++
Whether or not the MPI backend is in use, a FleCSI application is an MPI program, perhaps running many times in parallel (although there is no requirement in general that that number be the same as the number of colors in any particular topology).
The top-level action runs once on each process and must perform the same sequence of collective calls into FleCSI with the same arguments.
(In certain cases, it is the identity rather than the value of the arguments that matters; for example, a mesh coloring might be distributed (rather than replicated) over multiple processes, but that distributed object is the same object for the purpose of initializing a topology.)

In the common case where the top-level action invokes the actions associated with a control model, they are executed serially.
Tasks, however, are asynchronous: ``flecsi::execute`` may return before they complete and point tasks from multiple task launches may run in parallel.

The threads necessary to implement this impose the ordinary responsibility of thread safety among tasks as well as between them and the actions.
Because the threads may be pooled, they provide only the `parallel forward progress guarantee <https://en.cppreference.com/w/cpp/language/memory_model#Parallel_forward_progress>`_ (invalidating certain collective operations).
Because they may be implemented in user space, blocking operations provided by FleCSI (*e.g.*, ``flecsi::future::get``) may degrade the caller to be weakly parallel, with forward progress delegation provided only by other such blocking operations.
Furthermore, thread-local storage (whose utility is already limited by the pooling) may be invalidated by such an operation.

The ``flecsi::exec::parallel_`` operations (including the ``forall`` and ``reduceall`` macros) are asynchronous; values written by one may not be available to another in the same task.

MPI Tasks
+++++++++
The execution of ``mpi`` tasks (regardless of backend) is more predictable than that of normal tasks:

#. Because they always run one point task in each process, with access to the color corresponding to its MPI rank, no data relocation is needed between two MPI tasks.
   Therefore, fields used *only* by MPI tasks may use non-trivial data types (although resizing the field can still invalidate pointers).
#. Because they additionally run synchronously, their parameters may be of any type, including writable references.
#. Because additionally no other point tasks are executed concurrently with them, they can access global data without race conditions.
#. The resulting concurrent forward progress guarantee makes it valid for them to perform MPI communication (including via ``MPI_COMM_WORLD``).
   (Their field data is stored on the host and thus may be accessed directly by MPI.)
