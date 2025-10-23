Execution Model
***************

FleCSI has two mechanisms for expressing work:

Tasks
  Tasks operate on data distributed to one or more address spaces and
  use data privileges to maintain memory consistency. FleCSI tasks are
  like a more flexible version of MPI that does not require the user to
  explicitly copy data between processes and which does not use static process mappings: i.e., relocatable, distributed-memory
  data parallelism.

Kernels
  Kernels operate on data in a single address space but require
  explicit barriers to ensure consistency. This is generally referred to
  as a relaxed-consistency memory model. The kernel interface in
  FleCSI is defined by two parallel operations: *forall* and *reduceall*.
  Each of these is a fine-grained, data-parallel operation.
  The use of the *kernel* nomenclature is derived from CUDA and OpenCL
  and is conceptually consistent with those models. Please see the
  example of using *forall* kernels in the *parallel* section of the
  tutorial. 

Tasks are launched by *schedulers*.

----

Example 1: Single Tasks
+++++++++++++++++++++++
A `single` task launch calls a given function just once (across all processes).
This is in contrast to an `index` launch, which executes a task as a
data-parallel operation, potentially across many processes.

The ``trivial`` task is an example of a ``single`` task.
Consider the following from ``tutorial/3-execution/1-single-task.cc``:

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :start-at: // Trivial task (no arguments, no return).
  :end-at: }

Since they are not invoked directly, tasks cannot throw exceptions and must be declared ``noexcept``.
Execution of the task is a trivial use of the ``scheduler`` provided to the action:

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :start-at: // Execute a trivial task.
  :end-at: execute<trivial>();

A single task can return a value:

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :start-at: // Task with return value.
  :end-at: }

The return value can be retrieved with a ``future``:

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :start-at: // A future is
  :end-at: } // scope

Tasks can take many non-trivial C++ types as parameters,
e.g., a ``std::vector``:

.. caution::

    Because they run asynchronously and not necessarily the same number of times as their callers, normal tasks cannot accept pointers or references to non-const types.

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :start-at: non-trivial parameter
  :end-at: } //

Execution of such a task is what you would expect:

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :start-at: non-trivial argument
  :end-at: } // scope

FleCSI tasks can also be templated:

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :start-at: template<typename Type>
  :end-at: } // template

Again, execution is straightforward:

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :start-at: // Execute a templated task.
  :end-at: } // scope

Example 2: Index Tasks
++++++++++++++++++++++
An index task launch calls a given function a number of times asynchronously, typically distributed over multiple processes; each is called a *point task*.
The usual purpose is operating on different parts of a distributed data structure (different *colors* of a *topology*) in parallel.

In this example we explicitly ask to call ``task`` 4 times via
the ``launch_domain`` argument; the task must declare a parameter for it, but it need not be named or used.
To receive information about the task launch, a task can declare an *execution space* parameter; the task launch provides the dummy value ``exec::on`` to initialize it.
An execution space parameter also controls where the task runs; ``exec::cpu`` is the default, but others will be used later.

.. literalinclude:: ../../../../tutorial/3-execution/2-index-task.cc
  :language: cpp
  :start-at: // Task with special arguments.
  :end-at: // advance()

Launch Domains
^^^^^^^^^^^^^^

Launch domain (``exec::launch_domain``) is used to define how many index
points an index task should have. If no ``launch_domain`` is passed to the
``execute`` method, the default will be used.
If the task uses a field or topology accessor, the default is the number of colors of the topology used.
If no argument indicates a number, the default is to launch a single task.


Example 3: MPI Tasks
++++++++++++++++++++

MPI task is an index task that has launch domain size equal to number of
processes and index points mapped to corresponding MPI ranks.
Executing an
MPI task adds synchronization between Legion and MPI and, therefore,
should only be used when one needs to call MPI library.  
To execute an MPI task, ``flecsi::execute`` must be used, with its second template argument set to ``mpi``.
The ``launch`` information provided is equivalent to ``process`` and ``processes``.

.. literalinclude:: ../../../../tutorial/3-execution/3-mpi-task.cc
  :language: cpp
  :start-at: // Task with no arguments.
  :end-at: // advance()

.. vim: set tabstop=2 shiftwidth=2 expandtab fo=cqt tw=72 :
