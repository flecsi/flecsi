.. _TUT-EM:

Execution Model
***************

FleCSI has two mechanisms for expressing work:

Tasks
  Tasks operate on data distributed to one or more address spaces and
  use data privileges to maintain memory consistency. FleCSI tasks are
  like a more flexible version of MPI that does not require the user to
  explicitly update dependencies between different ranks and which does not use static process mappings: i.e., relocatable, distributed-memory
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

----

Tasks
*****

Example 1: Single Tasks
+++++++++++++++++++++++

A `single` task launches on a single process, i.e., only one instance of
the task is executed.
This is in contrast to an `index` launch, which executes a task as a
data-parallel operation, potentially across many processes.
FleCSI uses information about the arguments passed to a task to decide
how to launch the task: If no parameter is passed that defines a `launch
domain`, e.g., an explicit *launch domain*, a `topology` instance, or a
`future map`, FleCSI will launch the task as `single`.

The ``trivial`` task is an example of a ``single`` task.
Consider the following from ``tutorial/3-execution/1-single-task.cc``:

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :start-at: // Trivial task (no arguments, no return).
  :end-at: }

Execution of the task is trivial:

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :start-at: // Execute a trivial task.
  :end-at: execute<trivial>();

A single task can return a value:

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :start-at: // Task with return value.
  :end-at: }

The return value can be retrieved with a `future`:

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :start-at: // A future is
  :end-at: } // scope

FleCSI tasks can take any valid C++ type as an argument `by-value`,
e.g., a ``std::vector``:

.. caution::

    FleCSI tasks can take any valid C++ type by value. However, because
    task data must be relocatable, you cannot pass pointer arguments or
    arguments that contain pointers.  Modifications made to by-value
    data are local to the task and will not be reflected at the call
    site.

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :start-at: // Task with by-value argument.
  :end-at: } // with_by_value_argument

Execution of such a task is what you would expect:

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :start-at: // Execute a task that takes an argument by-value.
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

Index task is a task that is executed by several processes. It is often
used to operate on different parts of the input data (like partitioned
mesh) asynchronously. 

In this example we explicitly ask to execute ``task`` on 4 processes via
the ``launch_domain`` argument.

.. literalinclude:: ../../../../tutorial/3-execution/2-index-task.cc
  :language: cpp
  :start-at: // Task with no arguments.
  :end-at: // advance()

Launch Domains
^^^^^^^^^^^^^^

Launch domain (``exec::launch_domain``) is used to define how many index
points an index task should have. If no ``launch_domain`` is passed to the
``execute`` method, the default will be used. If a topology instance is
passed the default is the number of colors that instance has.
Otherwise, the default is to launch a single task.


Example 3: MPI Tasks
++++++++++++++++++++

MPI task is an index task that has launch domain size equal to number of
MPI ranks and index points mapped to corresponding MPI ranks. Executing
MPI task adds synchronization between Legion and MPI and, therefore,
should only be used when one needs to call MPI library.  
To execute an MPI task, the second template argument to the ``execute``
method should be set to ``mpi``.

.. literalinclude:: ../../../../tutorial/3-execution/3-mpi-task.cc
  :language: cpp
  :start-at: // Task with no arguments.
  :end-at: // advance()

.. vim: set tabstop=2 shiftwidth=2 expandtab fo=cqt tw=72 :
