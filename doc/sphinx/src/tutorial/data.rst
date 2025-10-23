Data Model
**********

FleCSI provides a data model that integrates with the task and kernel
abstractions to provide easy registration and access to various data
types with automatic dependency tracking.

.. sidebar:: Field definitions in headers

   Declaring field definitions as ``const`` namespace members gives them
   internal linkage. When putting them in headers make sure to declare them
   as ``inline const`` to avoid breaking the One-Definition Rule (ODR).

Example 1: Global data
++++++++++++++++++++++

Global fields are used to store global variables/objects that can be
accessed by any task.  For the common case of only one value for each field, it
is natural to use the ``data::single`` layout.

.. literalinclude:: ../../../../tutorial/4-data/1-global.cc
  :language: cpp
  :start-at: template<typename T>
  :end-at: const single<double>::definition<global> gfield;

To create a topology instance, declare a variable of type ``topology`` initialized with a scheduler and an argument appropriate to the topology called a *coloring*.
In general, a coloring describes the structure of a topology and its distribution among colors.
The global topology is a special case that does not actually use colors; its "coloring" is simply a count of values for each field.
Writing to a global field requires a single task launch.

.. literalinclude:: ../../../../tutorial/4-data/1-global.cc
  :language: cpp
  :start-at: void
  :end-at: // advance()

The first task that accesses a field must do so with ``wo`` privileges, since there are no previous values to read.
Field elements are default-initialized when first created; this does not give any usable value for primitive types like ``double``.
``std::fill`` can be used to set an initial value if required.

Example 2: Index data 
+++++++++++++++++++++

A field on an ``index`` topology stores one value for each color.

.. literalinclude:: ../../../../tutorial/4-data/2-index.cc
  :language: cpp
  :start-at: using namespace flecsi;
  :end-at: // advance()

Example 3: Dense data
+++++++++++++++++++++

A dense field is a field defined on a dense topology index space.  In
this example we allocate a ``pressure`` field on the ``cells`` index space of the ``canonical`` topology.

.. literalinclude:: ../../../../tutorial/4-data/3-dense.cc
  :language: cpp
  :start-at: const field<double>::definition<canon, canon::cells> pressure;
  :end-at: const field<double>::definition<canon, canon::cells> pressure;

One can access the field inside of the FleCSI task by passing
topology and field accessors with `access permissions` (wo/rw/ro).  
The ``canonical`` topology is a very simple specialization of the ``unstructured`` topology category.
It illustrates the use of the ``mpi_coloring`` type, which applies a specialization-defined rule for specifying a coloring.
Here, a file is the source of the mesh (for purposes of illustration).
The resulting coloring is used to initialize two meshes ``canonical`` and ``cp``, and the ``copy`` task operates on both of them at once using a low-level accessor.
The ``init`` and ``print`` tasks, by contrast, use a *topology accessor* as a parameter that provides access to the structure of the mesh via the ``entities`` function.

.. literalinclude:: ../../../../tutorial/4-data/3-dense.cc
  :language: cpp
  :start-at: void
  :end-at: // advance()

Example 4: Ragged data
++++++++++++++++++++++
A ragged field stores a variable amount of data at each index point.
It is defined in much the same way as a dense field, but using it involves additional steps.

.. literalinclude:: ../../../../tutorial/4-data/4-ragged.cc
  :language: cpp
  :start-at: ints
  :end-at: rag;

Because the total storage required for a ragged field is not determined by the size of the index space, an amount of memory to use for elements must be specified.
The field reference for a ragged field has a special interface for managing it.
Here we use a simple helper task (called ``allocate``) with that interface to specify a fixed, known total, although typically a heuristic overallocation is required:

.. literalinclude:: ../../../../tutorial/4-data/4-ragged.cc
  :language: cpp
  :start-at: predefined
  :end-at: }

After executing a task to store the sizes, they must be applied with ``resize``:

.. literalinclude:: ../../../../tutorial/4-data/4-ragged.cc
  :language: cpp
  :start-at: rag(mesh)
  :end-at: resize

Initializing the field, or changing the number of values stored at any point later, requires a *mutator*.
The interface is closely modeled on ``std::vector``:

.. literalinclude:: ../../../../tutorial/4-data/4-ragged.cc
  :language: cpp
  :start-at: initialize
  :end-before: Accessors

Using the field is much like a dense field, with a ``span`` at each index point:

.. literalinclude:: ../../../../tutorial/4-data/4-ragged.cc
  :language: cpp
  :start-at: Accessors
  :end-at: }

Tasks using either of these are launched with ordinary field references:

.. literalinclude:: ../../../../tutorial/4-data/4-ragged.cc
  :language: cpp
  :start-at: <init>
  :end-at: throw

----

.. vim: set tabstop=2 shiftwidth=2 expandtab fo=cqt tw=72 :
