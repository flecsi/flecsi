.. |br| raw:: html

   <br />

.. _TUT-DM:

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
  :lines: 13-17

Writing to a global field requires a single task launch.

.. literalinclude:: ../../../../tutorial/4-data/1-global.cc
  :language: cpp
  :lines: 19-35

Example 2: Index data 
+++++++++++++++++++++

A field on an ``index`` topology stores one value for each color.

.. literalinclude:: ../../../../tutorial/4-data/2-index.cc
  :language: cpp
  :lines: 12-38

Example 3: Dense data
+++++++++++++++++++++

A dense field is a field defined on a dense topology index space.  In
this example we allocate a `pressure` field on the `cells` index space
of the `canonical` topology.

.. literalinclude:: ../../../../tutorial/4-data/3-dense.cc
  :language: cpp
  :lines: 16-16

One can access the field inside of the FleCSI task by passing
topology and field accessors with `access permissions` (wo/rw/ro).  
The ``canonical`` topology is a very simple specialization of the ``unstructured`` core topology.
It illustrates the use of a *coloring slot*, which applies a specialization-defined rule for specifying a *coloring*: the run-time information for constructing a topology distributed over colors.
Here, a file is the source of the mesh (for purposes of illustration).
The resulting coloring is used to initialize two meshes ``canonical`` and ``cp``, and the ``copy`` task operates on both of them at once using a low-level accessor.
The ``init`` and ``print`` tasks, by contrast, use a *topology accessor* as a parameter that provides access to the structure of the mesh via the ``entities`` function.

.. literalinclude:: ../../../../tutorial/4-data/3-dense.cc
  :language: cpp
  :lines: 18-51

----

.. vim: set tabstop=2 shiftwidth=2 expandtab fo=cqt tw=72 :
