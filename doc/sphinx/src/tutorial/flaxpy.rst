.. |br| raw:: html

   <br />

DAXPY
*****

This example presents a simple but complete example of a FleCSI
program that manipulates a distributed data structure in parallel.
Specifically, it uses FleCSI to perform a distributed `DAXPY
<http://www.netlib.org/lapack/explore-html/de/da4/group__double__blas__level1_ga8f99d6a644d3396aa32db472e0cfc91c.html>`_
(double-precision *aX* + *Y*) vector operation.  The sequential C++
code looks like this:

.. code:: C++

    for (int i = 0; i < N; ++i)
      y[i] += a*x[i];

To further demonstrate FleCSI code structure and features we will also
allocate and initialize the distributed vectors before the DAXPY
operation and report the sum of all elements in *Y* at the end, as in
the following sequential code:

.. code:: C++

    // Allocate and initialize the two vectors.
    std::vector<double> x(N), y(N);
    for (size_t i = 0; i < N; ++i) {
      x[i] = static_cast<double>(i);
      y[i] = 0.0;
    }

    // Perform the DAXPY operation.
    const double a = 12.34;
    for (size_t i = 0; i < N; ++i)
      y[i] += a*x[i];

    // Report the sum over y.
    double sum = 0.0;
    for (size_t i = 0; i < N; ++i)
      sum += y[i];
    std::cout << "The sum over all elements in the final vector is " << sum << std::endl;

For pedagogical purposes the FleCSI version of DAXPY, which we'll call
"FLAXPY", will be expressed slightly differently from how a full
application would more naturally be implemented:

* FLAXPY is presented as a single file rather than as separate (and
  typically multiple) source and header files.

* C++ namespaces are referenced explicitly rather than imported with
  ``use``.

* Some function, method, and variable names are more verbose than they
  would normally be.


Preliminaries
+++++++++++++

Here's a simple ``CMakeLists.txt`` file for building FLAXPY:

.. literalinclude:: ../../../../tutorial/standalone/flaxpy/CMakeLists.txt
   :language: cpp
   :lines: 6-15

FLAXPY is implemented as a single file, ``flaxpy.cc``.  We begin by
including the header files needed to access the data model, execution
model, and other FleCSI components:

.. literalinclude:: ../../../../tutorial/standalone/flaxpy/flaxpy.cc
   :language: cpp
   :lines: 6-10

For user convenience, we define a ``--length`` (abbreviation: ``-l``)
command-line option for specifying the length of vectors *X* and *Y*,
with a default of 1,000,000 elements.  To do so we construct a
variable called ``vector_length`` of type ``flecsi::program_option``
(templated on the option type, ``std::size_t``) and within a
``flaxpy`` namespace that other source files—of which there are none
in this simple example—could import.  ``vector_length`` will be used
at run time to access the vector length.

.. literalinclude:: ../../../../tutorial/standalone/flaxpy/flaxpy.cc
   :language: cpp
   :lines: 12-21


Data structures
+++++++++++++++

FleCSI does not provide ready-to-use, distributed data structures.
Rather, it provides "proto data structures" called *topologies*.
These require additional compile-time information, such as the number
of a dimensions of a multidimensional array, and additional run-time
information, such as how to distribute their data, to form a concrete
data structure.  Application-defined *specializations* provide all of
this information.

FLAXPY is based on the ``user`` topology, so named because it is
arguably the simplest topology that behaves as a user would expect.
It is essentially a 1-D vector of user-defined *fields*.  All
topologies specify the type of a *coloring*, which is additional
run-time data the topology needs to produce a concrete data structure.
A specialization must define a ``color`` method that accepts whatever
parameters make sense for that specialization and returns a
topology-specific ``coloring``.  The ``user`` topology defines a
``coloring`` as a ``std::vector<std::size_t>`` that indicates the
number of vector indices to assign to each *color*.  (One can think of
a color as being like an MPI rank: a globally unique identifier for a
unit of computation.)  ``user`` does not require that the
specialization provide any compile-time information, but most
topologies do.

For FLAXPY we divide the indices as equally as possible among colors.
The following helper function, still within the ``flaxpy`` namespace,
handles mapping ``vector_length`` indices (see `Preliminaries`_ above)
onto a given number of colors:

.. literalinclude:: ../../../../tutorial/standalone/flaxpy/flaxpy.cc
   :language: cpp
   :lines: 25-36

Given that helper function, constructing a specialization of ``user``
is trivial.  FLAXPY names its specialization (still within the
``flaxpy`` namespace) ``dist_vector``:

.. literalinclude:: ../../../../tutorial/standalone/flaxpy/flaxpy.cc
   :language: cpp
   :lines: 40-47

Note that the specialization chooses the number of colors.
``dist_vector`` queries FleCSI for the number of processes and uses
that value for the color count.

At this point we have what is effectively a distributed 1-D vector
data type, templated over the element type.  The next step is to
indicate the element type required by the application.  Each element
comprises one or more *fields* of values.  FLAXPY adds two fields of
type ``double``: ``x_field`` and ``y_field``.  These are added outside
of the ``flaxpy`` namespace.  ``flaxpy.cc`` uses an anonymous
namespace to indicate that these fields are meaningful only locally
and not needed by the rest of the application.

.. literalinclude:: ../../../../tutorial/standalone/flaxpy/flaxpy.cc
   :language: cpp
   :lines: 96-98

``one_field`` is defined in the above to save typing.
``flecsi::data::layout::dense`` is in fact the default for a
``flecsi::field`` and is included to show where layouts are indicated.

Specializations typically require run-time information to produce a
usable object.  This information may not be available until a number
of libraries (FleCSI, Legion, MPI, and the like) have initialized and
perhaps synchronized across a distributed system.  To prevent
applications from directly constructing an object of a specialization
type and accessing this object before library initialization and
synchronization have completed, FleCSI imposes a particular means of
instantiating a specialization based on *slots*.  The following lines
of code declare the slot and *coloring slot* that will be used during
FLAXPY initialization to allocate distributed storage for its
distributed vector:

.. literalinclude:: ../../../../tutorial/standalone/flaxpy/flaxpy.cc
   :language: cpp
   :lines: 101-102


Control flow
++++++++++++

A FleCSI application's control flow is defined with a three-level
hierarchy.  *Control points* define the sequential skeleton of the
control flow and can include unbounded iteration (e.g., to repeat a
sequence of steps until convergence).  Each control point is
associated with a collection of *action*.  Actions form a directed
acyclic graph (DAG) and thereby support concurrent execution but no
iteration of tasks within the DAG.  Actions spawn *tasks*, which
manipulate distributed data.

The bulk of this section is presented in top-down fashion.  That is,
function invocations are presented before the functions being invoked
are defined.  With the exception of the code appearing in `Control
points`_, all of the control-flow code about to be presented appears
in an anonymous namespace, again to indicate that it is meaningful
only locally and not needed by the rest of the application.

A roadmap for FLAXPY's control-flow code is presented in
:numref:`flaxpy_control`.  `Control points`_ are drawn as white
rounded rectangles; `actions <Actions_>`_ are drawn as blue ellipses;
and `tasks <Tasks_>`_ are drawn as green rectangles.  As indicated by
the figure, FLAXPY is a simple application and uses a trivial sequence
of control points (no looping), a trivial DAG of actions (comprising a
single node), and trivial task launches (exactly one per action).

.. _flaxpy_control:
.. figure:: images/flaxpy-control-model.svg
   :align: center

   FLAXPY control model


Control points
--------------

FLAXPY defines three control points: ``initialize``, ``mul_add``, and
``finalize``.  These are introduced via an enumerated type, which
FLAXPY calls ``cp`` and defines within the ``flaxpy`` namespace:

.. literalinclude:: ../../../../tutorial/standalone/flaxpy/flaxpy.cc
   :language: cpp
   :lines: 50

FleCSI expects to be able to convert a ``cp`` to a string by
dereferencing it.  This requires overloading the ``*`` operator as
follows, still within the ``flaxpy`` namespace:

.. literalinclude:: ../../../../tutorial/standalone/flaxpy/flaxpy.cc
   :language: cpp
   :lines: 56-67

Once an application defines its control points it specifies a
sequential order for them to execute.  FLAXPY indicates with the
following code that ``initialize`` runs first, then ``mul_add``, and
lastly ``finalize``:

.. literalinclude:: ../../../../tutorial/standalone/flaxpy/flaxpy.cc
   :language: cpp
   :lines: 71-79

FLAXPY's ``control_policy`` class is used to define a fully qualified
control type that implements the control policy:

.. literalinclude:: ../../../../tutorial/standalone/flaxpy/flaxpy.cc
   :language: cpp
   :lines: 82


Actions
-------

Actions, implemented as C++ functions, must be associated with control
points.  The following code associates the ``initialize_action()``
action with the ``initialize`` control point, ``mul_add_action()``
with the ``mul_add`` control point, and ``finalize_action()`` with the
``finalize`` control point:

.. literalinclude:: ../../../../tutorial/standalone/flaxpy/flaxpy.cc
   :language: cpp
   :lines: 185-187

The variables declared by the preceding code (``init``, ``ma``, and
``fin``) are never used.  They exist only for the side effects induced
by instantiating a ``flaxpy::control::action``.

The ``initialize_action()`` action uses the slot and coloring slot
defined above in `Data structures`_ to allocate memory for the
``dist_vector`` specialization.  It then spawns the
``initialize_vectors_task()`` task, passing it *X* and *Y* via the
``x_field`` and ``y_field`` fields.

.. literalinclude:: ../../../../tutorial/standalone/flaxpy/flaxpy.cc
   :language: cpp
   :lines: 127-134

The ``mul_add_action()`` action spawns the ``mul_add_task()`` task,
also passing it *X* and *Y* via the ``x_field`` and ``y_field`` fields
as well as a scalar constant, *a*.

.. literalinclude:: ../../../../tutorial/standalone/flaxpy/flaxpy.cc
   :language: cpp
   :lines: 147-153

The third and final action, ``finalize_action()``, sums the elements
of *Y* by spawning a global-reduction task (of type ``flecsi::reduce``
instead of the ``flecsi::execute`` used in the preceding two actions).
It then uses the FleCSI logging facility, FLOG, to output the sum.
Finally, it deallocates the memory allocated by
``initialize_action()``.

.. literalinclude:: ../../../../tutorial/standalone/flaxpy/flaxpy.cc
   :language: cpp
   :lines: 170-180


Tasks
-----

Tasks are functions that concurrently process a partition of a
distributed data structure.  Because FleCSI follows `Legion
<https://legion.stanford.edu/>`_'s data and concurrency model, a task
is provided access to a data partition via an *accessor* templated on
the requested access rights: ``ro`` (read only), ``wo`` (write only),
``rw`` (read/write), or ``na`` (no access).

The ``initialize_vectors_task()`` task requests write-only access to a
partition of *X* and a partition of *Y*.  It uses
``divide_indices_among_colors``, defined above in `Data structures`_,
to compute the number of vector indices to which this instance of the
task has access and the global index corresponding to local index 0.
Once these are known, the task initializes *X*\ [*i*] ← *i* and *Y*\
[*i*] ← 0 for its subset of the distributed *X* and *Y* vectors.
FLAXPY uses FleCSI's ``forall`` macro to locally parallelize (e.g.,
using thread parallelism) the initialization of *Y*.

.. literalinclude:: ../../../../tutorial/standalone/flaxpy/flaxpy.cc
   :language: cpp
   :lines: 105-124

``mul_add_task()`` is the simplest of FLAXPY's three tasks but also
the one that performs the core DAXPY computation.  It accepts a scalar
*a* and requests read-only access to a partition of *X* and read/write
access to a partition of *Y*.  The task then computes *Y*\ [*i*] ←
*a*\ ⋅\ *X*\ [*i*] + *Y*\ [*i*] over its subset of the distributed *X*
and *Y* vectors.

.. literalinclude:: ../../../../tutorial/standalone/flaxpy/flaxpy.cc
   :language: cpp
   :lines: 137-144

The third and final task, ``reduce_y_task()`` computes and returns the
sum of a partition of *Y*.  For this it requests read/write access to
the partition and uses FleCSI's ``reduceall`` macro to locally
parallelize (e.g., using thread parallelism) the summation.

.. literalinclude:: ../../../../tutorial/standalone/flaxpy/flaxpy.cc
   :language: cpp
   :lines: 156-167
