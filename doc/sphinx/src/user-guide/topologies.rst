Topologies
**********
A *topology* is a distributed-memory object that stores user-registered fields on one or more *index spaces*.
It may also store structural information used to interpret those fields in terms appropriate to the *category* of topology (*e.g.*, an unstructured mesh).
Any number of instances may be created of any topology.

Index Spaces
++++++++++++
In FleCSI, index spaces are used to define field arrays that represent
the user's data. In simple terms, you can think of an index space as
just an enumeration of an array, with the added notion that an index
space represents a logical space of such arrays.

.. admonition:: Definition

  An index space is the space of possible enumerations of a logical set
  of points or indices.
  
As an example, consider the cells of a mesh. These represent a set that
can be enumerated to define an index space. If a particular mesh
instance has 100 cells, these cells define a vector (or index set) in
the index space of cells. *We often still refer to this as an index
space or index space instance.* The vertices and edges of the mesh also
define index spaces. In fact, in FleCSI, index spaces are used to
represent the logical elements or entities of all of our topology types.

There are several benefits of index spaces. One is that they can be
iterated upon or over.  As an example, consider a simple *for* loop:

.. code-block:: cpp

  for(auto i: mesh.cells()) {

    // Do something work on the ith
    // index of the cells index space.

  } // for

This example is just the C++ version of what was stated above about the
cells of a mesh defining an index space. However, as we will see in the
following sections, index spaces also improve our ability to reason about
parallelism, and free us from many of the computer science details that
obfuscate our algorithms.

Basic Categories
++++++++++++++++
The most basic topology provided by FleCSI is called the *global*
topology. It has a single implicit index space that has a single
implicit index: i.e., it provides a singleton of data that can be passed
into a FleCSI task.

The next most basic topology type provided by FleCSI is the *index*
topology. Like the global topology, it has a single implicit index
space, which you can think of as the *indices*. The index topology also
has a runtime-specified size that describes how many indices there
should be.

Field Registration
++++++++++++++++++
Let's look at an example of how to register fields against
the global and index topologies:

.. code-block:: cpp

  using namespace flecsi;

  using double_field = field<double, data::single>;

  namespace solver {
    const double_field::definition<topo::global> tolerance;
  }
  namespace hydro {
    const double_field::definition<topo::index> index_data[2];
  }

The first variable declaration in this example registers a field called ``solver::tolerance`` with type ``double``.
The second registers two fields called ``hydro::index_data`` with type ``double``.
(An array of fields can be used for data that logically have multiple states, as in a multi-step time evolution method.)

In both cases, the user does not need to explicitly specify index space.
As we will see, this is necessary for more complex
topology types that can be customized by a *specialization*.

Logically, registering a field against a topology type adds that field
to the type itself. This may not be intuitive to everyone, so let's
consider what this means. When we create a C++ type, e.g., a class or
struct, we add data members to it in the definition of the type:

.. code-block:: cpp

  struct field_data_t {
    double field_a;
    int field_b;
  }; // struct topology_t

This is done manually at the time the type is written. With FleCSI,
registering fields against a FleCSI topology type is a kind of
customization that is logically equivalent to our *field_data_t* example
type. So, for instance, if we want fields *field_a*, and *field_b* to be
defined for the FleCSI index topology, we would register them like so:

.. code-block:: cpp

  namespace radiation {
    const field<double, data::single>::definition<topo::index> field_a;
    const field<int, data::single>::definition<topo::index> field_b;
  }

Optionally, we could also just register the field_data_t struct:

.. code-block:: cpp

  namespace radiation {
    const field<field_data_t, data::single>::definition<topo::index> fields;
  }

Both of these methods of registering fields are valid, and it is left up
to the user to decide which way makes the most sense. The performance
implications of choosing one method over the other are equivalent to
choosing *array-of-struct (AoS)* or *struct-of-array (SoA)*. FleCSI does
not currently support switching or auto-tuning of the data layout.
However, we may do so in future versions.

.. sidebar:: Memory Allocation

  You may be wondering whether or not field registration in FleCSI
  implies that every instance of a topology type will necessarily create
  an instance of every registered field. This is a valid concern! The
  answer is *no!* FleCSI will only allocate memory for a field instance
  if it is actually accessed.

Colorings
+++++++++

Let's continue discussing the index topology so that we can add some
more details about its index space and define what *coloring* means in
FleCSI.

As stated above, the index topology has a single implicit index space.
For the index topology, we can think of the implicit index space as just
being the indices, with a particular instance being defined by its size.

The index topology also has an implicit coloring that
assigns each index of the topology's indices to its own color: i.e.,
index 0 is assigned to color 0, etc. This simple example illustrates the
definition of a coloring:

.. admonition:: Definition

  A coloring is a description of how the indices of an index space
  should be divided into partitions or colors.

The *size* of the default index topology instance is taken from the
number of processes with which the FleCSI runtime was launched. This is
a special case. In general, there is no implied size for a coloring, and
no association with the details of a particular execution space, i.e.,
the number of processes.  A coloring only describes how to divide the
indices of an index space into partitions (or colors in FleCSI's
nomenclature).

.. attention::

  A coloring is not associated with an execution space. This is
  different from the way that many people think about MPI, where a rank
  is statically mapped to a particular process.

Users are allowed to add addtional named instances of the global or index topologies.
Let's see how:

.. code-block:: cpp

  using namespace flecsi;

  topo::index::slot hydro_indices;

  void initialize() {  // called from the top-level action
    hydro_indices.allocate(42);
  }

.. vim: set tabstop=2 shiftwidth=2 expandtab fo=cqt tw=72 :
