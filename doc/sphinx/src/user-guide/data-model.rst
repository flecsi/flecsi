************************************************************************
Data Model
************************************************************************

FleCSI's data model provides the tools necessary to define parallel data
structures customized to your application, and to enable them to
interact efficiently with FleCSI's control model.  We begin with a brief
overview of the important concepts, as they interact with each other and
a detailed discussion cannot completely separate them out.  This will be
followed by a deeper dive into each topic.  Alongside the deeper dive,
in order to help illustrate concepts, we will develop the framework for
a simple simulation code along the way.  This framework will be designed
for finite-volume hydrodynamics on a three-dimensional, structured
(Cartesian-product), static grid.

Important Concepts
========================================================================

**Index Spaces**
    An *index space* is an enumeration of entities.  Different types of
    entities (such as cells or nodes or particles) each have their own
    index space.  FleCSI then defines an ordering of each kind of entity
    and assigns an index to each entity.  Each entity in an index space
    is also called an *index point*.

**Colors**
    FleCSI decomposes its data across a set of *colors*, each of which
    is a subset of the total data.  This decomposition is used to
    distribute calculations across multiple processing elements.  A
    color is not strictly bound to a particular memory space, but can be
    relocated by the task-based parallelism machinery depending on the
    needs of the application.

**Fields**
    A *field* defines a variable that exists in a particular index
    space, such as the mass in a cell or the momentum vector of a
    particle.  It provides the necessary information for an instance of
    that topology type to manage the memory for that variable.  A field
    object also acts as an accessor to retrieve the data for that
    variable from an instance of that topology type.

**Layouts**
    Every field has a *layout*, which defines information about the data
    container for the field.  This allows users to define, for example,
    scalar values (such as the kinetic energy of a particle) or arrays
    of values (such as a list of materials in a cell).

Colors
========================================================================

In order to parallelize our example simulation, we will decompose our
grid into blocks of cells.  FleCSI refers to these blocks as colors.
How the index spaces are distributed to different colors is the job of
the *coloring*, which is defined by the topology and/or specialization.
For our example, we should get good performance by choosing cells that
are physically close together to be within the same color, and by
choosing faces to be decomposed similarly to the cells.

.. attention::

  A color is not associated with an execution space.  This is
  different from the way that many people think about MPI, where a rank
  is statically mapped to a particular process.

Bear in mind that colors are not bound to execution spaces, but can be
relocated depending on the needs of the simulation.  Consider a few
possibilities within the context of our hydrodynamics example:

* You may know that some regions of your simulation domain will be more
  expensive to compute.  In this case, one processor will take a long
  time to compute a single "expensive" color, while other processors
  finish their work quickly and wait for the next task.  In this case,
  you may want to create more colors than there are execution spaces so
  that FleCSI can pack multiple "inexpensive" blocks together in a
  single execution space.

* You may have multiple physics packages that can run in parallel, such
  as a hydrodynamics solver and an energy source.  In that situation you
  could use fewer colors than processes so that some processors can
  perform calculations for the hydrodynamics and other processors can
  perform calculations for the energy source, and they can run at the
  same time.

While index points can move between colors (effectively creating a new
coloring), the total number of colors remains fixed.

.. _ghost-elements:

Ghost Elements
------------------------------------------------------------------------

Some topologies provide tools to support communication across colors
through ghost elements.

* A *ghost element* is an index point that belongs to another color, but
  a copy is provided to the current color.  An example would be the use
  of ghost cells in stencil-based codes so that each cell can read its
  neighbors' values and compute gradients.

* An index point that belongs to the current color but could be copied
  to be a ghost element of another color is called a *shared element*.

* Any index points belonging to the current color that are never copied
  to be ghost elements are known as *exclusive elements*.

The read/write permissions (see :ref:`field-accessors` for more detail)
may distinguish between these three varieties of index points in order
to expose additional opportunities for parallelization.

Fields
========================================================================

.. sidebar:: Memory Allocation

  You may be wondering whether or not field registration in FleCSI
  implies that every instance of a topology type will necessarily create
  an instance of every registered field.  This is a valid concern!  The
  answer is *no*!  FleCSI will only allocate memory for a field instance
  if it is actually accessed.

Now that we have defined our grid, we need to store data on the grid in
some way.  This is done with fields.  A field defines a variable and
associates it with an index space.  Note that declaring a field will add
that field to *any* instance of our specialization.  But allocation only
occurs as necessary, so you could have one grid with a field filled and
another grid where the same field is unallocated.

Registration
------------------------------------------------------------------------

In order to store data on a topology, it is necessary to *register* the
field with the topology.  Using our example specialization, we can
register variables as follows

.. code-block:: cpp

  using double_field = flecsi::field<double, flecsi::data::dense>;
  const double_field::definition<spec_t, spec_t::cells> mass_field;
  const double_field::definition<spec_t, spec_t::faces> massflux_field;

The ``double_field`` typedef provides an alias for fields where the
basic data type is ``double`` and the layout is ``flecsi::data::dense``
(see below for more detail about layouts; the ``dense`` layout means
there is one value for each index point).  The ``mass_field`` variable
declaration states that the ``spec_t`` specialization has a mass in each
cell.  The ``massflux_field`` variable declaration states that the
``spec_t`` specialization has a mass flux on each face.

Registering a field against a topology or specialization type adds that
field to the type itself, so that all instances of that topology type
have that field defined.  But the names are not explicitly added to the
type.  If we have an instance of ``spec_t`` called ``grid``, you cannot
access the mass field by calling ``grid.mass_field``.  Instead, the
field objects themselves become tools to extract the field data, so you
would get the mass field from ``grid`` by calling ``mass_field(grid)``.

If you have multiple fields on the same index space, you can declare
them in a way that's analogous to a struct of arrays or in a way that's
analogous to an array of structs.  In a hydrodynamics code, the
conserved variables in your cells would include both mass and energy, so
you can declare these fields in two different ways.  To get a
struct-of-arrays data structure, you would declare

.. code-block:: cpp

  const field<double, flecsi::data::dense>::definition<spec_t, spec_t::cells> mass_field;
  const field<double, flecsi::data::dense>::definition<spec_t, spec_t::cells> energy_field;

To get an array-of-structs data structure, you would instead declare

.. code-block:: cpp

  struct cell_data_t {
    double mass;
    double energy;
  };
  const field<cell_data_t, flecsi::data::dense>::definition<spec_t, spec_t::cells> cell_field;

The struct-of-arrays approach would allow you to allocate storage for
the mass field without also having to allocate storage for the energy
field, but in the array-of-structs approach you are allocating storage
for ``cell_data_t`` structs so you cannot allocate storage for the mass
without also allocating storage for the energy.

When you have a vector quantity, you have extra options.  In our example
hydrodynamics code, you also have a momentum variable in each cell, and
momentum is a three-dimensional vector.  To get a struct-of-arrays
approach, you could declare each component independently as

.. code-block:: cpp

  const field<double, flecsi::data::dense>::definition<spec_t, spec_t::cells> momentum0_field;
  const field<double, flecsi::data::dense>::definition<spec_t, spec_t::cells> momentum1_field;
  const field<double, flecsi::data::dense>::definition<spec_t, spec_t::cells> momentum2_field;

or you could simply declare

.. code-block:: cpp

  const field<double, flecsi::data::dense>::definition<spec_t, spec_t::cells> momentum_fields[3];

In both cases, you get a struct-of-arrays data structure and the
different momentum fields can be allocated independently of each other.
If you want an array-of-structs approach, you could declare

.. code-block:: cpp

  struct momenta_t {
    double x;
    double y;
    double z;
  };
  const field<momenta_t, flecsi::data::dense>::definition<spec_t, spec_t::cells> momentum_field;

or you could declare

.. code-block:: cpp

  const field<double[3], flecsi::data::dense>::definition<spec_t, spec_t::cells> momentum_field;

Both of these give an array-of-structs approach, where allocating
storage for one momentum requires allocating storage for all three
momenta.  An additional option would be to choose a different layout.
This is unlikely to be used for something like momentum where the number
of momenta will be the same in every cell.  But if you have something
like masses of different materials, where not every material is present
in every cell, you might choose to use a ``ragged`` layout or a
``sparse`` layout instead of a ``dense`` layout.  See :ref:`layouts` for
more information.

.. _field-accessors:

Accessors
------------------------------------------------------------------------

Operating on field data should be done through tasks (see
:doc:`programming`).  However, the task does not take the field
directly but rather an *accessor* to the field.  The use of accessors
allows tasks to have different permissions, which are used by the task
model to determine the order in which tasks may be executed.  For
example, two tasks that access the same field have to be serialized if
they both have read and write permissions for that field, but two tasks
with read-only access to the same field can run in parallel.

The available permissions are:

* ``na`` -- no access

* ``ro`` -- read-only

* ``wo`` -- write-only

* ``rw`` -- read-write

Permission settings may vary depending on the topology and
specialization.  In some cases there may only be a single permission
setting that applies to all index points within the current color.  More
complex permissions may distinguish between ghost, shared, and exclusive
elements (see :ref:`ghost-elements` for more detail) in order to allow
finer-grained parallelization strategies.

.. _field-mutators:

Mutators
------------------------------------------------------------------------

Changing data values can be done through an accessor, but some layouts
are also resizable and resizing requires the use of a *mutator*.  For
example, a field with the ``ragged`` layout is a resizable vector at
each index point.  An accessor allows access to the values in the
layout, but each vector must remain the same size.  Adding or removing
elements from the vector requires the use of a ``ragged`` mutator, which
provides an interface based on ``std::vector``.

When resizing the data structures, the ``ragged`` and ``sparse``
mutators will use buffers to temporarily hold data until its ``commit``
method is called (usually done by FleCSI at the end of a task), at which
point data will be repacked into persistent storage.  Data reallocation
is handled separately from repacking data, so resizable fields have
maximum sizes and adding too many new elements in a task causes the
``commit`` method to abort the process.

.. _layouts:

Layouts
========================================================================

Each field has a specific layout, which specifies details needed by
FleCSI for data layout and memory allocation.  We have already mentioned
some layouts when discussing fields: dense, ragged, and sparse.
Assuming that an individual data element is of type ``T``, FleCSI
provides the following layouts:

* The ``single`` layout stores one ``T`` for each color.  For example,
  if you have physics operators that only apply in certain regions of
  space, you could using the ``single`` layout to store a flag
  indicating if a given color needs to execute that physics operator.

* The ``dense`` layout stores one ``T`` for each index point.  This is a
  very common layout, used when every cell or particle has a value for
  that field, such as mass in a cell or energy of a particle.

* The ``ragged`` layout stores a resizable array of ``T`` for each index
  point.  It can be thought of as analogous to having a
  ``std::vector<T>`` for each index point.  This could be used to store
  a list of materials present in a given cell when not every cell has
  every material and the materials can move to different cells over
  time.

* The ``sparse`` layout can be thought of in several ways, depending on
  what is most useful:

  * A ``sparse`` layout can be thought of as similar to the ``ragged``
    layout, but without the requirement that every element in the array
    must exist.  That is, the fact that element 5 exists does not imply
    that elements 0 through 4 also exist.  This allows memory savings if
    there are large gaps where ``ragged`` would allocate memory that is
    not needed.

  * Alternately, a ``sparse`` layout can be thought of as analogous to a
    ``std::map<std::size_t,T>`` for each index point.  This is closer to
    the actual implementation.

  An example where this might be useful is to track the masses of
  different materials when not every cell must contain every material
  but you want the same index to always refer to the same material.

* The ``particle`` layout stores an unordered set of ``T`` for each
  color.

:ref:`field-mutators` are not needed for the ``single`` or ``dense``
layouts, but are provided for the ``ragged``, ``sparse``, and
``particle`` layouts.
