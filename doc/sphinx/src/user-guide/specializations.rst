Specializations
***************
A *specialization* is an adaptation of a core FleCSI topology to
create an interface that is suitable for developers of a family of 
domain-specific applications.
For example, in the Poisson tutorial a specialization of ``narray`` is provided to define a
two-dimensional finite-difference domain interface required by applications
with nearest-neighbor ghost dependencies.

Requirements
++++++++++++
The minimum required of a specialization is that it provide certain compile-time information needed to fully specify its topology category.
For example, the ``narray`` topology requires a dimensionality for the array, but ``unstructured`` does not because it deals not in geometry but only in generic mesh connectivity.

Index Spaces
^^^^^^^^^^^^
The index spaces that are required for a simulation depend on the numerical methods used.
For example, finite-volume hydrodynamics methods store how much
of certain conserved quantities is contained within each cell and compute fluxes of these conserved quantities across cell
boundaries.  These fluxes logically live at the boundaries between
cells, which we call faces.  Given that we need to track data both in
cells and on faces, that tells us that we need two index spaces: one for
cells and one for faces.  This would be declared as

.. code-block:: cpp

  enum index_space { cells, faces };
  using index_spaces = has<cells, faces>;

The enum is simply defining some names for convenience.  The
``using index_spaces`` declaration actually specifies the number and
ordering of index spaces.  These names chosen here are not special and
carry no meaning; they are purely for the convenience of users.

Coloring
^^^^^^^^
A specialization can define an MPI task function called ``color``, in which case an application can use ``mpi_coloring`` to launch that task to create a coloring.
The return type is the coloring type required by its topology category; for example, ``narray`` accepts options like periodicity and ghost halo depth.

Interface
^^^^^^^^^
The topology accessor used by application tasks is based on an interface defined in the specialization called ``interface``.
This class template inherits from a core topology interface and provides a set of member functions implemented in terms of ``protected`` members of that base class to present topology information to the application in a suitable form.
That suitability is in part derived from using domain-specific terminology that cannot be used in the internal interface.

Common interfaces include producing ranges of all index points or those of a particular kind (those on the boundary, those exclusive to the current color, `etc.`) as well as queries about a specific index point (its neighbors, its global id, `etc.`).

Utilities
+++++++++
Specializations are defined by inheriting from the ``topo::specialization`` class template, naming the topology category and the specialization being defined.
For convenience, this base class `provides <../../api/user/structflecsi_1_1topo_1_1specialization.html>`__ several helpful type aliases and defaults for many of the customizable options.

Additional `class and function templates <../../api/user/group__topology.html>`__ are provided to assist in writing specializations.
The class template ``topo::id`` exists to help prevent mistakes such as using a cell ID as if it were a vertex ID.
``topo::make_ids`` is a convenience function template to convert a range of ordinary integers into a range of ``id`` objects.
