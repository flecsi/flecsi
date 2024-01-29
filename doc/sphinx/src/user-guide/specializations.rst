Specializations
***************
A *specialization* is an adaptation of one or more core FleCSI topologies to
create an interface that is suitable for developers of a family of 
domain-specific applications.

For example, in the Poisson tutorial a specialization is provided to define a
two-dimensional finite-difference domain interface required by applications
with nearest-neighbor ghost dependencies.

Hydrodynamics Example
+++++++++++++++++++++
While we have chosen a topology for our example, FleCSI-provided
topologies are written for broad use-cases and need to be adapted to the
needs of the particular problem being solved.  This adaptation is done
by way of a specialization.  The specialization is code written by the
user to customize the FleCSI-provided topology for use in the user's
particular application.  It also brings together information about index
spaces and colors into a single type.  For the ``narray`` topology
specifically, a specialization will define things like

* How many dimensions is this simulation?  We said three dimensions.

* How many index spaces and what are they called?  We said two: cells
  and faces.

* How is the grid decomposed into colors?  The ``narray`` topology
  provides tools to ease this process, but the specialization still
  needs to specify certain information by writing a ``color`` method.
  This information would include questions such as:

  * Is the grid periodic?

  * How many ghost cells are used for communication between colors?

* What operations will be available for this grid?  This is defined by
  the ``interface`` struct defined in the specialization. A common way
  to design the ``interface`` struct is to write your scientific code
  until you realize, "I have a question and the topology has the
  information to answer it."  Then you add one or more methods to the
  ``interface`` struct to answer that question.  These methods may be
  things such as:

  * Given the local index of my cell (within the color), what is the
    global index (across the entire domain)?

  * Is this cell on the boundary of the grid, or is it an interior cell?

  * I want a C++ range I can iterate over containing all faces in this
    color.

  * I want a range containing all cells in this color that are not
    copied into neighboring ghost cells, so that I can update those
    cells without thinking about ghost copies yet.

Index Spaces
^^^^^^^^^^^^
For our example, we first partition our simulation domain into volumes,
which we call cells.  Finite-volume hydrodynamics methods store how much
of certain conserved quantities is contained within each cell.  In order
to update how much of each quantity is in each cell, a finite-volume
method computes fluxes of these conserved quantities across cell
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

Utilities
+++++++++
`Several class and function templates <../../api/user/group__topology.html>`__ are provided to assist in writing specializations.
The class template ``topo::id`` exists to help prevent mistakes such as using a cell ID as if it were a vertex ID.
``topo::make_ids`` is a convenience function template to convert a range of ordinary integers into a range of ``id`` objects.
