Topologies
**********

Terminology
+++++++++++
**Topologies**
    A *topology* is a distributed data structure that manages multiple
    index spaces across multiple colors.  While the name "topology" may
    seem to indicate that this is like a mesh or a layout of particles
    in space, the FleCSI topology has more to do with the layout of data
    across memory spaces than with the physical layout of the simulation
    space.  Data can be registered to an index space of a topology, and
    then it will be distributed among colors and assigned to different
    memory spaces.

**Specializations**
    A *specialization* is a customization of a FleCSI topology to create
    an interface that allows domain experts to implement operations on
    top of the topology.  The relevant operations will depend strongly
    on the needs of the domain experts, but will often include queries
    about the physical layout of points within the simulation space.

A topology may also store structural information used to interpret its fields in terms appropriate to the *category* of topology (*e.g.*, an unstructured mesh).
Any number of instances may be created of any topology.

Index Spaces
++++++++++++
FleCSI can have an arbitrary number of index spaces, each of which
represents a different kind of entity.  The names of the index spaces
are defined by the user, so they can be named "cells", "particles",
"nodes", or any other name that is helpful to the user.

An index space has no concept of relationships between different index
points in the same index space.  For example, nothing in the definition
of the index space specifies which entities are spatially adjacent to
which other entities.  This information would be encoded in the
specialization (sometimes with support from the underlying topology).

Different index spaces also have no knowledge of how they relate to each
other.  In our example, a set of faces defines the boundary of a cell
but the index spaces alone have no ability to know that, or to map from
a cell to the list of faces that surround it.  This information would be
encoded in the specialization.

Provided Topologies
+++++++++++++++++++
Because of our choice of how to structure the grid in our example, we
can use FleCSI's ``narray`` topology, which provides tools designed
specifically to help with this kind of structured grid.  For example,
the ``narray`` topology provides tools to easily access a cell's spatial
neighbors or connect from a face to the cells that it separates.

However, some topologies provide no such information, instead presenting
a generic interface that can be used to construct a variety of physical
structures and connectivities.  Typically, the specialization is
expected to implement the detailed physical information necessary to
write domain-science operations on top of a FleCSI topology.

FleCSI defines the following topologies:

* The ``global`` topology is the most basic topology provided by FleCSI.
  It has a single index space, and it is not partitioned between colors.
  The field values of the ``global`` topology can be written to by a
  single task at a time, or can be read by every task in a parallel
  launch.  This topology will typically be used for global configuration
  data (for example, whether the current simulation is 1D, 2D, or 3D).

* The ``index`` topology is the next most basic topology provided by
  FleCSI.  It has a single index space, by default simply labeled as
  "elements".  The index topology also has a runtime-specified size that
  describes how many indices there should be.  By default, the index
  topology has a coloring that assigns a single index point to each
  color, although this can be customized by a specialization.  The
  ``index`` topology could be used to store data about relationships to
  other colors (e.g., "my left neighbor is color X") or to turn on or
  off physics packages that are only run in subsets of the domain (e.g.,
  this color needs to run hydrodynamics but there are no energy
  sources).

* The ``narray`` topology represents a structured, Cartesian-product
  mesh, with operations to efficiently access neighboring index points.
  It includes features to help build faces, edges, and vertices
  (referring to them collectively as "auxiliaries").  It also helps
  manage communication across colors through ghost cells, and boundary
  conditions through boundary cells.  The ``narray`` topology is
  commonly used as the basis for the common pattern of a simulations on
  simple, structured, static grids.  Our example of a finite-volume
  hydrodynamics code on a Cartesian-product grid would use this topology
  to avoid having to re-implement all of the details already available
  from the ``narray`` topology.

* The ``ntree`` topology is designed to support particles and to
  efficiently find neighboring particles.  It uses a hashed binary tree,
  quadtree, or octree in one, two, or three dimensions respectively.
  The ``ntree`` topology is useful for particle-based methods such as
  smoothed-particle hydrodynamics.

Colorings
+++++++++

Given a coloring (which in these two simple cases can be just an integer), topology instances can be created:

.. code-block:: cpp

  using namespace flecsi;

  int top_level() {
    topo::global::slot pair;
    topo::index::slot hydro_indices;
    pair.allocate(2);
    hydro_indices.allocate(42);
    // ...
  }

Note the different interpretations of the sizes: ``pair`` doesn't have colors and holds 2 field values, while ``hydro_indices`` has 42 colors with one field value each.

Note also that the lifetime of topology instances must be limited to the top-level action (achieved here by making the slots local variables in it).

.. vim: set tabstop=2 shiftwidth=2 expandtab fo=cqt tw=72 :
