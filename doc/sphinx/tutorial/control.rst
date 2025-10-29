Control Model
*************
FleCSI uses a graph of operations called a `control model` to structure the application.
This tutorial demonstrates how to create a control model and how to extend one with additional operations.
We begin with an extremely simple control model but quickly build to a
more realistic example.

.. attention::

  This tutorial breaks our convention of avoiding details of the FleCSI
  specialization layer (We lied to you in the tutorial introduction!).
  This is necessary to demonstrate how the code for this example works.
  In general, the control model for an application will be defined by
  the specialization developer, with a fixed set of documented control
  points that are not modifiable by the application developer. Users of
  the specialization (application developers) will add actions that
  define the substance of the simulation.

----

Example 1: Simple
+++++++++++++++++

This example will show you how to construct the simplest non-trivial
control model that might make sense for an application.
:numref:`simple` shows the control model, which has three `control points`
(*initialize*, *advance*, and *finalize*) with one `action` (a C++ function to call) under each.
This is still not a very useful example because it lacks cycles
(introduced in the next example).
However, it will serve to introduce the definition of a control model
from the core FleCSI type.

.. _simple:
.. figure:: images/simple.svg
   :align: center
   :width: 45%

   Control Model for Example 1.

.. tip::

  This example introduces the notion of a policy type. Policies are C++
  types (structs or classes) that can be used to define the behavior of
  another templated C++ type. This design pattern was introduced by
  Andrei Alexandrescu in `Modern C++ Design`__. The details of the
  design pattern are not important to this example but may make
  interesting reading. For our purposes, a policy is simply a C++ struct
  that we pass to the core control type.

  __ https://en.wikipedia.org/wiki/Modern_C%2B%2B_Design

Defining Control Points
^^^^^^^^^^^^^^^^^^^^^^^

The first thing about a control model is the control points. To define
these, we use an enumeration. Consider the following from
*tutorial/2-control/1-simple.hh*:

.. literalinclude:: ../../../tutorial/2-control/1-simple.hh
  :language: cpp
  :start-at: // Enumeration defining the control point identifiers
  :end-at: enum class cp {

The name of the enumeration (``cp``) is arbitrary.
However, it is useful to make it concise because it will be used in the
application code.

In addition to the enumeration itself, we also overload the ``*`` operator:

.. literalinclude:: ../../../tutorial/2-control/1-simple.hh
  :language: cpp
  :start-at: // Define labels for the control points
  :end-before: // Control policy for this example.

Perhaps this looks complicated, but really all it does is to return a
string literal given one of the control point enumeration values defined
in ``cp``.
This approach is used by FleCSI because it is safer than defining a
static array of string literals.
The labels are used to create visualizations of the control model.

The following defines the actual control policy:

.. literalinclude:: ../../../tutorial/2-control/1-simple.hh
  :language: cpp
  :start-at: // Control policy for this example
  :end-at: // struct control_policy

The first type definition in the policy captures the control points
enumeration type.
This type is used in the control interface for declaring actions:

.. literalinclude:: ../../../tutorial/2-control/1-simple.hh
  :language: cpp
  :start-at: // Capture the control points enumeration type
  :end-at: using control_points_enum = cp

The actual control points are defined as a list of the typeified
integer-valued control points enumeration.
The templated ``point`` definition is a convenience interface for
typeifying the control points:

.. literalinclude:: ../../../tutorial/2-control/1-simple.hh
  :language: cpp
  :start-at: // The control_points list defines
  :end-at: // struct control_policy

In the above ``control_points`` list definition, the order is important,
as it is the order in which the control points will be sorted and thus
executed.

Finally, after the control policy, we define a fully-qualified control
type. This is the control type that we will use in our example application.

.. literalinclude:: ../../../tutorial/2-control/1-simple.hh
  :language: cpp
  :start-at: // Define a fully-qualified control type
  :end-at: using control = flecsi::run::control<control_policy>;

That's the entire control policy for this example.
Without comments, it is about 20 lines of code. Let's see how we use it!

Using the Control Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The source for this example is in *tutorial/2-control/1-simple.cc*.
Much of the actual main function is the same as in the previous
examples.

The last part of the main function is not really different from previous
examples; we just now use our own control type.
FleCSI therefore executes the actions registered on the control model.

.. literalinclude:: ../../../tutorial/2-control/1-simple.cc
   :language: cpp
   :start-at: // Run the control model
   :end-at: run.control

The return value is 0 or the value in a ``control_base::exception`` thrown by an action to terminate execution.

Now that we have defined the control model and added it to our runtime
setup, the only thing that remains is to add some actions under the
control points.

As stated, actions are nothing more than C/C++ functions.
For this example, there are three actions, which all have the same form.
We list only the ``initialize`` function here:

.. literalinclude:: ../../../tutorial/2-control/1-simple.cc
   :language: cpp
   :start-at: // Function definition of an initialize action
   :end-before: // Register the initialize action under 

To register an action with the control model, we declare a control
action:

.. literalinclude:: ../../../tutorial/2-control/1-simple.cc
   :language: cpp
   :start-at: // Register the initialize action under
   :end-at: control::action<initialize, cp::initialize>

The template parameters to ``control::action`` are the function pointer ``initialize`` and the control point ``cp::initialize`` (which is why it can be expedient to use a concise enumeration type name).

Running this example prints the output of each of the three functions.

.. code-block:: console

  [info all p0] initialize
  [info all p0] advance
  [info all p0] finalize

Not very interesting, but you should begin to see the simplicity of
defining and using a FleCSI control model.

----

Example 2: Cycles
+++++++++++++++++

:numref:`cycle` shows a slightly more realistic control model that
cycles over *advance* and *analyze* control points.
This example demonstrates how to add a cycle.

.. _cycle:
.. figure:: images/cycle.svg
   :align: center
   :width: 55%

   Example FleCSI Control Model with Cycle.

Starting from the previous example, we add the analyze control point:

.. literalinclude:: ../../../tutorial/2-control/2-cycle.hh
   :language: cpp
   :start-at: enum class cp { initialize, advance, analyze, finalize };
   :end-before: struct control_policy

We will use ``cp::advance`` and ``cp::analyze`` to define the cycle from the
core FleCSI cycle type:

.. literalinclude:: ../../../tutorial/2-control/2-cycle.hh
   :language: cpp
   :start-at: // A cycle type. Cycles are similar
   :end-at: cycle<cycle_control

Cycles are similar to the ``control_points`` list, with the addition of a
predicate function that controls termination of the cycle:

.. literalinclude:: ../../../tutorial/2-control/2-cycle.hh
   :language: cpp
   :start-at: // Cycle predicates are passed the policy object.
   :end-at: }

For this example, the control function simply iterates for five cycles.
In a real application, the control function could be arbitrarily
complex, e.g., invoking a reduction to compute a variable time step.

Notice that the ``cycle_control`` function (is static and) accepts the control object as a parameter.
In this case, we use it to access the ``step_`` data member that keeps
track of which simulation step we are on:

Although this example is simple, in general we can use this design
pattern to access simulation control state variables.

The last piece needed to add the cycle is the actual definition of the
``control_points`` list type:

.. literalinclude:: ../../../tutorial/2-control/2-cycle.hh
   :language: cpp
   :start-at: // The control_points list type takes
   :end-before: private:

Other than adding an action under the new analyze control point, the
main function for this example is the same.

.. important::

  The core cycle type itself can contain a mixture of typeified
  enumeration values and cycles, such that cycles can be nested. It
  seems unlikely that most HPC simulations would need more than a couple
  of levels of cycling. However, FleCSI supports arbitrary cycle depths.
  :numref:`subcycle` shows a sub-cycling control model. The details of
  the program to generate this figure are not discussed here. However,
  you can view the source in *tutorial/2-control/2-subcycle.hh*.

.. _subcycle:
.. figure:: images/subcycle.svg
   :align: center
   :width: 60%

   Example FleCSI Control Model with Sub-cycles.

Example 3: Dependencies
+++++++++++++++++++++++

The code for this example provides a more detailed demonstration of
adding dependencies between actions.
In particular, it demonstrates adding dependencies after-the-fact.
The resulting control model is shown in :numref:`dependencies`.

.. _dependencies:
.. figure:: images/dependencies.svg
   :align: center
   :width: 80%

   Example Action Dependencies.

For the most part, this example is self-explanatory.
Several actions are defined for the two control points in
*3-actions.hh*:

.. literalinclude:: ../../../tutorial/2-control/3-actions.hh
   :language: cpp
   :start-at: // Register several actions under control point one.
   :end-at: package_g_action

Additionally, several dependencies are defined in the same file:

.. literalinclude:: ../../../tutorial/2-control/3-actions.hh
   :language: cpp
   :start-at: // Add dependencies a -> b, b -> d, and a -> d, i.e.,
   :end-at: const auto dep_gf = package_g_action.add(package_f_action);

Finally, the additional dependencies from ``c`` to ``a`` and from ``d`` to ``c`` are
added in the *3-dependencies.cc* file:

.. literalinclude:: ../../../tutorial/2-control/3-dependencies.cc
   :language: cpp
   :start-at: // Add dependencies a -> c, and c -> d.
   :end-at: const auto dep_dc = package_d_action.add(package_c_action);

The point of defining the dependencies involving ``c`` in a different file
is to demonstrate that dependencies do not need to be collocated,
provided that they honor normal C++ declaration rules.


Example 4: Control State
++++++++++++++++++++++++

Actions can use state via the control policy object argument:

.. code-block:: cpp

    void my_action(my_policy &p) {p.my_method();}

Let's consider a concrete example of this. In
*tutorial/2-control/4-state.hh*, we define a control policy with several
methods and some private data:

.. literalinclude:: ../../../tutorial/2-control/4-state.hh
   :language: cpp
   :start-at: std::size_t & step() {
   :end-before: };

.. important::

    Although this example demonstrates the ability to allocate heap data
    through the control state interface, this approach must not be used
    to allocate data that will be accessed by tasks and modified during the simulation: i.e., control state data should be used only to hold global constants and/or implement the control logic of the run.
    As a special case, MPI tasks can access and modify such objects.
    The FleCSI data model provides other mechanisms for creating
    and managing state data, which are documented in the :doc:`data`
    section of this tutorial.

These interfaces are used to implement the example actions in
*tutorial/2-control/4-state.cc*. The basic structure of the example
allocates a simple array, initializes the values, modifies the values,
and frees the data. Again, the code is self-explanatory:

.. literalinclude:: ../../../tutorial/2-control/4-state.cc
   :language: cpp
   :start-after: using namespace state;
   :end-at: control::action<finalize, cp::finalize> finalize_action;

The primary take-away from this example should be that users can define
arbitrary C++ interfaces and data, given the concurrent access restrictions above.

.. vim: set tabstop=2 shiftwidth=2 expandtab fo=cqt tw=72 :
