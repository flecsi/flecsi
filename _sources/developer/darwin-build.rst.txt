Building for Darwin
+++++++++++++++++++

Darwin is a testbed cluster at LANL that provides a wide variety of
node hardware configurations.
Our CI pipelines run on a range of hardware of this cluster and as such it is important to target the same kind of nodes during development.

Recreating continuous integration (CI) builds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To avoid having to build a wide range of dependencies for the various hardware architectures, we are using a shared Spack deployment on Darwin.
Both the CI and developers can make use of this deployment to speed up creating their development environments.

In its simplest form, individual continuous integration jobs can be recreated with a FleCSI checkout on Darwin.
Each CI job will print a highlighted message at the beginning of the log and provide instructions on how to re-create the exact configuration.

These instructions include
- the `salloc` line for allocating the correct type of compute node to run the job
- a few steps to initiate the build and test workflow

For simplification of these instructions, we make use a LANL tool called ``kessel <https://github.com/lanl/kessel>``_.
``kessel`` is a utility for streamlining and generalizing developer and CI workflows for both cluster and local development.
It helps you set up a complete software stack for development and run predefined actions with it.

The CI instructions include sourcing ``.gitlab/kessel.sh`` to create a temporary deployment copy and activate it.
This creates a self contained, writable instance which points to a cluster-wide read-only Spack deployment as upstream.
To create a persistent copy of such a deployment, set ``KESSEL_WORKFLOW_DEPLOYMENT`` to a custom location prior to sourcing that script.

.. code-block:: console

   $ export KESSEL_WORKFLOW_DEPLOYMENT=<custom-path>
   $ source .gitlab/kessel.sh

Building on a local system
^^^^^^^^^^^^^^^^^^^^^^^^^^

Whether you want to build on a local system or don't want to use the prebuilt Spack deployments on the Darwin cluster, the FleCSI distribution also includes a utility script ``dev-setup.sh`` that helps with the initial setup of a clean build environment.

.. code-block:: console

  $ git clone https://github.com/flecsi/flecsi.git
  $ flecsi/tools/dev-setup.sh

This script will download both Spack and Kessel and place them in the parent directory of your FleCSI checkout.
In addition, it places an ``activate.sh`` in this folder to set the necessary environment variables to make both Kessel and Spack available in your current shell.

Simply source the ``activate.sh`` script after the script has completed.

With both Spack and Kessel in place, you can then go ahead and run Kessel workflows.
