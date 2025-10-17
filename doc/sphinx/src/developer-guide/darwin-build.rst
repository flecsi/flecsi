Building for Darwin
+++++++++++++++++++

Darwin is a testbed cluster at LANL that provides a wide variety of
node hardware configurations.
The FleCSI distribution provides a
script, ``tools/darwin.sh``, that automates downloading, building, and
installing FleCSI and all of its dependencies.  The script can be run
either from a checked-out version of the FleCSI repository, in which
case it will not re-download FleCSI, or as a standalone script, in
which case it will clone the ``flecsi`` repository and build from
there.

The former is the preferred approach.  From a back-end node, run

.. code-block:: console

  $ git clone ssh://git@re-git.lanl.gov:10022/flecsi/flecsi.git
  $ flecsi/tools/darwin.sh

The script performs the following operations:

1. Clone ``flecsi`` if the script was not run as above, from a cloned
   repository.

2. Install a version of the `Spack package manager
   <https://spack.readthedocs.io/>`_ known to work with FleCSI into
   ``$HOME/spack``.

3. Load Darwin's `environment modules
   <http://modules.sourceforge.net/>`_ for known-to-work-with-FleCSI
   versions of various tools.

4. Create and activate a ``flecsi-mpich`` Spack environment.
   Download, build, and install FleCSI's dependencies into this
   environment.  (This is by far the more time-consuming part of the
   script.  Plan on about 45 minutes.)

5. Configure, build, test, and install FleCSI into
   ``$HOME/flecsi-inst``, including documentation.

6. Configure and build the FleCSI tutorial files.  This ensures that
   it is possible to compile and link against the headers and
   libraries in ``$HOME/flecsi-inst``.

The script expects a fairly virgin environment.  It currently fails if
Spack is already installed, conflicting modules are already loaded, or
other aspects of the installation already have been run.

Once the script completes, you can activate the FleCSI environment with

.. code-block:: console

  $ source ~/spack/share/spack/setup-env.sh
  $ spack env activate flecsi-mpich

The complete ``tools/darwin.sh`` script is reproduced below.  Although
the script is intended to be run on the Darwin cluster, it should not
be too hard to adapt it to other systems or even simply use the script
as a reference for the commands needed to get FleCSI up and running.

.. literalinclude:: ../../../../tools/darwin.sh
  :language: bash
