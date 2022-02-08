#! /bin/bash

######################################
# Build FleCSI entirely from scratch #
# (tested on LANL's Darwin cluster)  #
######################################

# Trace commands and abort on the first error.
set -e
set -v

# Check if we're running from within a flecsi clone.  If not, clone
# flecsi into the current directory.  In either case, cd to the
# top-level flecsi directory.
script_dir=$(dirname $(readlink -f "$0"))
cd "$script_dir"
repo_name=$(basename -s .git $(git config --get remote.origin.url || echo not-flecsi))
if [ "$repo_name" == flecsi ] ; then
    cd $(git rev-parse --show-toplevel)
else
    cd -
    git clone git@gitlab.lanl.gov:flecsi/flecsi.git
    cd flecsi
    git checkout origin/2.1
fi
git rev-parse HEAD

# Create a build subdirectory and cd to it.
test -d build && rm -rf build
mkdir build
cd build

# Define an installation directory.
FLECSI_INSTALL="$HOME/flecsi-inst"

# Download a version of Spack known to work with FleCSI and activate it.
pushd "$HOME"
git clone https://github.com/spack/spack.git
cd spack
#git checkout v0.17.1
#git checkout origin/releases/v0.17
git switch -c origin/v0.17.2
git rev-parse HEAD
set +v
source "$HOME/spack/share/spack/setup-env.sh"
set -v
popd

# Load a newer CMake and versions of GCC and MPICH known to work with FleCSI.
# Expose these -- and whatever else happense to be sitting around -- as Spack
# externals.
module load cmake
module load gcc/9.4.0
spack external find
spack config remove packages:python   # Provides only a partial Sphinx
spack config remove packages:libtool  # Seems incomplete

# Install mpich/3.4.2 instead of relying on system build (no mpirun/mpiexec)
spack install mpich@3.4.2%gcc@9.4.0+hydra+romio~verbs device=ch4
spack load mpich

# Install FleCSI's dependencies with Spack.  The various Sphinx packages lead
# to a mess of dependencies that confuses Spack.  We temporarily specify
# "concretization: together" to get those installed then revert to the default
# of "concretization: separately".
spack env create flecsi-mpich
spack env activate flecsi-mpich
echo "  concretization: together" >> "$HOME/spack/var/spack/environments/flecsi-mpich/spack.yaml"
spack repo add ../spack-repo
spack add py-sphinx py-sphinx-rtd-theme py-recommonmark
spack install
sed -i -e 's/concretization: together/concretization: separately/' "$HOME/spack/var/spack/environments/flecsi-mpich/spack.yaml"
spack install graphviz +poppler
spack install --only dependencies flecsi%gcc@9.4.0 backend=legion +hdf5 +kokkos +flog ^mpich ^legion network=gasnet conduit=mpi build_type=Debug
spack load py-sphinx py-sphinx-rtd-theme py-recommonmark

# Build, test, and install FleCSI.
../tools/configure gnu legion -DCMAKE_INSTALL_PREFIX="$FLECSI_INSTALL"
make VERBOSE=1 -j
make test
make doc
make install
cd ..

# Ensure the tutorial examples build properly.
cd tutorial
test -d build && rm -rf build
mkdir build
cd build
cmake -DFleCSI_DIR="$FLECSI_INSTALL/lib64/cmake/FleCSI" ..
make
cd ../..
