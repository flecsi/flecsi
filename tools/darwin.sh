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

# Set GCC version we're going to use
GCC_VERSION=11.1.0

# Download a version of Spack known to work with FleCSI and activate it.
pushd "$HOME"
if [ ! -d spack ]; then
  git clone https://github.com/spack/spack.git
  cd spack
  git switch releases/v0.18
  git rev-parse HEAD
else
  cd spack
  echo "Found existing Spack install in ~/spack"
  git fetch origin
  if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/releases/v0.18)" ]; then
    echo "ERROR: The current checkout does not match origin/releases/v0.18!"
    echo
    echo "Please update manually with:"
    echo " git -C ~/spack switch releases/v0.18"
    echo " git -C ~/spack pull"
    echo
    echo "WARNING: This may invalidate other Spack environments that rely on" \
         "this Spack instance!"
    exit 1
  fi
fi
set +v
source "$HOME/spack/share/spack/setup-env.sh"
set -v
popd


# Create new Spack environment and activate it
spack env remove -y flecsi-mpich || true
spack env create flecsi-mpich
spack env activate flecsi-mpich

# Load GCC version known to work with FleCSI
# and make it visible to Spack
set +v
module load gcc/${GCC_VERSION}
spack compiler find
set -v

# ignore configuration in ~/.spack to avoid conflicts
export SPACK_DISABLE_LOCAL_CONFIG=true

# allow spack to dynamically concretize packages together instead of separately.
# this avoids some installation errors due to multiple python packages depending
# on different versions of a dependency
spack config add concretizer:unify:when_possible
spack config add concretizer:reuse:false
spack config add packages:all:compiler:["gcc@${GCC_VERSION}"]

# add FleCSI spack package repository
spack repo add ../spack-repo

# add documentation dependencies
spack add py-sphinx py-sphinx-rtd-theme py-recommonmark graphviz +poppler

# On Darwin we have a Spack upstream that already has prebuilt dependencies
DARWIN_SPACK_UPSTREAM=/projects/flecsi-devel/gitlab/spack-upstream/current

if [ -d "$DARWIN_SPACK_UPSTREAM" ] && [ -x "${DARWIN_SPACK_UPSTREAM}" ]; then
  # add spack upstream if accessible
  spack config add upstreams:default:install_tree:${DARWIN_SPACK_UPSTREAM}/opt/spack/
  cp ${DARWIN_SPACK_UPSTREAM}/etc/spack/packages.yaml $HOME/spack/etc/spack/
else
  # Otherwise, load a compatible cmake and expose whatever else happens to be
  # sitting around as Spack externals
  module load cmake/3.19.2
  spack external find
fi

# install packages added so far
spack install -j $(nproc)

# Install FleCSI's dependencies with Spack.
# This also builds a version of MPICH in Spack since the ones on Darwin do not include
# an mpiexec/mpirun that works.
spack install -j $(nproc) --only dependencies flecsi%gcc@${GCC_VERSION} backend=legion +hdf5 +kokkos +flog ^mpich@3.4.2%gcc@${GCC_VERSION}+hydra+romio~verbs device=ch4 ^legion network=gasnet conduit=mpi build_type=Debug

# Build, test, and install FleCSI.
../tools/configure gnu legion -DCMAKE_INSTALL_PREFIX="$FLECSI_INSTALL"
make VERBOSE=1 -j $(nproc)
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

# Build complete
set +v
echo "BUILD COMPLETE"
echo
echo "To continue working in the same Spack environment, execute the following commands:"
echo
echo " source \$HOME/spack/share/spack/setup-env.sh"
echo " spack env activate flecsi-mpich"
echo " module load gcc/${GCC_VERSION}"
echo
