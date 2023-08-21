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
SPACK_VERSION=develop-82d41a7be

if [[ "${SPACK_VERSION}" =~ "develop-" ]]; then
  SPACK_COMMIT=$(echo $SPACK_VERSION | cut -d- -f2)
fi

pushd "$HOME"
if [ ! -d spack ]; then
  git clone https://github.com/spack/spack.git
  cd spack
  if [[ "${SPACK_VERSION}" =~ "develop-" ]]; then
    git checkout $SPACK_COMMIT
  else
    git switch releases/$SPACK_VERSION
  fi
  git rev-parse HEAD
else
  cd spack
  echo "Found existing Spack install in ~/spack"
  git fetch origin

  if [[ "${SPACK_VERSION}" =~ "develop-" ]]; then
    if [ "$(_TMP=$(git rev-parse HEAD); echo ${_TMP:0:9})" != "${SPACK_COMMIT}" ]; then
      echo "ERROR: The current checkout does not match ${SPACK_COMMIT}!"
      echo
      echo "Please update manually with:"
      echo " git -C ~/spack checkout ${SPACK_COMMIT}"
      echo
      echo "WARNING: This may invalidate other Spack environments that rely on" \
           "this Spack instance!"
      exit 1
    fi
  else
    if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/releases/$SPACK_VERSION)" ]; then
      echo "ERROR: The current checkout does not match origin/releases/$SPACK_VERSION!"
      echo
      echo "Please update manually with:"
      echo " git -C ~/spack fetch +releases/$SPACK_VERSION:refs/remotes/origin/releases/$SPACK_VERSION"
      echo " git -C ~/spack switch origin/releases/$SPACK_VERSION"
      echo
      echo "WARNING: This may invalidate other Spack environments that rely on" \
           "this Spack instance!"
      exit 1
    fi
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
spack config add concretizer:reuse:false
spack config add packages:all:compiler:["gcc@${GCC_VERSION}"]

# add FleCSI spack package repository
spack repo add ../spack-repo

# On Darwin we have a Spack upstream that already has prebuilt dependencies
DARWIN_SPACK_UPSTREAM=/projects/flecsi-devel/gitlab/spack-upstream/$SPACK_VERSION

if [ -d "$DARWIN_SPACK_UPSTREAM" ] && [ -x "${DARWIN_SPACK_UPSTREAM}" ]; then
  # add spack upstream if accessible
  spack config add upstreams:default:install_tree:${DARWIN_SPACK_UPSTREAM}/opt/spack/
  cp ${DARWIN_SPACK_UPSTREAM}/etc/spack/{compilers.yaml,packages.yaml} $HOME/spack/etc/spack/
else
  # Otherwise, load a compatible cmake and expose whatever else happens to be
  # sitting around as Spack externals
  module load cmake/3.19.2
  spack external find
fi

# Install FleCSI's dependencies with Spack.
# This also builds a version of MPICH in Spack since the ones on Darwin do not include
# an mpiexec/mpirun that works.
spack add flecsi%gcc@${GCC_VERSION} backend=legion +doc +hdf5 +kokkos +flog \
          ^mpich%gcc@${GCC_VERSION}+hydra device=ch4 netmod=ucx \
          ^legion network=gasnet conduit=mpi build_type=Debug
spack install -j $(nproc) --only dependencies

# Build, test, and install FleCSI.

# configure FleCSI, inheriting (-i) the its configuration from the Spack environment
test -d build && rm -rf build
../tools/spack_cmake -i flecsi -- -DCMAKE_INSTALL_PREFIX="$FLECSI_INSTALL"
cmake --build build -j $(nproc)
ctest
cmake --build build --target install
cd ..

# Ensure the tutorial examples build properly.

# This makes the installed FleCSI visible to CMake's find_package
export CMAKE_PREFIX_PATH=$FLECSI_INSTALL:$CMAKE_PREFIX_PATH
cd tutorial
test -d build && rm -rf build

# configure tutorial examples with the same compiler and compiler flags as flecsi
../tools/spack_cmake -c -i flecsi
cmake --build build -j $(nproc)
cd ..

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
