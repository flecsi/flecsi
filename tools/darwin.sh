#! /bin/bash

######################################
# Build FleCSI entirely from scratch #
# (tested on LANL's Darwin cluster)  #
######################################

# Trace commands and abort on the first error.
set -e

# Clone flecsi into the current directory unless the script was run
# from what looks like a flecsi clone.
script_dir=$(dirname $(readlink -f "$0"))
cd "$script_dir"
if [ -d ../flecsi ] ; then
    cd ..
else
    cd -
    git clone ssh://git@re-git.lanl.gov:10022/flecsi/flecsi.git
    cd flecsi
fi

# Log where we came from in Git in case this is needed for troubleshooting.
(
    set +e
    git config --get remote.origin.url 2>/dev/null || \
      echo "${0}: Directory $(pwd) does not appear to have come from Git" 1>&2
    git rev-parse HEAD 2>/dev/null
)

# Define an installation directory.
FLECSI_INSTALL="$HOME/flecsi-inst"

# Set GCC version we're going to use
GCC_VERSION=$(echo $(grep GCC_VERSION: .gitlab-ci.yml | cut -d: -f2))

# Download a version of Spack known to work with FleCSI and activate it.
SPACK_VERSION=$(echo $(grep USE_SPACK_UPSTREAM: .gitlab-ci.yml | cut -d: -f2))

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
    if [ "$(_TMP=$(git rev-parse HEAD); echo ${_TMP:0:8})" != "${SPACK_COMMIT}" ]; then
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
source "$HOME/spack/share/spack/setup-env.sh"
popd


# Create new Spack environment and activate it
spack env remove -y flecsi-mpich || true
spack env create flecsi-mpich
spack env activate flecsi-mpich

# Load GCC version known to work with FleCSI
# and make it visible to Spack
module load gcc/${GCC_VERSION}

# ignore configuration in ~/.spack to avoid conflicts
export SPACK_DISABLE_LOCAL_CONFIG=true

# allow spack to dynamically concretize packages together instead of separately.
# this avoids some installation errors due to multiple python packages depending
# on different versions of a dependency
spack config add concretizer:reuse:false
spack config add packages:all:compiler:["gcc@${GCC_VERSION}"]

# add FleCSI spack package repository
spack repo add spack-repo

# On Darwin we have a Spack upstream that already has prebuilt dependencies
DARWIN_SPACK_UPSTREAM=/projects/flecsi-devel/gitlab/spack-upstream/$SPACK_VERSION

if [ -d "$DARWIN_SPACK_UPSTREAM" ] && [ -x "${DARWIN_SPACK_UPSTREAM}" ]; then
  # add spack upstream if accessible
  spack config add upstreams:default:install_tree:${DARWIN_SPACK_UPSTREAM}/opt/spack/
  cp ${DARWIN_SPACK_UPSTREAM}/etc/spack/{compilers.yaml,packages.yaml} $HOME/spack/etc/spack/
else
  # Otherwise, load a compatible cmake and expose whatever else happens to be
  # sitting around as Spack externals
  module load cmake/3.26.3
  spack compiler find
  spack external find
fi

# Install FleCSI's dependencies with Spack.
spack add flecsi%gcc@${GCC_VERSION} backend=legion +doc +graphviz +hdf5 +kokkos +flog \
          ^legion network=gasnet conduit=mpi build_type=Debug \
          ^mpich
spack install -j $(nproc) --only dependencies

# Build, test, and install FleCSI.

# configure FleCSI, inheriting (-i) the its configuration from the Spack environment
test -d build && rm -rf build
tools/spack_cmake -i -t flecsi -- -DCMAKE_INSTALL_PREFIX="$FLECSI_INSTALL"
cd build
cmake --build . -j $(nproc)
ctest
cmake --build . --target install
cd ..

# Ensure the tutorial examples build properly.

# This makes the installed FleCSI visible to CMake's find_package
export CMAKE_PREFIX_PATH=$FLECSI_INSTALL:$CMAKE_PREFIX_PATH
cd tutorial
test -d build && rm -rf build

# configure tutorial examples with the same compiler and compiler flags as flecsi
../tools/spack_cmake -c -i flecsi
cmake --build build -j $(nproc)

# Build complete
echo "BUILD COMPLETE"
echo
echo "To continue working in the same Spack environment, execute the following commands:"
echo
echo " source \$HOME/spack/share/spack/setup-env.sh"
echo " spack env activate flecsi-mpich"
echo " module load gcc/${GCC_VERSION}"
echo
