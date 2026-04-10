#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"

KESSEL_CHECKOUT_REF="v0.2.0"
SPACK_CHECKOUT_REF="releases/v1.1"
SPACK_PACKAGES_CHECKOUT_REF="7134022c6c6221cb26062aca31f19d9fb683bd13"

# Clone flecsi into the current directory unless the script was run
# from what looks like a flecsi clone.
cd "$SCRIPT_DIR"
if [ -d ../flecsi ] ; then
    cd ..
else
    cd -
    git clone https://github.com/flecsi/flecsi.git
    cd flecsi
fi

FLECSI_DIR=$PWD
DEV_SETUP_DIR=$(dirname "$PWD")

# Clone Spack and Kessel into same parent directory as flecsi unless SPACK_ROOT
# and KESSEL_ROOT are already set
if [ -d "$SPACK_ROOT" ]; then
  echo "Found existing Spack install in $SPACK_ROOT"
else
  if [ ! -d "$DEV_SETUP_DIR/spack" ]; then
    git clone https://github.com/spack/spack.git "$DEV_SETUP_DIR/spack"
    git -C "$DEV_SETUP_DIR/spack" checkout $SPACK_CHECKOUT_REF
  fi
  source "$DEV_SETUP_DIR/spack/share/spack/setup-env.sh"
fi

if [ -d "$KESSEL_ROOT" ]; then
  echo "Found existing Kessel install in $KESSEL_ROOT"
else
  if [ ! -d "$DEV_SETUP_DIR/kessel" ]; then
    git clone https://github.com/lanl/kessel.git "$DEV_SETUP_DIR/kessel"
    git -C "$DEV_SETUP_DIR/kessel" checkout $KESSEL_CHECKOUT_REF
  fi
  source "$DEV_SETUP_DIR/kessel/share/kessel/setup-env.sh"
fi

# generate activate script to isolated Spack/Kessel install
# this ensures that bootstrap and Spack configurations don't mess with other installations
printf 'export SPACK_ROOT=%q
export KESSEL_ROOT=%q
export SPACK_USER_CACHE_PATH=%q
export SPACK_DISABLE_LOCAL_CONFIG=true
export SPACK_SKIP_MODULES=true
source "$SPACK_ROOT/share/spack/setup-env.sh"
source "$KESSEL_ROOT/share/kessel/setup-env.sh"
echo "Activating FleCSI development environment"
' "$SPACK_ROOT" "$KESSEL_ROOT" "$DEV_SETUP_DIR/.spack" > "$DEV_SETUP_DIR/activate.sh"

source "$DEV_SETUP_DIR/activate.sh" > /dev/null

spack repo update --commit $SPACK_PACKAGES_CHECKOUT_REF builtin
spack bootstrap now
spack compiler find

echo "========================================================================="
echo "Development Environment Ready"
echo
echo "To activate, run the following in your shell:"
echo
echo "source $DEV_SETUP_DIR/activate.sh"
echo "========================================================================="
