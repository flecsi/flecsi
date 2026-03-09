# Helper script that detects a deployment on a cluster.
# Intended to work on sh, zsh, and bash.
DEPLOYMENT_VERSION="2025-10-21"
SCRIPT_PATH=${BASH_SOURCE[0]:-${(%):-%x}}
PARENT_DIR=$( cd "$( dirname "${SCRIPT_PATH}" )" &>/dev/null && pwd )

export FLECSI_CHECKOUT=$(realpath $PARENT_DIR/..)

if command -v sacctmgr >/dev/null 2>&1 && command -v jq >/dev/null 2>&1; then
  SYSTEM_NAME=$(sacctmgr list --json clusters  | jq -r '.clusters[0].name')
fi

# During a regular CI run the workflow deployment is always a temporary copy of
# the cluster deployment to allow customization. This can be overwritten by
# setting KESSEL_WORKFLOW_DEPLOYMENT to another persistent location. If the
# folder set in KESSEL_WORKFLOW_DEPLOYMENT doesn't exist, a copy of the cluster
# deployment will be made. If KESSEL_WORKFLOW_DEPLOYMENT set to "upstream", the cluster
# deployment is used, but typically remains read-only.
_KESSEL_WORKFLOW_DEPLOYMENT="$KESSEL_WORKFLOW_DEPLOYMENT"
export KESSEL_WORKFLOW_DEPLOYMENT=${KESSEL_WORKFLOW_DEPLOYMENT:-${TMPDIR:-/tmp}/$USER-ci-envs}

if [ "$SYSTEM_NAME" = "darwin" ]; then
  export KESSEL_DEPLOYMENT=${KESSEL_DEPLOYMENT:-/usr/projects/tpp/flecsi/deployments/$DEPLOYMENT_VERSION}
else
  echo "ERROR: Unknown system!" >&2
  return 1
fi

if [ "$KESSEL_WORKFLOW_DEPLOYMENT" = "upstream" ] && [ -d "$KESSEL_DEPLOYMENT" ]; then
  source "$KESSEL_DEPLOYMENT/activate.sh"
else
  if [ -d "$KESSEL_DEPLOYMENT" ] && [ ! -d "$KESSEL_WORKFLOW_DEPLOYMENT" ]; then
    source "$KESSEL_DEPLOYMENT/activate.sh"
    clone-deployment "$KESSEL_WORKFLOW_DEPLOYMENT"
  fi
  if [ ! -d "$KESSEL_WORKFLOW_DEPLOYMENT" ]; then
    echo "ERROR: $KESSEL_WORKFLOW_DEPLOYMENT does not exist!" >&2
    return 1
  elif [ -z "$_KESSEL_WORKFLOW_DEPLOYMENT" ] && [ ! -O "$KESSEL_WORKFLOW_DEPLOYMENT" ]; then
    echo "ERROR: $KESSEL_WORKFLOW_DEPLOYMENT not owned by $USER!" >&2
    return 1
  else
    source "$KESSEL_WORKFLOW_DEPLOYMENT/activate.sh"
  fi
fi

unset _KESSEL_WORKFLOW_DEPLOYMENT
