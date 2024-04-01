from spack.package import *
from spack.pkg.builtin.flecsi import Flecsi
import pathlib

with open(pathlib.Path(__file__).parent / "../../../.version") as f:
  dot_version = f.read().strip()

flecsi_selector = {"preferred": True}

if dot_version == "develop":
    flecsi_selector["branch"] = "develop"
    flecsi_version = "develop"
elif dot_version.startswith("v"):
    flecsi_selector["tag"] = dot_version[1:]
    flecsi_version = dot_version[1:]
else:
    flecsi_selector["branch"] = dot_version[1:]
    flecsi_version = dot_version[1:] + ".develop"

class Flecsi(Flecsi):
    depends_on('cmake@3.23:')
    version(flecsi_version, **flecsi_selector)
