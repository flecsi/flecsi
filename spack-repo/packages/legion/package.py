from spack.package import *
# Spack's import hook doesn't support "from spack.pkg.builtin import legion":
from spack.pkg.builtin.legion import Legion

class Legion(Legion):
    """
    Additional named versions for Legion.
    """
    version("cr-16", commit="45afa8e658ae06cb19d8f0374de699b7fe4a197c")
