from spack.package import *
# Spack's import hook doesn't support "from spack.pkg.builtin import legion":
from spack.pkg.builtin.legion import Legion

class Legion(Legion):
    """
    Additional named versions for Legion.
    """
    version('cr-15', commit='435183796d7c8b6ac1035a6f7af480ded750f67d')
