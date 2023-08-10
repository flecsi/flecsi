from spack.package import *
from spack.pkg.builtin.flecsi import Flecsi

class Flecsi(Flecsi):
    # since we compile with -Wextra -Werror for development
    # and https://gitlab.kitware.com/cmake/cmake/-/issues/23141
    depends_on('cmake@3.19:3.21,3.22.3:')

    depends_on("legion@cr-16:cr-99", when="backend=legion")
