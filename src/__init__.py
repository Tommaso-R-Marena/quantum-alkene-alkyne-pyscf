"""src — Core helpers for the quantum-alkene-alkyne-pyscf project.

Submodules are exposed with guarded relative imports so that missing
optional dependencies (PennyLane, OpenFermion) do not break the
package-level ``from src import *`` used in notebooks.
"""

from __future__ import annotations

__version__ = "0.3.0"

__all__: list[str] = []

try:
    from . import molecule_builder  # noqa: F401
    __all__.append("molecule_builder")
except Exception:  # pragma: no cover
    pass

try:
    from . import hamiltonian_utils  # noqa: F401
    __all__.append("hamiltonian_utils")
except Exception:  # pragma: no cover
    pass

try:
    from . import vqe_runner  # noqa: F401
    __all__.append("vqe_runner")
except Exception:  # pragma: no cover
    pass

try:
    from . import analysis  # noqa: F401
    __all__.append("analysis")
except Exception:  # pragma: no cover
    pass
