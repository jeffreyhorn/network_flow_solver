"""Solver adapters for comparison framework.

This module provides adapters for different network flow solvers, allowing
fair comparison of performance and solution quality. Solvers are optional
dependencies - the framework gracefully handles missing solvers.

Available adapters:
- NetworkSolverAdapter: Our network simplex implementation (always available)
- NetworkXAdapter: NetworkX min_cost_flow (usually available)
- ORToolsAdapter: Google OR-Tools (optional: pip install ortools)
- PuLPAdapter: PuLP with network flow formulation (optional: pip install pulp)
- LEMONAdapter: LEMON C++ library (optional: requires manual installation)

Usage:
    from benchmarks.solvers import get_available_solvers, NetworkSolverAdapter

    solvers = get_available_solvers()
    for solver_class in solvers:
        print(f"Available: {solver_class.name}")
"""

from .base import SolverAdapter, SolverResult
from .network_solver_adapter import NetworkSolverAdapter
from .networkx_adapter import NetworkXAdapter

# Try to import optional solvers
_AVAILABLE_SOLVERS = [NetworkSolverAdapter, NetworkXAdapter]

try:
    from .ortools_adapter import ORToolsAdapter

    _AVAILABLE_SOLVERS.append(ORToolsAdapter)
except ImportError:
    pass

try:
    from .pulp_adapter import PuLPAdapter

    _AVAILABLE_SOLVERS.append(PuLPAdapter)
except ImportError:
    pass

# LEMON requires manual C++ compilation - not easily installable
# Uncomment if you have LEMON Python bindings:
# try:
#     from .lemon_adapter import LEMONAdapter
#     if LEMONAdapter.is_available():
#         _AVAILABLE_SOLVERS.append(LEMONAdapter)
# except ImportError:
#     pass


def get_available_solvers() -> list[type[SolverAdapter]]:
    """Return list of available solver adapter classes.

    Returns:
        List of solver adapter classes that can be instantiated.
        Always includes NetworkSolverAdapter. Other solvers included
        only if their dependencies are installed.
    """
    return _AVAILABLE_SOLVERS


def get_solver_names() -> list[str]:
    """Return list of available solver names."""
    return [solver.name for solver in _AVAILABLE_SOLVERS]


__all__ = [
    "SolverAdapter",
    "SolverResult",
    "NetworkSolverAdapter",
    "NetworkXAdapter",
    "get_available_solvers",
    "get_solver_names",
]
