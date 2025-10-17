import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver.data import Arc, NetworkProblem, Node  # noqa: E402
from network_solver.simplex import NetworkSimplex  # noqa: E402


def test_unbounded_cycle_raises_runtime_error():
    """A negative-cost cycle with infinite capacity should be detected as unbounded."""
    problem = NetworkProblem(
        directed=True,
        nodes={
            "A": Node(id="A", supply=0.0),
            "B": Node(id="B", supply=0.0),
        },
        arcs=[
            Arc(tail="A", head="B", capacity=None, cost=-5.0),
            Arc(tail="B", head="A", capacity=None, cost=1.0),
        ],
        tolerance=1e-9,
    )

    solver = NetworkSimplex(problem)

    with pytest.raises(RuntimeError, match="Unbounded problem detected"):
        solver.solve()


def test_infeasible_demand_reports_infeasible_status():
    """A balanced but capacity-starved network should report infeasibility."""
    problem = NetworkProblem(
        directed=True,
        nodes={
            "s": Node(id="s", supply=5.0),
            "m": Node(id="m", supply=0.0),
            "t": Node(id="t", supply=-5.0),
        },
        arcs=[
            Arc(tail="s", head="m", capacity=5.0, cost=1.0),
        ],
        tolerance=1e-9,
    )

    solver = NetworkSimplex(problem)
    result = solver.solve(max_iterations=1000)

    assert result.status == "infeasible"
    assert result.objective == 0.0
    assert result.flows == {}
