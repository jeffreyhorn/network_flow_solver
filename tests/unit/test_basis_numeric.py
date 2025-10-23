import contextlib
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver import data  # noqa: E402
from network_solver.data import build_problem  # noqa: E402
from network_solver.simplex import NetworkSimplex  # noqa: E402


@contextlib.contextmanager
def _suppress_scipy(monkeypatch):
    """Temporarily make scipy imports fail so the dense fallback gets exercised."""
    # Remove existing scipy modules from sys.modules so imports are re-attempted.
    cached = {name: sys.modules.pop(name) for name in list(sys.modules) if name.startswith("scipy")}

    def _raise_import_error(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ModuleNotFoundError("scipy disabled for test")
        return original_import(name, *args, **kwargs)

    original_import = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", _raise_import_error)
    try:
        # Force reload of our helper so it sees the missing SciPy.
        import network_solver.basis_lu as basis_lu

        importlib.reload(basis_lu)
        yield
    finally:
        import network_solver.basis_lu as basis_lu

        importlib.reload(basis_lu)
        monkeypatch.setattr(importlib, "import_module", original_import)
        sys.modules.update(cached)


@pytest.mark.parametrize("sci_available", [True, False])
def test_tree_basis_numeric_representation(monkeypatch, sci_available):
    if not sci_available:
        with _suppress_scipy(monkeypatch):
            _run_basis_checks()
    else:
        _run_basis_checks()


def _run_basis_checks():
    # Validate that the tree basis builds coherent numeric factors for simple problems.
    nodes = [
        {"id": "a", "supply": 5.0},
        {"id": "b", "supply": -5.0},
    ]
    arcs = [
        {"tail": "a", "head": "b", "capacity": 10.0, "cost": 1.0},
    ]
    problem = build_problem(nodes=nodes, arcs=arcs, directed=True, tolerance=1e-6)
    # Enable dense inverse for this test since we're checking it specifically
    from network_solver.data import SolverOptions

    options = SolverOptions(use_dense_inverse=True)
    solver = NetworkSimplex(problem, options=options)
    basis = solver.basis
    assert basis.basis_matrix is not None
    assert basis.basis_inverse is not None
    assert basis.ft_engine is not None
    identity = basis.basis_matrix @ basis.basis_inverse
    assert np.allclose(identity, np.eye(identity.shape[0]))


def test_tree_basis_numeric_representation_without_ft(monkeypatch):
    # Drop the FT engine intentionally and ensure LU/dense fallbacks keep working.
    nodes = [
        data.Node(id="a", supply=5.0),
        data.Node(id="b", supply=-5.0),
        data.Node(id="c", supply=0.0),
    ]
    arcs = [
        data.Arc(tail="a", head="c", capacity=10.0, cost=1.0),
        data.Arc(tail="c", head="b", capacity=10.0, cost=1.0),
    ]
    problem = data.NetworkProblem(
        directed=True,
        nodes={node.id: node for node in nodes},
        arcs=arcs,
        tolerance=1e-6,
    )
    solver = NetworkSimplex(problem)

    original_build_numeric = solver.basis._build_numeric_basis

    def fake_build_numeric(*args, **kwargs):
        original_build_numeric(*args, **kwargs)
        solver.basis.ft_engine = None

    monkeypatch.setattr(solver.basis, "_build_numeric_basis", fake_build_numeric)
    solver.basis.rebuild(solver.tree_adj, solver.arcs)

    projection = solver.basis.project_column(solver.arcs[1])
    assert projection is not None

    # Rebuild without FT and ensure tree bookkeeping still functions.
    solver.basis.rebuild(solver.tree_adj, solver.arcs, build_numeric=False)
    assert solver.basis.ft_engine is None
    assert solver.basis.basis_matrix is not None


def test_rebuild_detects_disconnected_tree():
    # Simulate a broken in_tree flag and ensure rebuild spots the disconnection.
    problem = build_problem(
        nodes=[
            {"id": "a", "supply": 1.0},
            {"id": "b", "supply": 0.0},
            {"id": "c", "supply": -1.0},
        ],
        arcs=[
            {"tail": "a", "head": "b", "capacity": 5.0, "cost": 1.0},
            {"tail": "b", "head": "c", "capacity": 5.0, "cost": 1.0},
            {"tail": "a", "head": "c", "capacity": 5.0, "cost": 3.0},
        ],
        directed=True,
        tolerance=1e-6,
    )
    solver = NetworkSimplex(problem)
    solver.basis.rebuild(solver.tree_adj, solver.arcs)
    broken = solver.basis.tree_arc_indices[0]
    solver.arcs[broken].in_tree = False
    with pytest.raises(RuntimeError, match="disconnected spanning tree"):
        solver.basis.rebuild(solver.tree_adj, solver.arcs)


def test_project_column_falls_back_when_inverse_unavailable(monkeypatch):
    # Force np.linalg.inv to fail so LU fallback handles the projection.
    problem = build_problem(
        nodes=[
            {"id": "a", "supply": 1.0},
            {"id": "b", "supply": -1.0},
        ],
        arcs=[
            {"tail": "a", "head": "b", "capacity": 5.0, "cost": 1.0},
        ],
        directed=True,
        tolerance=1e-6,
    )
    solver = NetworkSimplex(problem)

    basis = solver.basis
    original_inv = np.linalg.inv
    monkeypatch.setattr(np.linalg, "inv", lambda _: (_ for _ in ()).throw(np.linalg.LinAlgError))
    basis.rebuild(solver.tree_adj, solver.arcs)
    monkeypatch.setattr(np.linalg, "inv", original_inv)

    assert basis.basis_inverse is None
    column = basis.project_column(solver.arcs[0])
    assert column is not None


def test_project_column_uses_lu_when_ft_and_inverse_missing(monkeypatch):
    # If both FT solve and inverse are unavailable, LU factors should still answer queries.
    problem = build_problem(
        nodes=[
            {"id": "a", "supply": 1.0},
            {"id": "b", "supply": -1.0},
        ],
        arcs=[
            {"tail": "a", "head": "b", "capacity": 5.0, "cost": 1.0},
        ],
        directed=True,
        tolerance=1e-6,
    )
    solver = NetworkSimplex(problem)

    original_matrix = solver.basis.basis_matrix

    from network_solver.core.forrest_tomlin import ForrestTomlin

    class AlwaysNoneFT(ForrestTomlin):
        def solve(self, rhs):
            return None

    solver.basis.ft_engine = AlwaysNoneFT(original_matrix)
    solver.basis.basis_inverse = None

    column = solver.basis.project_column(solver.arcs[0])
    assert column is not None


def test_replace_arc_updates_mapping_for_identical_column():
    # When a column replacement is effectively a no-op, mappings should remain stable.
    problem = build_problem(
        nodes=[
            {"id": "a", "supply": 1.0},
            {"id": "b", "supply": 0.0},
            {"id": "c", "supply": -1.0},
        ],
        arcs=[
            {"tail": "a", "head": "b", "capacity": 5.0, "cost": 1.0},
            {"tail": "b", "head": "c", "capacity": 5.0, "cost": 1.0},
            {"tail": "a", "head": "c", "capacity": 5.0, "cost": 3.0},
        ],
        directed=True,
        tolerance=1e-6,
    )
    solver = NetworkSimplex(problem)
    solver._run_simplex_iterations(5, allow_zero=True, phase_one=True)
    basis = solver.basis
    leaving_idx = basis.tree_arc_indices[0]
    before_arc_to_pos = basis.arc_to_pos.copy()
    before_tree = list(basis.tree_arc_indices)
    assert basis.replace_arc(leaving_idx, leaving_idx, solver.arcs, solver.tolerance)
    assert basis.arc_to_pos == before_arc_to_pos
    assert basis.tree_arc_indices == before_tree


def test_replace_arc_applies_sherman_morrison_update():
    # Distinct incoming columns should update the basis mappings accordingly.
    # This test specifically checks Sherman-Morrison updates, so enable dense inverse.
    problem = build_problem(
        nodes=[
            {"id": "a", "supply": 0.0},
            {"id": "b", "supply": 0.0},
            {"id": "c", "supply": 0.0},
            {"id": "d", "supply": 0.0},
        ],
        arcs=[
            {"tail": "a", "head": "b", "capacity": 5.0, "cost": 1.0},
            {"tail": "b", "head": "c", "capacity": 5.0, "cost": 1.0},
            {"tail": "c", "head": "d", "capacity": 5.0, "cost": 1.0},
            {"tail": "a", "head": "c", "capacity": 5.0, "cost": 2.0},
        ],
        directed=True,
        tolerance=1e-6,
    )
    from network_solver.data import SolverOptions

    options = SolverOptions(use_dense_inverse=True)
    solver = NetworkSimplex(problem, options=options)
    basis = solver.basis
    entering_idx = next(
        idx for idx, arc in enumerate(solver.arcs[: solver.actual_arc_count]) if not arc.in_tree
    )
    leaving_idx = basis.tree_arc_indices[0]
    assert basis.replace_arc(leaving_idx, entering_idx, solver.arcs, solver.tolerance)
    assert basis.arc_to_pos.get(entering_idx) == 0
    assert basis.tree_arc_indices[0] == entering_idx


def test_replace_arc_rebuilds_lu_when_only_sparse_available(monkeypatch):
    # Force the code path that reconstructs the matrix from sparse factors.
    problem = build_problem(
        nodes=[
            {"id": "a", "supply": 0.0},
            {"id": "b", "supply": 0.0},
            {"id": "c", "supply": 0.0},
        ],
        arcs=[
            {"tail": "a", "head": "b", "capacity": 5.0, "cost": 1.0},
            {"tail": "b", "head": "c", "capacity": 5.0, "cost": 1.0},
            {"tail": "c", "head": "a", "capacity": 5.0, "cost": 1.0},
        ],
        directed=True,
        tolerance=1e-6,
    )
    solver = NetworkSimplex(problem)
    basis = solver.basis
    basis.ft_engine = None
    basis.basis_inverse = None
    entering_idx = next(
        idx for idx, arc in enumerate(solver.arcs[: solver.actual_arc_count]) if not arc.in_tree
    )
    leaving_idx = basis.tree_arc_indices[0]
    original_column_vector = basis._column_vector

    def fake_column_vector(arc):
        if arc is solver.arcs[entering_idx]:
            return np.array([-1.0, 1.0, 0.0])
        return original_column_vector(arc)

    monkeypatch.setattr(basis, "_column_vector", fake_column_vector)

    import network_solver.basis as basis_module

    calls = {"reconstruct": False, "build": False}

    def fake_reconstruct(factors):
        calls["reconstruct"] = True
        return np.array(factors.dense_matrix, copy=True)

    class DummyLU:
        def __init__(self, matrix):
            self.dense_matrix = np.array(matrix, copy=True)
            self.sparse_matrix = None
            self.lu = object()

    def fake_build(matrix):
        calls["build"] = True
        return DummyLU(matrix)

    monkeypatch.setattr(basis_module, "reconstruct_matrix", fake_reconstruct)
    monkeypatch.setattr(basis_module, "build_lu", fake_build)

    assert basis.replace_arc(leaving_idx, entering_idx, solver.arcs, solver.tolerance)
    assert calls["reconstruct"] and calls["build"]


def test_replace_arc_handles_forrest_tomlin_failures(monkeypatch):
    # If FT raises or returns False, replace_arc should surface the failure gracefully.
    # Ensure clean module state (in case previous tests left monkeypatches)
    import importlib
    import network_solver.basis
    import network_solver.basis_lu

    importlib.reload(network_solver.basis_lu)
    importlib.reload(network_solver.basis)

    problem = build_problem(
        nodes=[
            {"id": "a", "supply": 1.0},
            {"id": "b", "supply": 0.0},
            {"id": "c", "supply": -1.0},
        ],
        arcs=[
            {"tail": "a", "head": "b", "capacity": 5.0, "cost": 1.0},
            {"tail": "b", "head": "c", "capacity": 5.0, "cost": 1.0},
            {"tail": "a", "head": "c", "capacity": 5.0, "cost": 3.0},
        ],
        directed=True,
        tolerance=1e-6,
    )
    solver = NetworkSimplex(problem)
    basis = solver.basis

    # Verify basis is properly initialized with LU factors
    assert basis.lu_factors is not None, "LU factors should be available for fallback"
    entering_idx = next(
        idx for idx, arc in enumerate(solver.arcs[: solver.actual_arc_count]) if not arc.in_tree
    )
    leaving_idx = basis.tree_arc_indices[0]

    class FalseFT:
        def update(self, *_args, **_kwargs):
            return False

    basis.ft_engine = FalseFT()
    pos = basis.arc_to_pos[leaving_idx]
    before_col = basis.basis_matrix[:, pos].copy()
    assert basis.replace_arc(leaving_idx, entering_idx, solver.arcs, solver.tolerance) is True
    assert not np.allclose(basis.basis_matrix[:, pos], before_col)

    class RaisingFT:
        def update(self, *_args, **_kwargs):
            raise ValueError("boom")

    basis.ft_engine = RaisingFT()
    basis.basis_inverse = None
    basis.lu_factors = None
    assert basis.replace_arc(leaving_idx, entering_idx, solver.arcs, solver.tolerance) is False
