"""Additional tests to improve basis.py coverage to >90%."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_solver.data import Arc, NetworkProblem, Node, SolverOptions, build_problem
from network_solver.simplex import NetworkSimplex


class TestTreeBasisInitialization:
    """Test TreeBasis initialization paths."""

    def test_auto_detect_dense_inverse_when_sparse_unavailable(self, monkeypatch):
        """Test auto-detection falls back to dense inverse when scipy unavailable."""
        # Mock has_sparse_lu to return False
        import network_solver.basis_lu as basis_lu_module

        monkeypatch.setattr(basis_lu_module, "has_sparse_lu", lambda: False)

        # Create a TreeBasis with use_dense_inverse=None (auto-detect)
        from network_solver.basis import TreeBasis

        basis = TreeBasis(
            node_count=3, root=0, tolerance=1e-6, use_dense_inverse=None, use_jit=True
        )

        # Should have auto-detected and set to True (dense)
        assert basis.use_dense_inverse is True

    def test_auto_detect_sparse_inverse_when_available(self, monkeypatch):
        """Test auto-detection uses sparse when scipy available."""
        import network_solver.basis_lu as basis_lu_module

        monkeypatch.setattr(basis_lu_module, "has_sparse_lu", lambda: True)

        from network_solver.basis import TreeBasis

        basis = TreeBasis(
            node_count=3, root=0, tolerance=1e-6, use_dense_inverse=None, use_jit=True
        )

        # Should have auto-detected and set to False (sparse)
        assert basis.use_dense_inverse is False


class TestCollectCycle:
    """Test collect_cycle edge cases."""

    def test_collect_cycle_when_tail_equals_head(self):
        """Test collect_cycle returns empty list when tail==head."""
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

        # Call collect_cycle with tail==head (same node)
        cycle_arcs = solver.basis.collect_cycle(solver.tree_adj, solver.arcs, tail=0, head=0)

        # Should return empty list when tail==head (line 118 in basis.py)
        assert cycle_arcs == []


class TestRebuildEdgeCases:
    """Test rebuild() edge cases."""

    def test_build_numeric_basis_early_returns_on_wrong_tree_arc_count(self):
        """Test _build_numeric_basis early returns when tree arc count is wrong."""
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

        # Save original state
        original_arcs = [
            (arc.tail, arc.head, arc.in_tree) for arc in solver.arcs[: solver.actual_arc_count]
        ]

        # Mark ALL arcs (including artificial) as NOT in tree
        # This creates wrong count = 0, expected = node_count - 1
        for arc in solver.arcs:
            arc.in_tree = False

        # Clear existing basis state to ensure fresh build
        basis.basis_matrix = None
        basis.basis_inverse = None
        basis.lu_factors = None
        basis.ft_engine = None

        # Call _build_numeric_basis directly - should early-return (lines 162-166)
        basis._build_numeric_basis(solver.arcs)

        # basis_matrix should still be None because tree arc count was wrong
        assert basis.basis_matrix is None
        assert basis.basis_inverse is None
        assert basis.lu_factors is None
        assert basis.ft_engine is None

        # Restore original arc states
        for idx, (tail, head, in_tree) in enumerate(original_arcs):
            solver.arcs[idx].in_tree = in_tree


class TestColumnVector:
    """Test _column_vector with root node involvement."""

    def test_column_vector_with_root_involvement(self):
        """Test _column_vector when arc involves root node."""
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

        # The solver adds artificial arcs from a special root node
        # Find an artificial arc that involves the actual root node
        root_arc = None
        for arc in solver.arcs:
            if arc.artificial and (arc.tail == basis.root or arc.head == basis.root):
                root_arc = arc
                break

        if root_arc is not None:
            # Test column vector generation for arc involving root
            vec = basis._column_vector(root_arc)

            # Build expected vector manually
            expected = np.zeros((basis.node_count - 1,))
            if root_arc.tail != basis.root:
                expected[basis.node_to_row[root_arc.tail]] = 1.0
            if root_arc.head != basis.root:
                expected[basis.node_to_row[root_arc.head]] = -1.0

            assert np.allclose(vec, expected)

    def test_column_vector_normal_arc(self):
        """Test _column_vector for normal arc not involving root."""
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

        # Get a regular arc (not involving root)
        arc = solver.arcs[0]

        # Verify it doesn't involve root
        assert arc.tail != basis.root
        assert arc.head != basis.root

        # Get column vector
        vec = basis._column_vector(arc)

        # Should have +1 at tail row, -1 at head row
        expected = np.zeros((basis.node_count - 1,))
        expected[basis.node_to_row[arc.tail]] = 1.0
        expected[basis.node_to_row[arc.head]] = -1.0

        assert np.allclose(vec, expected)


class TestProjectionCache:
    """Test projection cache functionality."""

    def test_projection_cache_stores_results(self):
        """Test that successful projections are stored in cache."""
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
        # Enable projection cache
        options = SolverOptions(projection_cache_size=100)
        solver = NetworkSimplex(problem, options=options)

        basis = solver.basis
        arc = solver.arcs[0]

        # Cache should be empty initially
        assert len(basis.projection_cache) == 0

        # Project column
        result = basis.project_column(arc)
        assert result is not None

        # Cache should now contain the result
        assert arc.key in basis.projection_cache
        assert np.allclose(basis.projection_cache[arc.key], result)

    def test_projection_cache_disabled_when_size_zero(self):
        """Test that cache is not used when size is 0."""
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
        # Disable projection cache
        options = SolverOptions(projection_cache_size=0)
        solver = NetworkSimplex(problem, options=options)

        basis = solver.basis
        arc = solver.arcs[0]

        # Project column
        result = basis.project_column(arc)
        assert result is not None

        # Cache should remain empty
        assert len(basis.projection_cache) == 0


class TestConditionNumberEstimation:
    """Test condition number estimation edge cases."""

    def test_condition_number_with_ft_engine_only(self, monkeypatch):
        """Test condition number estimation using FT engine when no dense inverse."""
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

        # Remove dense inverse, keep FT engine and LU factors
        basis.basis_inverse = None

        # This should use FT engine solve path for condition number estimation
        cond_num = basis.estimate_condition_number()
        assert cond_num is not None
        assert cond_num > 0

    def test_condition_number_with_lu_factors_only(self):
        """Test condition number estimation using only LU factors."""
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

        # Remove both dense inverse and FT engine, keep only LU factors
        basis.basis_inverse = None
        basis.ft_engine = None

        # This should use LU solve path for condition number estimation
        cond_num = basis.estimate_condition_number()
        assert cond_num is not None
        assert cond_num > 0

    def test_condition_number_returns_none_when_no_solvers(self):
        """Test condition number returns None when no inverse/LU/FT available."""
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

        # Remove all solving mechanisms
        basis.basis_inverse = None
        basis.ft_engine = None
        basis.lu_factors = None

        # Should return None
        cond_num = basis.estimate_condition_number()
        assert cond_num is None


class TestReplaceArcEdgeCases:
    """Test replace_arc edge cases."""

    def test_replace_arc_with_ft_engine_success(self):
        """Test replace_arc successfully uses FT engine update."""
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
        # Use dense inverse mode to trigger specific path
        options = SolverOptions(use_dense_inverse=True)
        solver = NetworkSimplex(problem, options=options)
        basis = solver.basis

        # Find entering arc (not in tree)
        entering_idx = next(
            idx for idx, arc in enumerate(solver.arcs[: solver.actual_arc_count]) if not arc.in_tree
        )
        leaving_idx = basis.tree_arc_indices[0]

        # Ensure FT engine is available
        assert basis.ft_engine is not None

        # Replace arc - should use FT update path
        success = basis.replace_arc(leaving_idx, entering_idx, solver.arcs, solver.tolerance)
        assert success is True

        # When using FT in dense mode, basis_inverse should be cleared
        assert basis.basis_inverse is None

    def test_replace_arc_returns_false_when_no_basis_matrix(self):
        """Test replace_arc returns False when basis_matrix is None."""
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

        # Clear basis matrix
        basis.basis_matrix = None

        # Should return False immediately
        result = basis.replace_arc(0, 0, solver.arcs, solver.tolerance)
        assert result is False

    def test_replace_arc_returns_false_when_arc_to_pos_is_none(self):
        """Test replace_arc returns False when arc_to_pos is None."""
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

        # Clear arc_to_pos
        basis.arc_to_pos = None

        # Should return False immediately
        result = basis.replace_arc(0, 0, solver.arcs, solver.tolerance)
        assert result is False
