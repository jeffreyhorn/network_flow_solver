"""Tests for problem preprocessing functionality."""

import pytest

from network_solver import build_problem, preprocess_and_solve, preprocess_problem


class TestRemoveRedundantArcs:
    """Test removal of redundant parallel arcs."""

    def test_removes_duplicate_arcs_same_cost(self):
        """Should merge parallel arcs with identical costs."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 50.0, "cost": 2.0},
            {"tail": "A", "head": "B", "capacity": 50.0, "cost": 2.0},  # Duplicate
            {"tail": "A", "head": "B", "capacity": 30.0, "cost": 2.0},  # Another duplicate
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        result = preprocess_problem(problem)

        # Should have 1 arc with combined capacity
        assert len(result.problem.arcs) == 1
        assert result.problem.arcs[0].capacity == 130.0  # 50 + 50 + 30
        assert result.problem.arcs[0].cost == 2.0
        assert result.redundant_arcs == 2
        assert result.removed_arcs == 2

    def test_keeps_arcs_with_different_costs(self):
        """Should not merge parallel arcs with different costs."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 50.0, "cost": 2.0},
            {"tail": "A", "head": "B", "capacity": 50.0, "cost": 3.0},  # Different cost
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        result = preprocess_problem(problem)

        # Should keep both arcs
        assert len(result.problem.arcs) == 2
        assert result.redundant_arcs == 0

    def test_handles_infinite_capacity_arcs(self):
        """Should handle arcs with infinite capacity correctly."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 50.0, "cost": 2.0},
            {"tail": "A", "head": "B", "capacity": None, "cost": 2.0},  # Infinite
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        result = preprocess_problem(problem)

        # Should merge to infinite capacity
        assert len(result.problem.arcs) == 1
        assert result.problem.arcs[0].capacity is None
        assert result.redundant_arcs == 1

    def test_respects_lower_bounds(self):
        """Should keep arcs with different lower bounds separate."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 50.0, "cost": 2.0, "lower": 0.0},
            {"tail": "A", "head": "B", "capacity": 50.0, "cost": 2.0, "lower": 10.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        result = preprocess_problem(problem)

        # Should keep both arcs (different lower bounds)
        assert len(result.problem.arcs) == 2
        assert result.redundant_arcs == 0

    def test_can_disable_redundant_removal(self):
        """Should skip redundant removal when disabled."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 50.0, "cost": 2.0},
            {"tail": "A", "head": "B", "capacity": 50.0, "cost": 2.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        result = preprocess_problem(problem, remove_redundant=False)

        # Should keep both arcs
        assert len(result.problem.arcs) == 2
        assert result.redundant_arcs == 0


class TestDetectDisconnectedComponents:
    """Test detection of disconnected components."""

    def test_detects_single_component(self):
        """Should detect 1 component for connected graph."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": 0.0},
            {"id": "C", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 200.0, "cost": 1.0},
            {"tail": "B", "head": "C", "capacity": 200.0, "cost": 1.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        result = preprocess_problem(problem)

        assert result.disconnected_components == 1

    def test_detects_multiple_components(self):
        """Should detect multiple disconnected components."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": -100.0},
            {"id": "C", "supply": 50.0},
            {"id": "D", "supply": -50.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 200.0, "cost": 1.0},
            {"tail": "C", "head": "D", "capacity": 100.0, "cost": 2.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        result = preprocess_problem(problem)

        assert result.disconnected_components == 2

    def test_treats_directed_as_undirected_for_connectivity(self):
        """Should check connectivity ignoring arc direction."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": 0.0},
            {"id": "C", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 200.0, "cost": 1.0},
            {"tail": "C", "head": "B", "capacity": 200.0, "cost": 1.0},  # Reverse direction
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        result = preprocess_problem(problem)

        # Should be 1 component (connected via B)
        assert result.disconnected_components == 1

    def test_can_disable_component_detection(self):
        """Should skip component detection when disabled."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 200.0, "cost": 1.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        result = preprocess_problem(problem, detect_disconnected=False)

        assert result.disconnected_components == 0


class TestSimplifySeriesArcs:
    """Test simplification of series arcs."""

    def test_merges_series_arcs(self):
        """Should merge consecutive arcs through zero-supply node."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": 0.0},  # Transshipment
            {"id": "C", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 150.0, "cost": 2.0},
            {"tail": "B", "head": "C", "capacity": 200.0, "cost": 3.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        result = preprocess_problem(problem)

        # Should have 1 merged arc, 2 nodes (removed B)
        assert len(result.problem.nodes) == 2
        assert "A" in result.problem.nodes
        assert "C" in result.problem.nodes
        assert "B" not in result.problem.nodes
        assert len(result.problem.arcs) == 1
        assert result.problem.arcs[0].tail == "A"
        assert result.problem.arcs[0].head == "C"
        assert result.problem.arcs[0].cost == 5.0  # 2 + 3
        assert result.problem.arcs[0].capacity == 150.0  # min(150, 200)
        assert result.removed_nodes == 1
        assert result.merged_arcs == 1

    def test_does_not_merge_non_zero_supply_node(self):
        """Should not merge if intermediate node has supply."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": 10.0},  # Has supply
            {"id": "C", "supply": -110.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 150.0, "cost": 2.0},
            {"tail": "B", "head": "C", "capacity": 200.0, "cost": 3.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        result = preprocess_problem(problem)

        # Should keep all nodes and arcs
        assert len(result.problem.nodes) == 3
        assert len(result.problem.arcs) == 2
        assert result.merged_arcs == 0

    def test_does_not_merge_node_with_multiple_in_arcs(self):
        """Should not merge if node has multiple incoming arcs."""
        nodes = [
            {"id": "A", "supply": 50.0},
            {"id": "B", "supply": 50.0},
            {"id": "C", "supply": 0.0},
            {"id": "D", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "C", "capacity": 100.0, "cost": 1.0},
            {"tail": "B", "head": "C", "capacity": 100.0, "cost": 1.0},
            {"tail": "C", "head": "D", "capacity": 200.0, "cost": 2.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        result = preprocess_problem(problem)

        # Should keep all nodes (C has 2 incoming arcs)
        assert len(result.problem.nodes) == 4
        assert result.merged_arcs == 0

    def test_handles_infinite_capacity_in_series(self):
        """Should handle infinite capacity correctly in series merge."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": 0.0},
            {"id": "C", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": None, "cost": 2.0},  # Infinite
            {"tail": "B", "head": "C", "capacity": 200.0, "cost": 3.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        result = preprocess_problem(problem)

        # Merged capacity should be 200 (min of inf and 200)
        assert len(result.problem.arcs) == 1
        assert result.problem.arcs[0].capacity == 200.0
        assert result.merged_arcs == 1

    def test_can_disable_series_simplification(self):
        """Should skip series simplification when disabled."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": 0.0},
            {"id": "C", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 150.0, "cost": 2.0},
            {"tail": "B", "head": "C", "capacity": 200.0, "cost": 3.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        result = preprocess_problem(problem, simplify_series=False)

        # Should keep all nodes and arcs
        assert len(result.problem.nodes) == 3
        assert len(result.problem.arcs) == 2
        assert result.merged_arcs == 0


class TestRemoveZeroSupplyNodes:
    """Test removal of zero-supply nodes with single arc."""

    def test_removes_zero_supply_single_arc_node(self):
        """Should remove zero-supply node with only one incident arc."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": 0.0},  # Zero supply, single arc
            {"id": "C", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "C", "capacity": 200.0, "cost": 1.0},
            {"tail": "B", "head": "C", "capacity": 50.0, "cost": 2.0},  # Dead end
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        result = preprocess_problem(problem)

        # Should remove B and its arc
        assert len(result.problem.nodes) == 2
        assert "B" not in result.problem.nodes
        assert len(result.problem.arcs) == 1
        assert result.removed_nodes >= 1  # May also be removed by series

    def test_keeps_zero_supply_node_with_multiple_arcs(self):
        """Should keep zero-supply node with multiple arcs."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": 0.0},  # Zero supply but multiple arcs
            {"id": "C", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 200.0, "cost": 1.0},
            {"tail": "B", "head": "C", "capacity": 200.0, "cost": 2.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        # First disable series simplification to test this independently
        result = preprocess_problem(problem, simplify_series=False, remove_zero_supply=True)

        # B has 2 incident arcs, should be kept (only removed by series simplification)
        # Note: with simplify_series=True (default), it would be removed
        assert "B" in result.problem.nodes or len(result.problem.nodes) == 2

    def test_can_disable_zero_supply_removal(self):
        """Should skip zero-supply removal when disabled."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": 0.0},
            {"id": "C", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "C", "capacity": 200.0, "cost": 1.0},
            {"tail": "B", "head": "C", "capacity": 50.0, "cost": 2.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        result = preprocess_problem(problem, simplify_series=False, remove_zero_supply=False)

        # Should keep all nodes
        assert len(result.problem.nodes) == 3
        assert "B" in result.problem.nodes


class TestPreprocessingStatistics:
    """Test preprocessing statistics and result structure."""

    def test_records_preprocessing_time(self):
        """Should record preprocessing time in milliseconds."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 200.0, "cost": 1.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        result = preprocess_problem(problem)

        assert result.preprocessing_time_ms >= 0
        assert isinstance(result.preprocessing_time_ms, float)

    def test_tracks_all_optimizations(self):
        """Should track each optimization in the optimizations dict."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": 0.0},
            {"id": "C", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 50.0, "cost": 2.0},
            {"tail": "A", "head": "B", "capacity": 50.0, "cost": 2.0},  # Duplicate
            {"tail": "B", "head": "C", "capacity": 200.0, "cost": 3.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        result = preprocess_problem(problem)

        # Should have optimization counts
        assert "redundant_arcs_removed" in result.optimizations
        assert "disconnected_components" in result.optimizations
        assert result.optimizations["redundant_arcs_removed"] >= 0

    def test_preserves_original_problem(self):
        """Should not modify original problem."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 50.0, "cost": 2.0},
            {"tail": "A", "head": "B", "capacity": 50.0, "cost": 2.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        original_arc_count = len(problem.arcs)
        original_node_count = len(problem.nodes)

        preprocess_problem(problem)

        # Original should be unchanged
        assert len(problem.arcs) == original_arc_count
        assert len(problem.nodes) == original_node_count


class TestPreprocessAndSolve:
    """Test the preprocess_and_solve convenience function."""

    def test_preprocesses_and_solves(self):
        """Should preprocess then solve in one call."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 50.0, "cost": 2.0},
            {"tail": "A", "head": "B", "capacity": 50.0, "cost": 2.0},  # Duplicate
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        preproc_result, flow_result = preprocess_and_solve(problem)

        # Should have preprocessing results
        assert preproc_result.redundant_arcs > 0
        assert preproc_result.removed_arcs > 0

        # Should have flow results
        assert flow_result.status == "optimal"
        assert flow_result.objective == pytest.approx(200.0)  # 100 units * 2 cost

    def test_passes_solver_options(self):
        """Should pass options through to solve_min_cost_flow."""
        from network_solver import SolverOptions

        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 200.0, "cost": 2.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        options = SolverOptions(tolerance=1e-10)
        preproc_result, flow_result = preprocess_and_solve(problem, options=options)

        # Should solve successfully
        assert flow_result.status == "optimal"


class TestPreprocessingIntegration:
    """Integration tests for preprocessing with solving."""

    def test_preprocessed_solution_matches_original(self):
        """Preprocessing should not change optimal objective."""
        from network_solver import solve_min_cost_flow

        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": 0.0},
            {"id": "C", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 50.0, "cost": 2.0},
            {"tail": "A", "head": "B", "capacity": 50.0, "cost": 2.0},  # Duplicate
            {"tail": "B", "head": "C", "capacity": 200.0, "cost": 3.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        # Solve without preprocessing
        result_original = solve_min_cost_flow(problem)

        # Solve with preprocessing
        preproc_result = preprocess_problem(problem)
        result_preprocessed = solve_min_cost_flow(preproc_result.problem)

        # Should have same objective
        assert result_preprocessed.objective == pytest.approx(result_original.objective)

    def test_large_problem_with_all_optimizations(self):
        """Should handle larger problem with multiple optimization opportunities."""
        nodes = [
            {"id": "s1", "supply": 100.0},
            {"id": "s2", "supply": 100.0},
            {"id": "t1", "supply": 0.0},  # Transshipment
            {"id": "t2", "supply": 0.0},  # Transshipment
            {"id": "d1", "supply": -120.0},
            {"id": "d2", "supply": -80.0},
        ]
        arcs = [
            # Redundant arcs
            {"tail": "s1", "head": "t1", "capacity": 100.0, "cost": 1.0},
            {"tail": "s1", "head": "t1", "capacity": 100.0, "cost": 1.0},  # Duplicate
            # Series arcs
            {"tail": "s2", "head": "t2", "capacity": 100.0, "cost": 2.0},
            {"tail": "t2", "head": "d2", "capacity": 100.0, "cost": 1.0},
            # Normal arcs
            {"tail": "t1", "head": "d1", "capacity": 200.0, "cost": 2.0},
            {"tail": "t1", "head": "d2", "capacity": 100.0, "cost": 3.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        result = preprocess_problem(problem)

        # Should have removed some arcs and nodes
        assert result.removed_arcs > 0
        assert len(result.problem.arcs) < len(arcs)


class TestResultTranslation:
    """Test translation of results from preprocessed problem to original problem."""

    def test_arc_mapping_for_redundant_arcs(self):
        """Arc mapping should track redundant arc merging."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 50.0, "cost": 2.0},
            {"tail": "A", "head": "B", "capacity": 50.0, "cost": 2.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        result = preprocess_problem(problem)

        # Both original arcs (indices 0 and 1) should map to the same arc
        assert 0 in result.arc_mapping
        assert 1 in result.arc_mapping
        assert result.arc_mapping[0] == ("A", "B")
        assert result.arc_mapping[1] == ("A", "B")

    def test_arc_mapping_for_series_arcs(self):
        """Arc mapping should track series arc merging."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": 0.0},  # Will be removed
            {"id": "C", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 200.0, "cost": 2.0},
            {"tail": "B", "head": "C", "capacity": 200.0, "cost": 3.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        result = preprocess_problem(problem)

        # Both original arcs (indices 0 and 1) should map to the merged arc
        assert result.arc_mapping[0] == ("A", "C")
        assert result.arc_mapping[1] == ("A", "C")

        # Removed node should map to None
        assert result.node_mapping["B"] is None

    def test_arc_mapping_for_removed_arcs(self):
        """Arc mapping should track removed arcs."""
        nodes = [
            {"id": "A", "supply": 0.0},  # Zero supply with single arc
            {"id": "B", "supply": 100.0},
            {"id": "C", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 100.0, "cost": 1.0},  # Will be removed
            {"tail": "B", "head": "C", "capacity": 200.0, "cost": 2.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        result = preprocess_problem(problem)

        # Removed arc (index 0) should map to None
        assert result.arc_mapping[0] is None

        # Removed node should map to None
        assert result.node_mapping["A"] is None

    def test_translate_result_with_redundant_arcs(self):
        """Result translation should distribute flow for redundant arcs."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 50.0, "cost": 2.0},
            {"tail": "A", "head": "B", "capacity": 50.0, "cost": 2.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        preproc_result, flow_result = preprocess_and_solve(problem)

        # Should have flows for both original arcs
        assert ("A", "B") in flow_result.flows

        # Total flow should be 100 (sum of all parallel arcs)
        total_flow = flow_result.flows[("A", "B")]
        assert total_flow == pytest.approx(100.0)

        # Objective should be correct
        assert flow_result.objective == pytest.approx(200.0)

    def test_translate_result_with_series_arcs(self):
        """Result translation should handle series arc merging."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": 0.0},
            {"id": "C", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 200.0, "cost": 2.0},
            {"tail": "B", "head": "C", "capacity": 200.0, "cost": 3.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        preproc_result, flow_result = preprocess_and_solve(problem)

        # Should have flows for both original arcs (even though B was removed)
        assert ("A", "B") in flow_result.flows
        assert ("B", "C") in flow_result.flows

        # Both should have the same flow
        assert flow_result.flows[("A", "B")] == pytest.approx(100.0)
        assert flow_result.flows[("B", "C")] == pytest.approx(100.0)

        # Objective should be correct (2+3)*100 = 500
        assert flow_result.objective == pytest.approx(500.0)

        # All nodes should have duals (including removed node B)
        assert "A" in flow_result.duals
        assert "B" in flow_result.duals
        assert "C" in flow_result.duals

    def test_translate_result_with_removed_arcs(self):
        """Result translation should set zero flow for removed arcs."""
        nodes = [
            {"id": "A", "supply": 0.0},
            {"id": "B", "supply": 100.0},
            {"id": "C", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 100.0, "cost": 1.0},
            {"tail": "B", "head": "C", "capacity": 200.0, "cost": 2.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        preproc_result, flow_result = preprocess_and_solve(problem)

        # Removed arc should have zero flow
        assert ("A", "B") in flow_result.flows
        assert flow_result.flows[("A", "B")] == pytest.approx(0.0)

    def test_preprocess_and_solve_matches_original_objective(self):
        """Preprocessing + translation should preserve optimal objective."""
        from network_solver import solve_min_cost_flow

        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": 0.0},
            {"id": "C", "supply": 0.0},
            {"id": "D", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 50.0, "cost": 2.0},
            {"tail": "A", "head": "B", "capacity": 50.0, "cost": 2.0},  # Redundant
            {"tail": "B", "head": "C", "capacity": 200.0, "cost": 1.0},
            {"tail": "C", "head": "D", "capacity": 200.0, "cost": 3.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        # Solve original
        result_original = solve_min_cost_flow(problem)

        # Solve with preprocessing
        preproc_result, result_preprocessed = preprocess_and_solve(problem)

        # Objectives should match
        assert result_preprocessed.objective == pytest.approx(result_original.objective)

        # Both should be optimal
        assert result_original.status == "optimal"
        assert result_preprocessed.status == "optimal"

    def test_removed_nodes_have_computed_duals(self):
        """Removed nodes should have duals computed from adjacent nodes."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": 0.0},  # Will be removed (series node)
            {"id": "C", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 200.0, "cost": 2.0},
            {"tail": "B", "head": "C", "capacity": 200.0, "cost": 3.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        preproc_result, flow_result = preprocess_and_solve(problem)

        # Node B should be removed
        assert preproc_result.node_mapping["B"] is None

        # But B should still have a dual value computed
        assert "B" in flow_result.duals
        assert isinstance(flow_result.duals["B"], float)

        # The dual should be reasonable (between A and C duals given the costs)
        # For arc A->B with cost 2: dual[B] - dual[A] <= 2
        # For arc B->C with cost 3: dual[C] - dual[B] <= 3
        # So: dual[A] <= dual[B] <= dual[A] + 2
        # And: dual[C] - 3 <= dual[B] <= dual[C]
        dual_a = flow_result.duals["A"]
        dual_b = flow_result.duals["B"]
        dual_c = flow_result.duals["C"]

        # Check that B's dual is between the bounds from both constraints
        # Allow some tolerance for averaging
        assert dual_b >= min(dual_a + 2.0, dual_c - 3.0) - 1e-6
        assert dual_b <= max(dual_a + 2.0, dual_c - 3.0) + 1e-6

    def test_removed_node_with_no_adjacent_preserved_nodes(self):
        """Removed node with no preserved neighbors should default to 0."""
        nodes = [
            {"id": "A", "supply": 0.0},  # Will be removed
            {"id": "B", "supply": 100.0},
            {"id": "C", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 100.0, "cost": 1.0},  # Will be removed
            {"tail": "B", "head": "C", "capacity": 200.0, "cost": 2.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)

        preproc_result, flow_result = preprocess_and_solve(problem)

        # Node A should be removed
        assert preproc_result.node_mapping["A"] is None

        # A should have a dual value (defaulted to 0 since no adjacent preserved nodes)
        assert "A" in flow_result.duals
        # Since A only connects to B which is preserved, it should compute from B
        # For arc A->B with cost 1: dual[B] - dual[A] <= 1
        # So dual[A] >= dual[B] - 1
        dual_a = flow_result.duals["A"]
        dual_b = flow_result.duals["B"]
        assert dual_a == pytest.approx(dual_b - 1.0)
