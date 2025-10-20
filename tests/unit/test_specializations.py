"""Tests for network specialization detection."""

from network_solver import NetworkType, analyze_network_structure, build_problem


class TestTransportationDetection:
    """Test detection of transportation problems."""

    def test_detect_simple_transportation_problem(self):
        """Test detection of basic transportation problem."""
        # 2 sources, 3 sinks - classic transportation
        problem = build_problem(
            nodes=[
                {"id": "s1", "supply": 50.0},
                {"id": "s2", "supply": 30.0},
                {"id": "t1", "supply": -40.0},
                {"id": "t2", "supply": -20.0},
                {"id": "t3", "supply": -20.0},
            ],
            arcs=[
                {"tail": "s1", "head": "t1", "capacity": 100.0, "cost": 5.0},
                {"tail": "s1", "head": "t2", "capacity": 100.0, "cost": 3.0},
                {"tail": "s1", "head": "t3", "capacity": 100.0, "cost": 4.0},
                {"tail": "s2", "head": "t1", "capacity": 100.0, "cost": 2.0},
                {"tail": "s2", "head": "t2", "capacity": 100.0, "cost": 6.0},
                {"tail": "s2", "head": "t3", "capacity": 100.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        structure = analyze_network_structure(problem)

        assert structure.network_type == NetworkType.TRANSPORTATION
        assert structure.is_bipartite
        assert len(structure.source_nodes) == 2
        assert len(structure.sink_nodes) == 3
        assert len(structure.transshipment_nodes) == 0
        assert structure.is_balanced
        assert not structure.has_lower_bounds

    def test_transportation_with_transshipment_not_detected(self):
        """Transportation with transshipment nodes should not be detected as transportation.

        With single source/sink and uniform costs, this is detected as MAX_FLOW,
        which is appropriate since max flow problems can have transshipment nodes.
        """
        problem = build_problem(
            nodes=[
                {"id": "s1", "supply": 50.0},
                {"id": "mid", "supply": 0.0},  # Transshipment
                {"id": "t1", "supply": -50.0},
            ],
            arcs=[
                {"tail": "s1", "head": "mid", "capacity": 100.0, "cost": 1.0},
                {"tail": "mid", "head": "t1", "capacity": 100.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        structure = analyze_network_structure(problem)

        # Has transshipment node, so not pure transportation
        # With single source/sink and uniform costs, it's a max flow problem
        assert structure.network_type == NetworkType.MAX_FLOW
        assert len(structure.transshipment_nodes) == 1

    def test_transportation_with_lower_bounds_not_detected(self):
        """Transportation with lower bounds should be general."""
        problem = build_problem(
            nodes=[
                {"id": "s1", "supply": 50.0},
                {"id": "t1", "supply": -50.0},
            ],
            arcs=[
                {
                    "tail": "s1",
                    "head": "t1",
                    "capacity": 100.0,
                    "cost": 1.0,
                    "lower": 10.0,  # Lower bound
                },
            ],
            directed=True,
            tolerance=1e-6,
        )

        structure = analyze_network_structure(problem)

        assert structure.network_type == NetworkType.GENERAL
        assert structure.has_lower_bounds


class TestAssignmentDetection:
    """Test detection of assignment problems."""

    def test_detect_balanced_assignment_problem(self):
        """Test detection of balanced assignment (n workers, n jobs)."""
        # 3 workers, 3 jobs, all unit supply/demand
        problem = build_problem(
            nodes=[
                {"id": "w1", "supply": 1.0},
                {"id": "w2", "supply": 1.0},
                {"id": "w3", "supply": 1.0},
                {"id": "j1", "supply": -1.0},
                {"id": "j2", "supply": -1.0},
                {"id": "j3", "supply": -1.0},
            ],
            arcs=[
                {"tail": "w1", "head": "j1", "capacity": 1.0, "cost": 10.0},
                {"tail": "w1", "head": "j2", "capacity": 1.0, "cost": 15.0},
                {"tail": "w1", "head": "j3", "capacity": 1.0, "cost": 20.0},
                {"tail": "w2", "head": "j1", "capacity": 1.0, "cost": 12.0},
                {"tail": "w2", "head": "j2", "capacity": 1.0, "cost": 9.0},
                {"tail": "w2", "head": "j3", "capacity": 1.0, "cost": 15.0},
                {"tail": "w3", "head": "j1", "capacity": 1.0, "cost": 8.0},
                {"tail": "w3", "head": "j2", "capacity": 1.0, "cost": 11.0},
                {"tail": "w3", "head": "j3", "capacity": 1.0, "cost": 14.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        structure = analyze_network_structure(problem)

        assert structure.network_type == NetworkType.ASSIGNMENT
        assert structure.is_bipartite
        assert len(structure.source_nodes) == 3
        assert len(structure.sink_nodes) == 3
        assert structure.is_balanced
        assert structure.total_supply == 3.0
        assert structure.total_demand == 3.0

    def test_unbalanced_assignment_is_transportation(self):
        """Unbalanced assignment (different counts) is transportation."""
        # 2 workers, 2 jobs, but different total supply (not unit each)
        problem = build_problem(
            nodes=[
                {"id": "w1", "supply": 2.0},  # Can do 2 jobs
                {"id": "w2", "supply": 1.0},  # Can do 1 job
                {"id": "j1", "supply": -1.0},  # Needs 1 worker
                {"id": "j2", "supply": -2.0},  # Needs 2 workers
            ],
            arcs=[
                {"tail": "w1", "head": "j1", "capacity": 10.0, "cost": 10.0},
                {"tail": "w1", "head": "j2", "capacity": 10.0, "cost": 15.0},
                {"tail": "w2", "head": "j1", "capacity": 10.0, "cost": 12.0},
                {"tail": "w2", "head": "j2", "capacity": 10.0, "cost": 9.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        structure = analyze_network_structure(problem)

        # Not unit supply/demand, so transportation not assignment
        assert structure.network_type == NetworkType.TRANSPORTATION
        assert structure.is_balanced  # But still balanced


class TestBipartiteMatchingDetection:
    """Test detection of bipartite matching problems."""

    def test_detect_bipartite_matching(self):
        """Test detection of bipartite matching with unit values."""
        problem = build_problem(
            nodes=[
                {"id": "u1", "supply": 1.0},
                {"id": "u2", "supply": 1.0},
                {"id": "v1", "supply": -1.0},
                {"id": "v2", "supply": -1.0},
                {"id": "v3", "supply": 0.0},  # Unmatched node
            ],
            arcs=[
                {"tail": "u1", "head": "v1", "capacity": 1.0, "cost": 0.0},
                {"tail": "u1", "head": "v2", "capacity": 1.0, "cost": 0.0},
                {"tail": "u2", "head": "v2", "capacity": 1.0, "cost": 0.0},
                {"tail": "u2", "head": "v3", "capacity": 1.0, "cost": 0.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        structure = analyze_network_structure(problem)

        assert structure.network_type == NetworkType.BIPARTITE_MATCHING
        assert structure.is_bipartite
        assert len(structure.source_nodes) == 2
        assert len(structure.sink_nodes) == 2
        assert len(structure.transshipment_nodes) == 1


class TestMaxFlowDetection:
    """Test detection of max flow problems."""

    def test_detect_max_flow_single_source_sink(self):
        """Test detection of max flow with single source/sink."""
        problem = build_problem(
            nodes=[
                {"id": "s", "supply": 100.0},
                {"id": "a", "supply": 0.0},
                {"id": "b", "supply": 0.0},
                {"id": "t", "supply": -100.0},
            ],
            arcs=[
                {"tail": "s", "head": "a", "capacity": 50.0, "cost": 0.0},
                {"tail": "s", "head": "b", "capacity": 60.0, "cost": 0.0},
                {"tail": "a", "head": "t", "capacity": 40.0, "cost": 0.0},
                {"tail": "b", "head": "t", "capacity": 50.0, "cost": 0.0},
                {"tail": "a", "head": "b", "capacity": 30.0, "cost": 0.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        structure = analyze_network_structure(problem)

        assert structure.network_type == NetworkType.MAX_FLOW
        assert len(structure.source_nodes) == 1
        assert len(structure.sink_nodes) == 1


class TestShortestPathDetection:
    """Test detection of shortest path problems."""

    def test_detect_shortest_path_unit_flow(self):
        """Test detection of shortest path with unit flow."""
        problem = build_problem(
            nodes=[
                {"id": "s", "supply": 1.0},
                {"id": "a", "supply": 0.0},
                {"id": "b", "supply": 0.0},
                {"id": "t", "supply": -1.0},
            ],
            arcs=[
                {"tail": "s", "head": "a", "capacity": 10.0, "cost": 5.0},
                {"tail": "s", "head": "b", "capacity": 10.0, "cost": 3.0},
                {"tail": "a", "head": "t", "capacity": 10.0, "cost": 2.0},
                {"tail": "b", "head": "t", "capacity": 10.0, "cost": 4.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        structure = analyze_network_structure(problem)

        assert structure.network_type == NetworkType.SHORTEST_PATH
        assert structure.total_supply == 1.0
        assert structure.total_demand == 1.0


class TestBipartiteGraphDetection:
    """Test bipartite graph detection algorithm."""

    def test_simple_bipartite_graph(self):
        """Test that simple bipartite graph is detected."""
        problem = build_problem(
            nodes=[
                {"id": "a", "supply": 10.0},
                {"id": "b", "supply": 10.0},
                {"id": "x", "supply": -10.0},
                {"id": "y", "supply": -10.0},
            ],
            arcs=[
                {"tail": "a", "head": "x", "capacity": 20.0, "cost": 1.0},
                {"tail": "a", "head": "y", "capacity": 20.0, "cost": 2.0},
                {"tail": "b", "head": "x", "capacity": 20.0, "cost": 3.0},
                {"tail": "b", "head": "y", "capacity": 20.0, "cost": 4.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        structure = analyze_network_structure(problem)

        assert structure.is_bipartite
        assert structure.partitions is not None
        p0, p1 = structure.partitions
        # Check that sources are in one partition, sinks in the other
        assert ("a" in p0 and "b" in p0 and "x" in p1 and "y" in p1) or (
            "a" in p1 and "b" in p1 and "x" in p0 and "y" in p0
        )

    def test_non_bipartite_graph_with_cycle(self):
        """Test that graph with odd cycle is not bipartite."""
        problem = build_problem(
            nodes=[
                {"id": "a", "supply": 0.0},
                {"id": "b", "supply": 0.0},
                {"id": "c", "supply": 0.0},
            ],
            arcs=[
                # Triangle (odd cycle)
                {"tail": "a", "head": "b", "capacity": 10.0, "cost": 1.0},
                {"tail": "b", "head": "c", "capacity": 10.0, "cost": 1.0},
                {"tail": "c", "head": "a", "capacity": 10.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        structure = analyze_network_structure(problem)

        assert not structure.is_bipartite
        assert structure.partitions is None


class TestStructureProperties:
    """Test extraction of network structure properties."""

    def test_source_sink_transshipment_categorization(self):
        """Test correct categorization of node types."""
        problem = build_problem(
            nodes=[
                {"id": "source1", "supply": 50.0},
                {"id": "source2", "supply": 30.0},
                {"id": "transship", "supply": 0.0},
                {"id": "sink1", "supply": -40.0},
                {"id": "sink2", "supply": -40.0},
            ],
            arcs=[
                {"tail": "source1", "head": "transship", "capacity": 100.0, "cost": 1.0},
                {"tail": "source2", "head": "transship", "capacity": 100.0, "cost": 1.0},
                {"tail": "transship", "head": "sink1", "capacity": 100.0, "cost": 1.0},
                {"tail": "transship", "head": "sink2", "capacity": 100.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        structure = analyze_network_structure(problem)

        assert len(structure.source_nodes) == 2
        assert "source1" in structure.source_nodes
        assert "source2" in structure.source_nodes

        assert len(structure.sink_nodes) == 2
        assert "sink1" in structure.sink_nodes
        assert "sink2" in structure.sink_nodes

        assert len(structure.transshipment_nodes) == 1
        assert "transship" in structure.transshipment_nodes

        assert structure.total_supply == 80.0
        assert structure.total_demand == 80.0
        assert structure.is_balanced

    # Note: Cannot test unbalanced problems because build_problem validates balance.
    # All valid network problems must have balanced supply/demand.


class TestSpecializationInfo:
    """Test specialization info generation."""

    def test_transportation_problem_info(self):
        """Test info for transportation problem."""
        from network_solver import get_specialization_info

        problem = build_problem(
            nodes=[
                {"id": "s1", "supply": 50.0},
                {"id": "t1", "supply": -50.0},
            ],
            arcs=[
                {"tail": "s1", "head": "t1", "capacity": 100.0, "cost": 1.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        structure = analyze_network_structure(problem)
        info = get_specialization_info(structure)

        assert info["type"] == "transportation"
        assert info["is_bipartite"]
        assert "description" in info
        assert "optimization_hint" in info
        assert "transportation" in info["description"].lower()

    def test_general_problem_info(self):
        """Test info for general problem."""
        from network_solver import get_specialization_info

        problem = build_problem(
            nodes=[
                {"id": "a", "supply": 10.0},
                {"id": "b", "supply": 0.0},
                {"id": "c", "supply": -10.0},
            ],
            arcs=[
                {"tail": "a", "head": "b", "capacity": 20.0, "cost": 1.0},
                {"tail": "b", "head": "c", "capacity": 20.0, "cost": 1.0},
                {"tail": "a", "head": "c", "capacity": 20.0, "cost": 3.0},
            ],
            directed=True,
            tolerance=1e-6,
        )

        structure = analyze_network_structure(problem)
        info = get_specialization_info(structure)

        assert info["type"] == "general"
        assert "General minimum-cost flow" in info["description"]
