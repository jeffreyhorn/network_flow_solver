"""Tests for visualization utilities."""

import pytest

from network_solver import build_problem, solve_min_cost_flow

# Check if visualization dependencies are available
try:
    import matplotlib.pyplot as plt

    from network_solver import visualize_bottlenecks, visualize_flows, visualize_network

    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    plt = None

pytestmark = pytest.mark.skipif(
    not HAS_VISUALIZATION,
    reason="Visualization dependencies (matplotlib, networkx) not installed",
)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    if plt is not None:
        plt.close("all")


@pytest.fixture
def simple_problem():
    """Create a simple transportation problem for testing."""
    nodes = [
        {"id": "factory", "supply": 100.0},
        {"id": "warehouse", "supply": -100.0},
    ]
    arcs = [
        {"tail": "factory", "head": "warehouse", "capacity": 150.0, "cost": 2.5},
    ]
    return build_problem(nodes, arcs, directed=True, tolerance=1e-6)


@pytest.fixture
def multi_node_problem():
    """Create a problem with multiple nodes and arcs."""
    nodes = [
        {"id": "factory_a", "supply": 100.0},
        {"id": "factory_b", "supply": 50.0},
        {"id": "warehouse_1", "supply": -80.0},
        {"id": "warehouse_2", "supply": -70.0},
    ]
    arcs = [
        {"tail": "factory_a", "head": "warehouse_1", "capacity": 100.0, "cost": 2.0},
        {"tail": "factory_a", "head": "warehouse_2", "capacity": 100.0, "cost": 3.0},
        {"tail": "factory_b", "head": "warehouse_1", "capacity": 50.0, "cost": 1.5},
        {"tail": "factory_b", "head": "warehouse_2", "capacity": 50.0, "cost": 2.5},
    ]
    return build_problem(nodes, arcs, directed=True, tolerance=1e-6)


@pytest.fixture
def transshipment_problem():
    """Create a problem with transshipment nodes."""
    nodes = [
        {"id": "source", "supply": 100.0},
        {"id": "hub", "supply": 0.0},
        {"id": "sink", "supply": -100.0},
    ]
    arcs = [
        {"tail": "source", "head": "hub", "capacity": 120.0, "cost": 1.0},
        {"tail": "hub", "head": "sink", "capacity": 120.0, "cost": 1.5},
    ]
    return build_problem(nodes, arcs, directed=True, tolerance=1e-6)


@pytest.fixture
def bottleneck_problem():
    """Create a problem with a clear bottleneck."""
    nodes = [
        {"id": "source", "supply": 100.0},
        {"id": "sink", "supply": -100.0},
    ]
    arcs = [
        {"tail": "source", "head": "sink", "capacity": 100.0, "cost": 1.0},  # Tight capacity
    ]
    return build_problem(nodes, arcs, directed=True, tolerance=1e-6)


class TestVisualizeNetwork:
    """Tests for visualize_network function."""

    def test_simple_network(self, simple_problem):
        """Test visualization of simple network."""
        fig = visualize_network(simple_problem)
        assert fig is not None
        assert len(fig.axes) == 1

    def test_multi_node_network(self, multi_node_problem):
        """Test visualization of network with multiple nodes."""
        fig = visualize_network(multi_node_problem)
        assert fig is not None
        assert len(fig.axes) == 1

    def test_custom_layout(self, simple_problem):
        """Test different layout algorithms."""
        for layout in ["spring", "circular", "kamada_kawai"]:
            fig = visualize_network(simple_problem, layout=layout)
            assert fig is not None

    def test_invalid_layout_fallback(self, simple_problem):
        """Test that invalid layout falls back to spring."""
        fig = visualize_network(simple_problem, layout="invalid_layout")
        assert fig is not None

    def test_custom_figsize(self, simple_problem):
        """Test custom figure size."""
        fig = visualize_network(simple_problem, figsize=(10, 6))
        assert fig is not None
        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 6

    def test_hide_arc_labels(self, simple_problem):
        """Test hiding arc labels."""
        fig = visualize_network(simple_problem, show_arc_labels=False)
        assert fig is not None

    def test_custom_title(self, simple_problem):
        """Test custom title."""
        custom_title = "My Custom Network"
        fig = visualize_network(simple_problem, title=custom_title)
        assert fig is not None
        assert fig.axes[0].get_title() == custom_title

    def test_transshipment_nodes(self, transshipment_problem):
        """Test visualization with transshipment nodes."""
        fig = visualize_network(transshipment_problem)
        assert fig is not None


class TestVisualizeFlows:
    """Tests for visualize_flows function."""

    def test_simple_flows(self, simple_problem):
        """Test visualization of simple flow solution."""
        result = solve_min_cost_flow(simple_problem)
        fig = visualize_flows(simple_problem, result)
        assert fig is not None
        assert len(fig.axes) == 1

    def test_multi_node_flows(self, multi_node_problem):
        """Test visualization of multi-node flow solution."""
        result = solve_min_cost_flow(multi_node_problem)
        fig = visualize_flows(multi_node_problem, result)
        assert fig is not None

    def test_highlight_bottlenecks(self, bottleneck_problem):
        """Test bottleneck highlighting."""
        result = solve_min_cost_flow(bottleneck_problem)
        fig = visualize_flows(
            bottleneck_problem, result, highlight_bottlenecks=True, bottleneck_threshold=0.9
        )
        assert fig is not None

    def test_no_highlight_bottlenecks(self, bottleneck_problem):
        """Test without bottleneck highlighting."""
        result = solve_min_cost_flow(bottleneck_problem)
        fig = visualize_flows(bottleneck_problem, result, highlight_bottlenecks=False)
        assert fig is not None

    def test_show_zero_flows(self, multi_node_problem):
        """Test showing zero flows."""
        result = solve_min_cost_flow(multi_node_problem)
        fig = visualize_flows(multi_node_problem, result, show_zero_flows=True)
        assert fig is not None

    def test_hide_zero_flows(self, multi_node_problem):
        """Test hiding zero flows."""
        result = solve_min_cost_flow(multi_node_problem)
        fig = visualize_flows(multi_node_problem, result, show_zero_flows=False)
        assert fig is not None

    def test_custom_layout(self, simple_problem):
        """Test different layout algorithms."""
        result = solve_min_cost_flow(simple_problem)
        for layout in ["spring", "circular"]:
            fig = visualize_flows(simple_problem, result, layout=layout)
            assert fig is not None

    def test_custom_title(self, simple_problem):
        """Test custom title."""
        result = solve_min_cost_flow(simple_problem)
        custom_title = "My Flow Solution"
        fig = visualize_flows(simple_problem, result, title=custom_title)
        assert fig is not None
        assert fig.axes[0].get_title() == custom_title

    def test_custom_bottleneck_threshold(self, bottleneck_problem):
        """Test custom bottleneck threshold."""
        result = solve_min_cost_flow(bottleneck_problem)
        fig = visualize_flows(
            bottleneck_problem,
            result,
            highlight_bottlenecks=True,
            bottleneck_threshold=0.5,
        )
        assert fig is not None


class TestVisualizeBottlenecks:
    """Tests for visualize_bottlenecks function."""

    def test_bottleneck_visualization(self, bottleneck_problem):
        """Test bottleneck visualization."""
        result = solve_min_cost_flow(bottleneck_problem)
        fig = visualize_bottlenecks(bottleneck_problem, result, threshold=0.9)
        assert fig is not None
        assert len(fig.axes) == 2  # Main axis + colorbar

    def test_no_bottlenecks(self):
        """Test visualization when no bottlenecks exist."""
        # Create problem with large capacity (no bottlenecks)
        nodes = [
            {"id": "source", "supply": 100.0},
            {"id": "sink", "supply": -100.0},
        ]
        arcs = [
            {"tail": "source", "head": "sink", "capacity": 1000.0, "cost": 1.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
        result = solve_min_cost_flow(problem)
        fig = visualize_bottlenecks(problem, result, threshold=0.95)
        assert fig is not None
        # Should show "No bottlenecks found" message

    def test_custom_threshold(self, bottleneck_problem):
        """Test custom bottleneck threshold."""
        result = solve_min_cost_flow(bottleneck_problem)
        fig = visualize_bottlenecks(bottleneck_problem, result, threshold=0.5)
        assert fig is not None

    def test_custom_layout(self, bottleneck_problem):
        """Test different layout algorithms."""
        result = solve_min_cost_flow(bottleneck_problem)
        for layout in ["spring", "circular"]:
            fig = visualize_bottlenecks(bottleneck_problem, result, layout=layout)
            assert fig is not None

    def test_custom_title(self, bottleneck_problem):
        """Test custom title."""
        result = solve_min_cost_flow(bottleneck_problem)
        custom_title = "Bottleneck Analysis Custom"
        fig = visualize_bottlenecks(bottleneck_problem, result, title=custom_title)
        assert fig is not None
        assert fig.axes[0].get_title() == custom_title

    def test_multi_node_bottlenecks(self):
        """Test bottleneck visualization with multiple nodes."""
        # Create problem with tight capacities (bottlenecks)
        nodes = [
            {"id": "factory_a", "supply": 100.0},
            {"id": "factory_b", "supply": 50.0},
            {"id": "warehouse_1", "supply": -80.0},
            {"id": "warehouse_2", "supply": -70.0},
        ]
        arcs = [
            {"tail": "factory_a", "head": "warehouse_1", "capacity": 60.0, "cost": 2.0},
            {"tail": "factory_a", "head": "warehouse_2", "capacity": 60.0, "cost": 3.0},
            {"tail": "factory_b", "head": "warehouse_1", "capacity": 60.0, "cost": 1.5},
            {"tail": "factory_b", "head": "warehouse_2", "capacity": 60.0, "cost": 2.5},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
        result = solve_min_cost_flow(problem)
        fig = visualize_bottlenecks(problem, result, threshold=0.8)
        assert fig is not None


class TestVisualizationErrors:
    """Tests for error handling in visualization functions."""

    def test_missing_dependencies_error(self, simple_problem, monkeypatch):
        """Test that missing dependencies raise ImportError."""
        # This test only runs if visualization IS installed
        # We simulate the missing dependencies scenario
        import network_solver.visualization as viz_module

        monkeypatch.setattr(viz_module, "_HAS_VISUALIZATION_DEPS", False)

        with pytest.raises(ImportError, match="Visualization requires optional dependencies"):
            viz_module.visualize_network(simple_problem)

        with pytest.raises(ImportError, match="Visualization requires optional dependencies"):
            result = solve_min_cost_flow(simple_problem)
            viz_module.visualize_flows(simple_problem, result)

        with pytest.raises(ImportError, match="Visualization requires optional dependencies"):
            result = solve_min_cost_flow(simple_problem)
            viz_module.visualize_bottlenecks(simple_problem, result)


class TestVisualizationIntegration:
    """Integration tests for visualization functions."""

    def test_network_then_flows(self, multi_node_problem):
        """Test creating network and flow visualizations in sequence."""
        # Visualize network structure
        fig1 = visualize_network(multi_node_problem)
        assert fig1 is not None

        # Solve and visualize flows
        result = solve_min_cost_flow(multi_node_problem)
        fig2 = visualize_flows(multi_node_problem, result)
        assert fig2 is not None

        # Visualize bottlenecks
        fig3 = visualize_bottlenecks(multi_node_problem, result, threshold=0.5)
        assert fig3 is not None

    def test_all_layouts(self, simple_problem):
        """Test all layout algorithms work."""
        layouts = ["spring", "circular", "kamada_kawai"]
        result = solve_min_cost_flow(simple_problem)

        for layout in layouts:
            fig1 = visualize_network(simple_problem, layout=layout)
            assert fig1 is not None

            fig2 = visualize_flows(simple_problem, result, layout=layout)
            assert fig2 is not None

            fig3 = visualize_bottlenecks(simple_problem, result, layout=layout)
            assert fig3 is not None


class TestVisualizationEdgeCases:
    """Test edge cases and error handling in visualization functions."""

    def test_infinite_capacity_display(self):
        """Test that arcs with infinite capacity display ∞ symbol."""
        nodes = [
            {"id": "source", "supply": 100.0},
            {"id": "sink", "supply": -100.0},
        ]
        arcs = [
            {"tail": "source", "head": "sink", "capacity": None, "cost": 1.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
        fig = visualize_network(problem, show_arc_labels=True)
        assert fig is not None
        plt.close(fig)

    def test_undirected_network_visualization(self):
        """Test visualization of undirected network."""
        nodes = [
            {"id": "node1", "supply": 50.0},
            {"id": "node2", "supply": -50.0},
        ]
        arcs = [
            {"tail": "node1", "head": "node2", "capacity": 100.0, "cost": 1.0},
        ]
        problem = build_problem(nodes, arcs, directed=False, tolerance=1e-6)
        fig = visualize_network(problem)
        assert fig is not None
        plt.close(fig)

    def test_undirected_flows_visualization(self):
        """Test flow visualization of undirected network."""
        nodes = [
            {"id": "node1", "supply": 50.0},
            {"id": "node2", "supply": -50.0},
        ]
        arcs = [
            {"tail": "node1", "head": "node2", "capacity": 100.0, "cost": 1.0},
        ]
        problem = build_problem(nodes, arcs, directed=False, tolerance=1e-6)
        result = solve_min_cost_flow(problem)
        fig = visualize_flows(problem, result)
        assert fig is not None
        plt.close(fig)

    def test_undirected_bottlenecks_visualization(self):
        """Test bottleneck visualization of undirected network."""
        nodes = [
            {"id": "source", "supply": 100.0},
            {"id": "hub", "supply": 0.0},
            {"id": "sink", "supply": -100.0},
        ]
        arcs = [
            {"tail": "source", "head": "hub", "capacity": 150.0, "cost": 1.0},
            {"tail": "hub", "head": "sink", "capacity": 100.0, "cost": 1.0},
        ]
        problem = build_problem(nodes, arcs, directed=False, tolerance=1e-6)
        result = solve_min_cost_flow(problem)
        fig = visualize_bottlenecks(problem, result, threshold=0.9)
        assert fig is not None
        plt.close(fig)

    def test_planar_layout_network(self):
        """Test planar layout algorithm for network visualization."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 150.0, "cost": 1.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
        # Planar layout may work or fall back to spring
        fig = visualize_network(problem, layout="planar")
        assert fig is not None
        plt.close(fig)

    def test_planar_layout_flows(self):
        """Test planar layout algorithm for flow visualization."""
        nodes = [
            {"id": "A", "supply": 100.0},
            {"id": "B", "supply": -100.0},
        ]
        arcs = [
            {"tail": "A", "head": "B", "capacity": 150.0, "cost": 1.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
        result = solve_min_cost_flow(problem)
        fig = visualize_flows(problem, result, layout="planar")
        assert fig is not None
        plt.close(fig)

    def test_planar_layout_bottlenecks(self):
        """Test planar layout algorithm for bottleneck visualization."""
        nodes = [
            {"id": "source", "supply": 100.0},
            {"id": "sink", "supply": -100.0},
        ]
        arcs = [
            {"tail": "source", "head": "sink", "capacity": 100.0, "cost": 1.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
        result = solve_min_cost_flow(problem)
        fig = visualize_bottlenecks(problem, result, layout="planar", threshold=0.9)
        assert fig is not None
        plt.close(fig)

    def test_mostly_source_nodes(self):
        """Test visualization when most nodes are sources (supply > 0)."""
        nodes = [
            {"id": "source1", "supply": 50.0},
            {"id": "source2", "supply": 50.0},
            {"id": "sink", "supply": -100.0},
        ]
        arcs = [
            {"tail": "source1", "head": "sink", "capacity": 100.0, "cost": 1.0},
            {"tail": "source2", "head": "sink", "capacity": 100.0, "cost": 1.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
        fig = visualize_network(problem)
        assert fig is not None
        plt.close(fig)

    def test_all_transshipment_nodes(self):
        """Test visualization when all nodes are transshipment (supply = 0)."""
        nodes = [
            {"id": "node1", "supply": 0.0},
            {"id": "node2", "supply": 0.0},
        ]
        arcs = [
            {"tail": "node1", "head": "node2", "capacity": 100.0, "cost": 1.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
        fig = visualize_network(problem)
        assert fig is not None
        plt.close(fig)

    def test_visualize_flows_zero_capacity_arc(self):
        """Test flow visualization with arc capacity below tolerance."""
        nodes = [
            {"id": "source", "supply": 10.0},
            {"id": "sink", "supply": -10.0},
        ]
        arcs = [
            {"tail": "source", "head": "sink", "capacity": 1e-10, "cost": 1.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
        result = solve_min_cost_flow(problem)
        # Should not crash on division by near-zero capacity
        fig = visualize_flows(problem, result, highlight_bottlenecks=True)
        assert fig is not None
        plt.close(fig)

    def test_visualize_bottlenecks_with_infinite_capacity(self):
        """Test bottleneck visualization skips arcs with infinite capacity."""
        nodes = [
            {"id": "source", "supply": 100.0},
            {"id": "hub", "supply": 0.0},
            {"id": "sink", "supply": -100.0},
        ]
        arcs = [
            {"tail": "source", "head": "hub", "capacity": None, "cost": 1.0},
            {"tail": "hub", "head": "sink", "capacity": 100.0, "cost": 1.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
        result = solve_min_cost_flow(problem)
        fig = visualize_bottlenecks(problem, result, threshold=0.9)
        assert fig is not None
        plt.close(fig)

    def test_visualize_bottlenecks_with_zero_capacity(self):
        """Test bottleneck visualization when arc has near-zero capacity."""
        nodes = [
            {"id": "source", "supply": 100.0},
            {"id": "hub", "supply": 0.0},
            {"id": "sink", "supply": -100.0},
        ]
        arcs = [
            {"tail": "source", "head": "hub", "capacity": 1e-10, "cost": 1.0},
            {"tail": "hub", "head": "sink", "capacity": 120.0, "cost": 1.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
        result = solve_min_cost_flow(problem)
        # Should skip near-zero capacity arc
        fig = visualize_bottlenecks(problem, result, threshold=0.8)
        assert fig is not None
        plt.close(fig)

    def test_visualize_flows_all_zero_flows(self):
        """Test flow visualization when all flows are zero."""
        nodes = [
            {"id": "source", "supply": 0.0},
            {"id": "sink", "supply": 0.0},
        ]
        arcs = [
            {"tail": "source", "head": "sink", "capacity": 100.0, "cost": 1.0},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
        result = solve_min_cost_flow(problem)
        fig = visualize_flows(problem, result, show_zero_flows=True)
        assert fig is not None
        plt.close(fig)

    def test_visualize_network_large_numbers(self):
        """Test visualization with very large costs and capacities."""
        nodes = [
            {"id": "source", "supply": 1e9},
            {"id": "sink", "supply": -1e9},
        ]
        arcs = [
            {"tail": "source", "head": "sink", "capacity": 1e12, "cost": 1e6},
        ]
        problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
        fig = visualize_network(problem)
        assert fig is not None
        plt.close(fig)
