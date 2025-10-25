# Running Benchmarks

This guide explains how to run performance benchmarks on the min-cost flow solver using the benchmark runner framework.

## Quick Start

```bash
# Run on small instances (fast, good for testing)
python benchmarks/scripts/run_benchmark.py --small

# Run on medium instances (slower, for performance evaluation)
python benchmarks/scripts/run_benchmark.py --medium

# Save results to JSON
python benchmarks/scripts/run_benchmark.py --small --output results.json

# Generate markdown report
python benchmarks/scripts/run_benchmark.py --small --output report.md
```

**Prerequisites**: Download benchmark instances first using `download_dimacs.py` (see [DOWNLOADING.md](DOWNLOADING.md))

## Benchmark Runner Features

The `run_benchmark.py` script provides:

- **Automatic instance discovery** - Finds all instances in benchmark directories
- **Performance metrics collection** - Measures solve time, iterations, and objective values
- **Multiple output formats** - JSON, CSV, and markdown table formats
- **Progress reporting** - Live updates as benchmarks run
- **Error handling** - Gracefully handles solver errors and timeouts
- **Flexible filtering** - Run on specific files, directories, or size categories

## Usage Examples

### Run on Specific Size Categories

```bash
# Small instances (256-512 nodes, fast)
python benchmarks/scripts/run_benchmark.py --small

# Medium instances (4K-16K nodes, moderate)
python benchmarks/scripts/run_benchmark.py --medium

# Large instances (>16K nodes, slow) - Phase 6+
python benchmarks/scripts/run_benchmark.py --large
```

### Run on Specific Files or Directories

```bash
# Single instance
python benchmarks/scripts/run_benchmark.py --file benchmarks/problems/lemon/netgen/netgen_8_08a.min

# All instances in a directory
python benchmarks/scripts/run_benchmark.py --dir benchmarks/problems/lemon/netgen/

# Specific instance family
python benchmarks/scripts/run_benchmark.py --dir benchmarks/problems/lemon/gridgen/
```

### Output Formats

#### JSON Output

```bash
python benchmarks/scripts/run_benchmark.py --small --output results.json
```

Produces structured JSON with metadata and detailed results:

```json
{
  "metadata": {
    "timestamp": "2025-10-25 12:00:00",
    "total_instances": 11,
    "successful": 11,
    "failed": 0,
    "timeout": 0
  },
  "results": [
    {
      "instance_name": "netgen_8_08a",
      "instance_path": "benchmarks/problems/lemon/netgen/netgen_8_08a.min",
      "nodes": 256,
      "arcs": 2048,
      "status": "optimal",
      "objective": 142274536.0,
      "iterations": 777,
      "solve_time_ms": 12132.2,
      "parse_time_ms": 45.3,
      "total_time_ms": 12177.5,
      "error": null
    }
  ]
}
```

#### CSV Output

```bash
python benchmarks/scripts/run_benchmark.py --small --output results.csv
```

Produces CSV file suitable for spreadsheet analysis:

```csv
instance_name,instance_path,nodes,arcs,status,objective,iterations,solve_time_ms,parse_time_ms,total_time_ms,error
netgen_8_08a,benchmarks/problems/lemon/netgen/netgen_8_08a.min,256,2048,optimal,142274536.0,777,12132.2,45.3,12177.5,
```

#### Markdown Output

```bash
python benchmarks/scripts/run_benchmark.py --small --output report.md
```

Produces human-readable markdown table:

```markdown
# Benchmark Results

**Timestamp**: 2025-10-25 12:00:00
**Total Instances**: 11
**Successful**: 11
**Failed**: 0
**Timeout**: 0

## Results Table

| Instance | Nodes | Arcs | Status | Objective | Iterations | Solve Time (ms) |
|----------|-------|------|--------|-----------|------------|-----------------|
| netgen_8_08a | 256 | 2048 | optimal | 142274536.00 | 777 | 12132.20 |
| netgen_8_08b | 256 | 2048 | optimal | 143172848.00 | 769 | 11956.50 |
...
```

### Timeout Configuration

Set custom timeout (default is 300 seconds = 5 minutes):

```bash
# 60 second timeout per instance
python benchmarks/scripts/run_benchmark.py --medium --timeout 60

# 10 minute timeout for difficult instances
python benchmarks/scripts/run_benchmark.py --large --timeout 600
```

## Performance Metrics

The benchmark runner collects the following metrics for each instance:

- **instance_name**: Name of the instance file (without extension)
- **instance_path**: Full path to the instance file
- **nodes**: Number of nodes in the network
- **arcs**: Number of arcs in the network
- **status**: Solution status (`optimal`, `error`, `timeout`, etc.)
- **objective**: Optimal objective value (if solved successfully)
- **iterations**: Number of simplex iterations
- **solve_time_ms**: Time spent solving (milliseconds)
- **parse_time_ms**: Time spent parsing DIMACS file (milliseconds)
- **total_time_ms**: Total time (parse + solve)
- **error**: Error message (if solver failed)

## Benchmark Results Directory

Results can be saved to `benchmarks/results/` for tracking performance over time:

```bash
# Create results directory structure
mkdir -p benchmarks/results/{small,medium,large}

# Save results with timestamps
python benchmarks/scripts/run_benchmark.py --small --output "benchmarks/results/small/run-$(date +%Y%m%d-%H%M%S).json"

# Compare results across different versions
python benchmarks/scripts/run_benchmark.py --small --output benchmarks/results/small/baseline.json
# (after code changes)
python benchmarks/scripts/run_benchmark.py --small --output benchmarks/results/small/optimized.json
```

## Instance Size Guidelines

| Category | Nodes | Arcs | Solve Time | Use Case |
|----------|-------|------|------------|----------|
| **Small** | 256-512 | 2K-4K | ~10-15s | Quick testing, CI/CD |
| **Medium** | 4K-16K | 32K-128K | ~1-10 min | Performance evaluation |
| **Large** | >16K | >128K | >10 min | Scalability testing |

**Note**: Solve times are approximate and depend on instance difficulty and hardware.

## Command-Line Reference

```bash
python benchmarks/scripts/run_benchmark.py [OPTIONS]

Instance Selection:
  --small               Run on small instances (256-512 nodes)
  --medium              Run on medium instances (4K-16K nodes)
  --large               Run on large instances (>16K nodes)
  --file FILE           Run on specific instance file
  --dir DIR             Run on all instances in directory

Output Options:
  --output OUTPUT       Output file for results (.json, .csv, .md)
  --format FORMAT       Output format (json, csv, markdown)
                        Overrides file extension

Performance Options:
  --timeout TIMEOUT     Timeout in seconds per instance (default: 300)

Help:
  -h, --help           Show help message and exit
```

## Interpreting Results

### Success Rate

- **Status: optimal** - Solver found optimal solution
- **Status: error** - Parser or solver error occurred
- **Status: timeout** - Exceeded time limit

### Performance Analysis

Look at these metrics to evaluate solver performance:

1. **Solve time** - How fast is the solver?
2. **Iterations** - How efficient is the algorithm? (fewer is better)
3. **Success rate** - Percentage of instances solved successfully
4. **Scalability** - How does solve time grow with problem size?

### Example Analysis

```bash
# Run benchmarks
python benchmarks/scripts/run_benchmark.py --small --output results.json

# Check results
cat results.json | jq '.metadata'
# {
#   "total_instances": 11,
#   "successful": 11,      # 100% success rate
#   "failed": 0,
#   "timeout": 0
# }

# Find slowest instance
cat results.json | jq '.results | sort_by(.solve_time_ms) | reverse | .[0]'
```

## Integration with CI/CD

Run lightweight benchmarks in continuous integration:

```bash
# In .github/workflows/benchmark.yml or similar
- name: Run small benchmarks
  run: |
    python benchmarks/scripts/download_dimacs.py --small
    python benchmarks/scripts/run_benchmark.py --small --output benchmark-results.json --timeout 60
    
- name: Check performance regression
  run: |
    # Compare with baseline (implementation depends on your workflow)
    python scripts/compare_benchmarks.py baseline.json benchmark-results.json
```

## Troubleshooting

### No instances found

**Problem**: `No instances found to run`

**Solution**: Download instances first:
```bash
python benchmarks/scripts/download_dimacs.py --small
```

### Solver too slow

**Problem**: Benchmarks taking too long

**Solutions**:
- Use smaller instances: `--small` instead of `--medium`
- Reduce timeout: `--timeout 30`
- Run on specific fast instances: `--file path/to/instance.min`

### Import errors

**Problem**: `ModuleNotFoundError` when running script

**Solution**: Run from project root directory:
```bash
cd /path/to/network_flow_solver
python benchmarks/scripts/run_benchmark.py --small
```

## Future Enhancements (Phase 6+)

Planned improvements to the benchmark framework:

- **Parallel execution** - Run multiple benchmarks simultaneously
- **Comparison tools** - Compare results across solver versions
- **Visualization** - Generate performance charts and graphs
- **Profiling integration** - Collect detailed performance profiles
- **Known solutions validation** - Verify optimality against known solutions
- **Regression detection** - Automatically detect performance regressions

## See Also

- [DOWNLOADING.md](DOWNLOADING.md) - How to download benchmark instances
- [benchmarks/metadata/problem_catalog.json](metadata/problem_catalog.json) - Catalog of available instances
- [benchmarks/parsers/](parsers/) - DIMACS parser implementation

---

**Last Updated**: 2025-10-25  
**Phase**: Phase 5 - Medium Instances & Benchmark Runner
