# Phase 5 Benchmark Runner Enhancements

## Summary

Completed Phase 5 of the Benchmark Suite Plan by adding missing features to `benchmarks/scripts/run_benchmark.py`.

## Enhancements Implemented

### 1. Solution Validation Against Known Optimal Values ✓

**Feature**: Compares computed objectives against known optimal solutions from `benchmarks/metadata/known_solutions.json`.

**Implementation**:
- `load_known_solutions()`: Loads known optimal values from metadata
- Computes absolute and relative error
- Validates with configurable tolerance (default: 1e-6)
- Reports validation status: `correct`, `incorrect_objective`, or `unknown`

**Output**:
- Shows `[VALIDATED]` tag for correct solutions
- Shows `[ERROR: off by X]` for incorrect solutions
- Summary includes validation statistics

### 2. Correctness Validation ✓

**Feature**: Validates solution correctness independent of known optimal values.

**Checks**:
- **Flow conservation**: Verifies supply/demand balance at all nodes
- **Capacity constraints**: Verifies flows respect lower bounds and capacities

**Implementation**:
- `validate_solution()`: Performs all correctness checks
- Handles both finite and infinite capacities
- Uses tolerance-based comparison (1e-6)

**Fields Added**:
- `flow_conservation_ok`: Boolean indicating if flows balance
- `capacity_constraints_ok`: Boolean indicating if capacity constraints satisfied

### 3. Optional Memory Tracking ✓

**Feature**: Tracks memory usage during solver execution (requires `psutil`).

**Implementation**:
- Checks for `psutil` availability at startup
- Gracefully degrades if `psutil` not installed
- Tracks memory before and after solving

**Usage**:
```bash
python benchmarks/scripts/run_benchmark.py --small --track-memory
```

**Fields Added**:
- `memory_mb`: Memory usage at solve start (MB)
- `peak_memory_mb`: Memory usage at solve end (MB)

**Installation**:
```bash
pip install psutil
```

### 4. Automatic Result Archiving ✓

**Feature**: Automatically archives results to `benchmarks/results/latest/` with timestamps.

**Implementation**:
- `archive_results()`: Saves timestamped JSON files
- Creates `benchmark_results_YYYYMMDD_HHMMSS.json`
- Also saves as `latest.json` for easy access
- Archives enabled by default, can be disabled with `--no-archive`

**Metadata Included**:
- Timestamp
- Total instances
- Success/failure/timeout counts
- Validation statistics (validated, incorrect)

**Directory Structure**:
```
benchmarks/results/latest/
├── benchmark_results_20251025_155448.json
├── benchmark_results_20251025_155347.json
└── latest.json  # Symlink to most recent
```

## New Command-Line Options

```bash
--track-memory          # Enable memory tracking (requires psutil)
--no-archive           # Disable automatic archiving
```

## Enhanced Output

### Progress Display
```
[1/3] Running goto_8_08a.min... ✓ 13012.6ms, 1869 iterations [VALIDATED]
[2/3] Running netgen_8_08a.min... ✓ 4920.7ms, 777 iterations
[3/3] Running gridgen_8_08a.min... ✓ 3845.2ms, 652 iterations
```

### Summary Statistics
```
======================================================================
BENCHMARK SUMMARY
======================================================================
Total instances: 3
  ✓ Successful: 3
  ✗ Failed: 0
  ⏱ Timeout: 0

Validation (against known solutions):
  ✓ Validated correct: 1
  ✗ Incorrect objective: 0

Performance:
  Average solve time: 7259.50 ms
  Average iterations: 1099
  Average memory: 145.2 MB
  Average peak memory: 156.8 MB
======================================================================
```

## Result Format

### JSON Output Fields

```json
{
  "instance_name": "goto_8_08a",
  "instance_path": "benchmarks/problems/lemon/goto/goto_8_08a.min",
  "nodes": 256,
  "arcs": 2048,
  "status": "optimal",
  "objective": 560870539.0,
  "iterations": 1869,
  "solve_time_ms": 13012.6,
  "parse_time_ms": 20.3,
  "total_time_ms": 13032.9,
  "error": null,
  
  // NEW: Validation fields
  "known_optimal": 560870539.0,
  "objective_error": 0.0,
  "validation_status": "correct",
  
  // NEW: Memory tracking (optional)
  "memory_mb": 145.2,
  "peak_memory_mb": 156.8,
  
  // NEW: Correctness checks
  "flow_conservation_ok": true,
  "capacity_constraints_ok": true
}
```

## Testing

**Tested on**:
- Single file: `goto_8_08a.min` - ✓ All features work
- Multiple files: `netgen_8_08a.min` - ✓ Archiving works
- Validation: Flow conservation and capacity checks pass

**Lint/Format**: ✓ All checks pass

## Phase 5 Completion Status

| Requirement | Status |
|-------------|--------|
| Benchmark runner script | ✅ Complete |
| Performance metrics (time, iterations) | ✅ Complete |
| Memory tracking (optional with psutil) | ✅ Complete |
| Result storage (JSON/CSV/Markdown) | ✅ Complete |
| Correctness validation | ✅ Complete |
| Known solution comparison | ✅ Complete |
| Automatic archiving | ✅ Complete |

**Phase 5**: ✅ **COMPLETE**

## Next Steps

**Phase 6**: Solver Comparison Framework
- Implement solver adapters (OR-Tools, NetworkX, LEMON)
- Create multi-solver comparison runner
- Compare our solver against established implementations
