# Benchmarks

This directory contains benchmark infrastructure for the network flow solver, including performance benchmarks, solver comparisons, and test problem collections.

## Directory Structure

```
benchmarks/
├── README.md                          # This file
├── problems/                          # Benchmark problem files (not committed to git)
│   ├── dimacs/                        # DIMACS format problems
│   │   ├── netgen/                    # NETGEN-generated instances
│   │   ├── gridgen/                   # GRIDGEN instances
│   │   ├── goto/                      # GOTO instances
│   │   └── real_world/                # Real-world instances (TIGER/Line, etc.)
│   ├── or_library/                    # OR-Library instances
│   ├── lemon/                         # LEMON benchmark instances
│   ├── commalab/                      # CommaLAB multicommodity instances
│   └── generated/                     # Locally generated test problems
├── metadata/                          # Problem metadata, solutions, licenses
│   ├── problem_catalog.json           # Catalog of all problems
│   ├── known_solutions.json           # Known optimal solutions
│   └── licenses.json                  # License information for datasets
├── results/                           # Benchmark results
│   ├── latest/                        # Most recent benchmark run
│   └── reports/                       # Generated reports
├── solvers/                           # Solver adapter implementations (future)
├── parsers/                           # Format parsers (future)
└── benchmark_*.py                     # Benchmark scripts
```

**Note**: Problem files in `benchmarks/problems/` are excluded from git via `.gitignore`. Use download scripts to obtain benchmark instances.

---

## Performance Benchmarks

These benchmarks demonstrate performance improvements from optimization projects.

### benchmark_cumulative_speedup.py
**Cumulative Performance Analysis**

Measures the combined speedup from all optimization projects.

**Usage:**
```bash
python benchmarks/benchmark_cumulative_speedup.py
```

**See Also**: `docs/project_plans/CUMULATIVE_SPEEDUP_ANALYSIS.md`

---

### benchmark_jit_compilation.py
**JIT Compilation with Numba**

Demonstrates performance improvement from Numba JIT compilation of hot-path functions.

**Usage:**
```bash
python benchmarks/benchmark_jit_compilation.py
```

---

### benchmark_vectorized_pricing.py
**Project 2: Vectorize Pricing Operations**

Demonstrates the performance improvement from vectorized pricing operations using NumPy arrays.

**Results:**
- Small problems (300 arcs): 162% speedup (2.6x faster)
- Medium problems (600 arcs): 92% speedup (1.9x faster)  
- Average improvement: 127% speedup (2.3x faster)

**Usage:**
```bash
python benchmarks/benchmark_vectorized_pricing.py
```

---

### benchmark_batch_devex.py
**Project 3: Batch Devex Weight Updates (Deferred Updates)**

Demonstrates the reduction in weight update calls from deferred updates.

**Results:**
- Weight update calls: 127,063 → ~357 (97.5% reduction)
- Loop-based pricing: 37% faster than previous implementation
- Maintains same convergence behavior

**Usage:**
```bash
python benchmarks/benchmark_batch_devex.py
```

---

### benchmark_vectorize_residuals.py
**Project 4: Vectorize Residual Calculations**

Demonstrates the elimination of residual calculation function calls through cached arrays.

**Results:**
- Forward residual calls: 750,014 → 0 (100% eliminated)
- Backward residual calls: 749,955 → 0 (100% eliminated)
- Total function calls eliminated: ~1.5 million per solve
- Expected time saved: ~2.5s (3.7% of baseline)

**Usage:**
```bash
python benchmarks/benchmark_vectorize_residuals.py
```

---

## Benchmark Problem Collections

Publicly available benchmark problems for minimum cost network flow testing and solver comparison.

### Documentation

- **`docs/benchmarks/BENCHMARK_SOURCES.md`** - Comprehensive catalog of benchmark sources
  - 6 benchmark repositories documented
  - 4 problem generators cataloged
  - License information and restrictions
  - Access methods and citations

- **`benchmarks/metadata/licenses.json`** - Structured license audit
  - License terms for each source
  - Redistribution permissions
  - Citation requirements

### Recommended Sources (MVP)

1. **DIMACS** - Public domain, standard format, large collection
2. **LEMON** - Boost Software License 1.0, high-quality benchmark suite

### Problem Collection (Phase 3+)

Problem files are **not committed** to the repository. Instead, download scripts will be provided to obtain benchmark instances from their original sources.

**Rationale**:
- Respects license terms (attribution, non-commercial restrictions)
- Avoids repository bloat (some instances are large)
- Ensures users get latest versions from authoritative sources

**Directory organization** (when downloaded):
- `benchmarks/problems/dimacs/netgen/` - NETGEN-generated instances
- `benchmarks/problems/dimacs/gridgen/` - GRIDGEN instances
- `benchmarks/problems/lemon/` - LEMON benchmark suite
- `benchmarks/problems/generated/` - Locally generated test instances

---

## Running All Performance Benchmarks

```bash
# Run all benchmark scripts
for script in benchmarks/benchmark_*.py; do
    echo "Running $script..."
    python "$script"
    echo ""
done
```

---

## Benchmark Methodology

All performance benchmarks:
1. Use realistic network flow problems of varying sizes
2. Compare performance with and without optimizations (where applicable)
3. Report speedup ratios and time savings
4. Validate that solutions are identical (correctness check)

---

## Future Phases

The benchmark suite is being developed in phases (see `docs/project_plans/BENCHMARK_SUITE_PLAN.md`):

- **Phase 1 (Complete)**: Research and cataloging of benchmark sources
- **Phase 2 (In Progress)**: Directory structure and organization
- **Phase 3**: Problem download scripts and DIMACS parser
- **Phase 4**: Solver adapters (OR-Tools, NetworkX, LEMON)
- **Phase 5**: Comparison framework and result analysis
- **Phase 6**: Visualization and reporting
- **Phase 7**: Documentation and user guide
- **Phase 8**: Integration with CI/CD

---

## See Also

- **Benchmark Sources**: `docs/benchmarks/BENCHMARK_SOURCES.md` - Detailed catalog with licenses
- **Benchmark Suite Plan**: `docs/project_plans/BENCHMARK_SUITE_PLAN.md` - 8-phase development plan
- **Project Plans**: `docs/project_plans/` - Optimization project planning documents
- **Cumulative Analysis**: `docs/project_plans/CUMULATIVE_SPEEDUP_ANALYSIS.md` - Overall speedup summary
- **Optimization Roadmap**: `docs/project_plans/OPTIMIZATION_PROJECTS_2025.md` - Complete roadmap

---

## License and Attribution

Benchmark problems from external sources retain their original licenses. See:
- `benchmarks/metadata/licenses.json` - License information for all sources
- `docs/benchmarks/BENCHMARK_SOURCES.md` - Detailed license terms and restrictions

Always provide proper attribution when using or redistributing benchmark instances.

---

**Last Updated**: 2025-10-25  
**Phase**: Phase 2 - Directory Structure and Organization
