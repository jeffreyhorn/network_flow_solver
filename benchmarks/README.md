# Benchmark Scripts

This directory contains benchmark scripts demonstrating the performance improvements from optimization projects.

## Available Benchmarks

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

## Running All Benchmarks

```bash
# Run all benchmarks
for script in benchmarks/benchmark_*.py; do
    echo "Running $script..."
    python "$script"
    echo ""
done
```

## Benchmark Methodology

All benchmarks:
1. Use realistic network flow problems of varying sizes
2. Compare performance with and without optimizations (where applicable)
3. Report speedup ratios and time savings
4. Validate that solutions are identical (correctness check)

## See Also

- **Project Plans:** `docs/project_plans/` - Detailed planning documents and analysis
- **Cumulative Analysis:** `docs/project_plans/CUMULATIVE_SPEEDUP_ANALYSIS.md` - Overall speedup summary
- **Optimization Roadmap:** `docs/project_plans/OPTIMIZATION_PROJECTS_2025.md` - Complete roadmap
