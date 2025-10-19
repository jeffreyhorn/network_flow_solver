# Running the Jupyter Notebook Tutorial

## Quick Start

1. **Install tutorial dependencies** (if not already installed):
   ```bash
   pip install -e ".[tutorial]"
   ```
   
   This installs:
   - Jupyter (for running notebooks)
   - matplotlib (for visualizations)
   - networkx (for network graphs)

2. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   
   This will:
   - Start the Jupyter server
   - Open your web browser automatically
   - Show the file browser

3. **Navigate to the tutorial**:
   - In the browser, navigate to `tutorials/`
   - Click on `network_flow_tutorial.ipynb`

4. **Run the notebook**:
   - Click on each code cell and press `Shift+Enter` to execute
   - Or use `Cell → Run All` to execute all cells at once

## Alternative: JupyterLab (Modern Interface)

If you prefer the modern JupyterLab interface:

```bash
# Install JupyterLab
pip install jupyterlab

# Launch JupyterLab
jupyter lab
```

Then navigate to `tutorials/network_flow_tutorial.ipynb`

## Alternative: VS Code

If you use Visual Studio Code:

1. Install the "Jupyter" extension
2. Open the notebook file directly in VS Code
3. Click "Run All" or run cells individually

## Troubleshooting

### Jupyter command not found

Make sure you're in your virtual environment:
```bash
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

Then install the tutorial dependencies:
```bash
pip install -e ".[tutorial]"
```

### ModuleNotFoundError when running cells

The notebook expects the `network_solver` package to be installed. Make sure you've installed it:
```bash
pip install -e .
```

Or install everything at once:
```bash
pip install -e ".[all]"
```

### Visualization not working

Make sure matplotlib and networkx are installed:
```bash
pip install matplotlib networkx
```

Or install via the tutorial group:
```bash
pip install -e ".[tutorial]"
```

## What's in the Tutorial?

The notebook covers:

1. **Installation and Setup** - Getting started
2. **First Network Flow Problem** - Basic transportation problem
3. **Solving and Interpreting Results** - Understanding the solution
4. **Dual Values and Sensitivity Analysis** - Shadow prices
5. **Maximum Flow Problem** - Converting max-flow formulation
6. **Solver Configuration** - Tuning parameters
7. **Incremental Resolving** - Efficient re-solving
8. **Bottleneck Analysis** - Finding capacity constraints
9. **Visualization** (Optional) - Plotting networks
10. **Summary and Next Steps** - Key takeaways

Each section has:
- **Markdown cells** explaining the concepts
- **Code cells** with executable examples
- **Output** showing expected results

## Tips

- **Execute cells in order** - Later cells may depend on earlier ones
- **Experiment freely** - Modify the code and re-run to see what happens
- **Restart kernel** if things get messy: `Kernel → Restart & Clear Output`
- **Save your work** - The notebook autosaves, but you can also save manually with `Cmd+S` (macOS) or `Ctrl+S` (Windows/Linux)

## Next Steps

After completing the tutorial, check out:

- **[docs/examples.md](docs/examples.md)** - More detailed examples
- **[docs/api.md](docs/api.md)** - Complete API reference
- **[docs/algorithm.md](docs/algorithm.md)** - Algorithm details
- **[examples/](examples/)** - Standalone Python scripts
