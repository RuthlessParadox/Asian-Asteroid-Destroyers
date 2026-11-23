[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Asian-Asteroid-Destroyers

> This project uses evolutionary training to evolve fuzzy systems for a Kessler game controller in preparation for the Explainable Fuzzy Challenge. Built for CSP 200 at DigiPen Institute of Technology.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

1. **Install `uv`** (if you haven't already):
   ```bash
   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   ```bash
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone the repository and install dependencies:**
   ```bash
   # Clone the repository.
   git clone https://github.com/RuthlessParadox/Asian-Asteroid-Destroyers
   # Move your terminal into the project folder.
   cd Asian-Asteroid-Destroyers
   # Create a virtual environment if necessary, and install dependencies.
   uv sync
   ```

## Usage and Examples

### Starting a Training Session

```bash
# Unspecified hyperparameters use defaults defined in pyproject.toml under [tool.evolution]
uv run scripts/evolve.py 
```

```bash
# Train with 500 generations and 160 individuals, saving every 20 generations with cx_pb 0.6 and mut_pb 0.35 using custom scenarios.
uv run scripts/evolve.py --generations 500 --population-size 160 --save-interval 20 --cx-pb 0.6 --mut-pb 0.35 --scenarios-path "custom_scenarios.json"
```

### Resuming a Training Session

```bash
# Resume the training session whose data is stored in the directory "example_run".
uv run scripts/evolve.py --resume "example_run"
```

### Evaluating an Individual

```bash
# Evaluate an elite individual from a history and logbook using specified scenarios.
uv run scripts/evaluate_elite.py --history-path "example_history.parquet" --logbook-path "example_logbook.parquet" --scenarios-path "example_scenarios.json"
```

## Project Structure

```text
Asian-Asteroid-Destroyers/
├── src/
│   ├── asian_asteroid_destroyers/
│   │   ├── evokit/         # Core evolutionary algorithm logic
│   │   ├── controllers/    # Kessler controllers
│   │   ├── infrastructure/ # Infrastructure necessary to run training
│   │   ├── analysis/       # Training data analysis
│   │   ├── config/         # Argument parsing
│   │   ├── geometry/       # Geometric utilities
│   │   ├── evaluation.py   # Evaluator for evolutionary training
│   │   ├── chromosomes.py  # Fuzzy chromosome and related functionality
│   │   ├── fuzzy.py        # Fuzzy library
│   │   ├── sa.py           # Situational awareness for controllers
│   │   ├── scenarios.py    # Loading scenarios from JSONs
├── scripts/                # Scripts to run from the terminal
├── assets/
│   ├── scenarios/          # JSONs containing lists of scenarios
├── tests/                  # Unit and integration tests
├── typings/                # Type stubs for basedpyright
├── pyproject.toml          # Dependencies, build settings, and tool config
├── LICENSE                 # MIT License
└── README.md
```

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more
information.
