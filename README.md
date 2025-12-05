# Constrained Proximal Policy Optimization (CPPO)

This repository reproduces the paper
**“Constrained Proximal Policy Optimization”** (*Xuan et al., 2023, arXiv:2305.14216*)
and provides a minimal, readable PyTorch implementation focused on the **E-step / M-step** framework of CPPO.

> 📄 Paper: [arXiv:2305.14216](https://arxiv.org/abs/2305.14216)


## 🌟 Overview

**Constrained Proximal Policy Optimization (CPPO)** reformulates Constrained Reinforcement Learning (CRL) as a **probabilistic inference problem**.
It introduces a two-step iterative framework:

1. **E-step:**
   Solves for the feasible posterior ratio \( v = \frac{q(a|s)}{p_\pi(a|s)} \) under reward, cost, and KL constraints.
2. **M-step:**
   Updates the policy parameters to track \( v \) with forward-KL regularization.

This repository provides:

* A faithful reimplementation of CPPO’s key components.
* Support for **Safety-Gymnasium** environments.
* Modularized code for **E-step**, **M-step**, **dual-branch GAE**, and **reward–cost critics**.
* Easy extensibility for model-based (Dyna-style) or multi-environment experiments.


## 🧩 Repository Structure

```text
cppo-main/
├── agents/                  # Policy and Value networks
│   ├── policy.py            # Gaussian policy class (μ, logσ)
│   └── value.py             # Dual critics: reward (V_r) and cost (V_c)
├── algo/                    # Algorithmic components
│   ├── advantages.py        # Dual-branch GAE for A and A_c
│   ├── estep.py             # E-step optimization for ratio v
│   └── mstep.py             # M-step (policy tracking update)
├── envs/                    # Environment wrappers
│   ├── safety_gym_wrappers.py  # Safety-Gymnasium interface
│   └── circle_wrappers.py      # (stub for Circle environments)
├── utils/                   # Utilities and helpers
│   ├── log.py               # Console/file logging
│   ├── schedule.py          # Linear learning-rate schedule
│   └── seed.py              # Global seeding for reproducibility
├── main.py                  # Main training entry point
├── environment.yml          # Conda environment file (Windows-specific)
└── logs/                    # Runtime logs (created automatically)
````

## ⚙️ Installation

### Python version

* The implementation has been run successfully with **Python 3.10**.
* With **Python 3.11**, the environment may fail to resolve or download correctly, so Python 3.10 is recommended.

### Option 1: Conda (recommended on Windows)

```bash
conda env create -f environment.yml
conda activate cppo
```

* The provided `environment.yml` is **Windows-specific**.

### Option 2: pip + virtualenv (cross-platform)

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install torch gymnasium numpy safety-gymnasium
# Optional (for rendering):
pip install mujoco mujoco-python-viewer
```

> On macOS/Apple Silicon, prefer `mujoco` over `mujoco-py` for compatibility.

## 🚀 Quick Start

Run the minimal training example:

```bash
python main.py
```

The default configuration (inside `main.py`) uses:

* `env_id = "SafetyPointGoal1-v0"`
* `total_iters = 10000`
* `horizon = 1024`
* `gamma = 0.99`, `lam = 0.95`
* `cost constraint d = 25.0`
* `KL budget δ = 0.02`
* `policy lr = 3e-4`, `value lr = 3e-4`
* `seed = 0`

Logs are automatically saved in `./logs/`.

## 🧭 Supported Environments

| Environment           | Source           | Default Constraint |
| --------------------- | ---------------- | ------------------ |
| `SafetyPointGoal1-v0` | Safety-Gymnasium | 25                 |
| `SafetyPointPush1-v0` | Safety-Gymnasium | 25                 |
| `SafetyCarPush1-v0`   | Safety-Gymnasium | 25                 |

To change the environment, edit the `env_id` in `main.py`.

## 🧾 Logging & Outputs

* Console prints iteration stats (reward, cost, KL, entropy, etc.).
* Log files are saved in `./logs/`.
* You can replace `utils/log.py` with TensorBoard or Weights & Biases for more advanced tracking if needed.

## 🔁 Reproducibility

* `utils/seed.py` provides seeding utilities (for Python, NumPy, and PyTorch) to improve reproducibility.
* The main script `main.py` sets a fixed `seed` by default (e.g., `seed = 0`).
* To reproduce a run, use the same Python version (3.10), the same dependencies, and the same random seed.

## 🔬 Extending the Repo

You can extend this base for:

* **Dyna-style CPPO:** add a learned dynamics model ( \hat{P}_\phi(s'|s,a) ) to generate model rollouts.
* **Parallel Environments:** wrap rollout collection using `gym.vector`.
* **Custom Constraints:** modify `algo/estep.py` to define new feasible regions.

## 🧪 Example Experiment Settings

| Parameter        | Description          | Example |
| ---------------- | -------------------- | ------- |
| `--horizon`      | rollout length       | 1024    |
| `--total-iters`  | total updates        | 1e5     |
| `--d-constraint` | cost constraint      | 25      |
| `--kl-budget`    | reverse KL radius    | 0.02    |
| `--ratio-lb`     | PPO lower clip bound | 0.6     |
| `--seed`         | random seed          | 0       |

*(Currently, arguments are passed via `main.py`; you can add `argparse` for CLI usage if desired.)*
