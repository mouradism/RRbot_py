# RRbot_py

A Python simulation and control framework for a **3-DOF Cable-Driven Robot** (Planar 3-Link Redundant Manipulator). This project implements forward/inverse kinematics, dynamics computation, cable-based actuation, trajectory planning, and closed-loop control.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Running Tests](#running-tests)
- [Architecture](#architecture)
- [Key Algorithms](#key-algorithms)
- [License](#license)

---

## Overview

RRbot_py models a **3-link serial manipulator** actuated by **4 cables** routed through a parallel frame structure. The redundant cable system (4 cables for 3 DOF) enables robust tension control with no slack in any cable. The simulation integrates the full rigid-body dynamics via an ODE solver and tracks Cartesian trajectories using a PID-based controller.

**Robot Parameters (default):**

| Parameter | Value |
|-----------|-------|
| Link masses | m₁=6 kg, m₂=4 kg, m₃=1 kg |
| Link lengths | l₁=0.5 m, l₂=0.5 m, l₃=0.3 m |
| Pulley spacing | La=2.0 m, Lb=2.0 m |
| Winch radius | r=0.25 m |
| Gravity | g=9.8 m/s² |

---

## Features

- **Inverse Kinematics** – Closed-form IK solver for the 3-link planar arm
- **Jacobian Computation** – Geometric Jacobian and its time derivative for all 4 cable attachment points
- **Mass Matrix** – Full configuration-dependent 3×3 inertia matrix (Lagrangian formulation)
- **Dynamics (ODE)** – Coriolis, centrifugal, and gravity forces; integrated with `scipy.integrate.solve_ivp`
- **Cable Force Distribution** – Maps cable tensions (4×1) to joint torques (3×1) via a direction matrix
- **Trajectory Planning** – Cubic polynomial trajectory from initial to target end-effector position
- **Control Law** – PD task-space controller (Kp=750, Kv≈54.8) with gravity compensation, null-space optimization for cable tensions, and saturation enforcement (T ∈ [0, 500] N)

---

## Project Structure

```
RRbot_py/
├── 0Rrobot2024.py          # Main robot initialisation & IVP setup
├── 0var.py                 # Parameter display utility (debugging)
├── RRode.py                # ODE system – robot dynamics
├── invKIN.py               # Inverse kinematics solver
├── jak.py                  # Jacobian matrix computation
├── cables.py               # Cable routing and force distribution
├── controle_perp.py        # Trajectory planning & control loop
├── Mass_matrix.py          # Mass/inertia matrix computation
├── TETA.py                 # Cable direction angle utility
├── configr.py              # Robot configuration helper
├── passByref.py            # Pass-by-reference demonstration
│
├── FULL_inegration_test.py # Master test runner
├── invKIN_test_1.py        # Inverse kinematics unit tests
├── cables_test_1.py        # Cable system unit tests
├── controle_perp_test_.py  # Control algorithm unit tests
├── Mass_matrix_test_.py    # Mass matrix unit tests
├── RRode_test_.py          # ODE system unit tests
├── RRode_computeForces_test_.py  # Force computation unit tests
├── TETA_test_.py           # Angle calculation unit tests
├── configr_test_.py        # Configuration helper unit tests
├── jak_test_.py            # Jacobian unit tests
│
├── version_0/              # Legacy v0 implementation
├── v_0/                    # Alternative legacy version
├── old/                    # Archived files
├── LICENSE                 # GNU GPL v3
└── README.md
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Linear algebra, array operations |
| `scipy` | Pseudoinverse, null-space, `solve_ivp` ODE solver |

Python **3.8+** is recommended (tested on Python 3.12).

---

## Installation

```bash
# Clone the repository
git clone https://github.com/mouradism/RRbot_py.git
cd RRbot_py

# Install dependencies
pip install numpy scipy
```

---

## Usage

### Run the main simulation

```bash
python3 0Rrobot2024.py
```

This initialises the robot with the default parameters, computes the initial joint configuration via inverse kinematics, and sets up the IVP for simulation.

### Use as a library

```python
import numpy as np
from invKIN import invKIN
from RRode import RRode
from controle_perp import controle
from scipy.integrate import solve_ivp

# Define robot parameters
Args = {
    "m1": 6, "m2": 4, "m3": 1,
    "l1": 0.5, "l2": 0.5, "l3": 0.3,
    "La": 2.0, "Lb": 2.0, "r": 0.25,
    "g": 9.8,
    # ... other parameters (see 0Rrobot2024.py)
}

# Compute initial joint angles from end-effector position
Xi = np.array([-0.5, -0.4])   # Initial position [m]
q0 = invKIN(Xi, base_pos, Args)

# Integrate the robot dynamics
sol = solve_ivp(
    lambda t, x: RRode(t, x, Args),
    t_span=[0, 10],
    y0=initial_state,
    dense_output=True,
)
```

### Display robot parameters

```bash
python3 0var.py
```

---

## Running Tests

Run the full integration test suite:

```bash
python3 FULL_inegration_test.py
```

Run individual module tests:

```bash
python3 invKIN_test_1.py
python3 jak_test_.py
python3 cables_test_1.py
python3 Mass_matrix_test_.py
python3 controle_perp_test_.py
python3 RRode_test_.py
python3 RRode_computeForces_test_.py
python3 TETA_test_.py
python3 configr_test_.py
```

---

## Architecture

```
0Rrobot2024.py  (entry point)
       │
       ▼
   invKIN.py  ──── Closed-form IK
       │
       ├── cables.py  ──── jak.py ──── TETA.py
       │        │
       │    configr.py
       │
       ├── Mass_matrix.py
       │
       ├── controle_perp.py  (trajectory planning + PD control)
       │
       └── RRode.py  (ODE: dynamics integration)
```

**Data flow per control step:**
1. Measure current state **x** = [q, q̇]
2. Compute cable Jacobians & force distribution (`cables.py`, `jak.py`)
3. Plan desired Cartesian trajectory (`controle_perp.py`)
4. Calculate control torques and cable tensions (`controle_perp.py`)
5. Integrate equations of motion (`RRode.py` via `solve_ivp`)
6. Advance state to the next time step

---

## Key Algorithms

### Inverse Kinematics

Given end-effector position **X₂** and link-3 base **X₃**, joint angles are solved in closed form using the two-argument `arctan2` function:

```
c = (|X₂|² − l₁² − l₂²) / (2 l₁ l₂)
q₂ = arctan2(√(1 − c²), c)
q₁ = arctan2(y₂, x₂) − arctan2(l₂ sin q₂, l₁ + l₂ cos q₂)
q₃ = derived from X₃ − X₂ direction
```

### Control Law

Task-space PD control with gravity compensation:

```
ẍ_cmd = ẍ_ref + Kv · ė + Kp · e
τ     = M(q) · J⁺ · (ẍ_cmd − J̇ q̇) + h(q, q̇)
T     = (S J)⁺ τ  +  N λ   (null-space for tension feasibility)
```

where **S** is the cable direction matrix, **J⁺** is the pseudoinverse of the Jacobian, **N** is the null-space projector, and **λ** is chosen to keep all tensions non-negative.

---

## License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.