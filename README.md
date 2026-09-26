# Optimization Visualization Toolbox for Statistical Computing
*A Comparative Study of Gradient-Based Algorithms*

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-013243?logo=numpy&logoColor=white)](https://numpy.org/)

An interactive Python visualization toolbox built for exploring gradient-based numerical optimization algorithms in statistical computing. This project bridges pure mathematical optimization and statistical machine learning by visualizing loss landscapes, optimization trajectories, empirical loss dynamics, and dynamic decision boundaries.

All algorithms and mathematical gradients are implemented from scratch using **pure NumPy** vectorization.

---

## 📌 Project Overview

Optimization algorithms are commonly introduced through compact mathematical update formulas, but static equations fail to convey intuitive behaviors such as adaptive scaling, momentum acceleration, stochastic oscillation, or overshooting.

This toolbox bridges this pedagogical gap across two distinct modules:
1. **Module 1: Mathematical Optimization (Deterministic Surfaces)**: Visualizes optimizer navigation on 2D mathematical benchmark surfaces driven by explicit analytic gradients.
2. **Module 2: Statistical Learning Optimization (Data-Driven Estimation)**: Embeds optimizers into empirical risk minimization for binary classification, incorporating synthetic data generation, mini-batch updates, polynomial feature expansion, and $L_2$ (Ridge) regularization.

---

## 🚀 Key Modules & Architecture

### 🏔️ Module 1: Mathematical Optimization
Analyzes optimization mechanics on 2D objective landscapes with explicit analytical gradients:
* **Benchmark Objectives**:
  * **Quadratic Bowl**: Baseline convex surface ($f(x,y)=Ax^2+By^2+Cxy+Dx+Ey+F$) equipped with an automatic eigenvalue positive-definiteness convexity check.
  * **Rosenbrock Function**: A narrow, parabolic curved valley testing an optimizer's capability to traverse sharp turns without excessive oscillation.
  * **Rastrigin & Ackley Functions**: Highly multi-modal landscapes loaded with countless local minima and flat plateaus, testing escaping capabilities.
* **Interactive UI**:
  * **Sidebar**: Select objective function, optimizer, hyperparameters, initial point, iteration limit, and plot window boundaries.
  * **Contour Visualization**: Interactive 2D Plotly contour map showing the trajectory path, starting point, and global minimum.
  * **Diagnostic Panel**: Iteration loss curves (with optional log scale) and full numerical coordinate/gradient norm history tables.

---

### 🎯 Module 2: Statistical Learning Optimization
Applies optimization to empirical loss minimization via logistic regression:
* **Synthetic Data Generation (`utils.py`)**:
  * Supports **Linear Gaussian**, **XOR**, and **2 Moons** datasets.
  * Configurable class separation, Gaussian dispersion noise, and label flipping.
  * Automated training/test split ($70\% / 30\%$) and training-set Z-score feature standardization.
* **Polynomial Feature Mapping**:
  * Transforms raw 2D input $[x_1, x_2]$ up to Degree $d \in [1, 5]$ ($\Phi(x) \in \mathbb{R}^p$, where $p = \frac{(d+1)(d+2)}{2}$).
  * Degree 2 expansion: $\Phi_2(x) = [1, x_1, x_2, x_1^2, x_2^2, x_1x_2]$.
  * Evaluates non-linear decision boundaries on a virtual grid at $P(Y=1 \mid X) = 0.5$.
* **Loss Function & Regularization**:
  * Numerically stable cross-entropy implementation via `logaddexp`:
    $$\ell_i(\beta) = \log(1 + \exp(z_i)) - y_i z_i, \quad z_i = \Phi(x_i)^T \beta$$
  * **Ridge Regularization ($L_2$ Penalty)** applied strictly to non-intercept weights:
    $$L_\lambda(\beta) = \frac{1}{n} \sum_{i=1}^n \ell_i(\beta) + \frac{\lambda}{2} \Vert{}\beta_{-0}\Vert{}_2^2$$
    $$\nabla L_\lambda(\beta) = \frac{1}{n} \Phi^T (p - y) + \lambda \beta_{-0}$$
* **Diagnostics**:
  * Train Loss vs. Test Loss curves tracking the generalization gap.
  * Dynamic decision boundary slider by epoch, test classification accuracy, and learned polynomial coefficient inspection.

---

## ⚙️ Implemented Optimizers

All optimizers are implemented from scratch in `optimizers.py`:

| Optimizer | Formulation / Core Mechanics |
| :--- | :--- |
| **Gradient Descent (GD) / Mini-batch GD** | Decayed learning rate: $\alpha_t = \alpha_0 t^{-\gamma}$, $x_{t+1} = x_t - \alpha_t \nabla f(x_t)$ |
| **SGD with Momentum (SGDM)** | Velocity vector tracking direction: $v_t = \rho v_{t-1} - \alpha_t \nabla f(x_t)$, $x_{t+1} = x_t + v_t$ |
| **RMSProp** | Exponentially weighted moving average of squared gradients: $s_t = \rho s_{t-1} + (1-\rho) g_t^2$, $x_{t+1} = x_t - \frac{\eta}{\sqrt{s_t} + \epsilon} g_t$ |
| **Adam** | First and second moment estimation with bias corrections: $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$, $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$, $\hat{m}_t = \frac{m_t}{1-\beta_1^t}$, $\hat{v}_t = \frac{v_t}{1-\beta_2^t}$, $x_{t+1} = x_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$ |

*In Module 2, all optimizers execute via a unified `mini_batch_optimizer` pipeline supporting batch sizes from $1$ (pure stochastic) up to $N$ (full-batch).*

---

## 📂 Repository Structure

```plaintext
Optimization_Visualization_Toolbox/
├── app.py              # Streamlit frontend, Plotly visualization layout, and UI callbacks
├── objective.py        # Objective functions, analytic gradients (Mathematical & Logistic/Ridge Loss)
├── optimizers.py       # Core optimization engine (GD, SGDM, RMSProp, Adam, mini-batch updates)
├── utils.py            # Dataset generators (Gaussian, XOR, Moons), feature mapping, Z-score scaling
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
