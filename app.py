from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from objective import (
    ackley,
    ackley_grad,
    logistic,
    logistic_grad,
    quadratic,
    quadratic_grad,
    rastrigin,
    rastrigin_grad,
    rosenbrock,
    rosenbrock_grad,
)
from optimizers import ADAM, RMSProp, SGDM, mini_batch_optimizer, standard_GD
from utils import make_module2_data, map_feature, zscore_transform


MODULES = ("Module 1: Mathematical Optimization", "Module 2: Machine Learning Optimization")
QUAD_DEFAULT = {"A": 5.0, "B": 5.0, "C": -6.0, "D": 0.0, "E": 0.0, "F": 0.0}
OBJECTIVES = ("Quadratic bowl", "Rosenbrock", "Rastrigin", "Ackley")
OPTIMIZERS = ("Gradient Descent", "SGD + Momentum", "RMSProp", "Adam")
MODULE2_OBJECTIVE = "Logistic loss"
MODULE2_OPTIMIZERS = ("Mini-batch GD", "SGD + Momentum", "RMSProp", "Adam")
MODULE2_OPTIMIZER = MODULE2_OPTIMIZERS[0]
MODULE2_DATASETS = ("Linear Gaussian", "XOR", "Make Moons")
MODULE2_DATASET_MODES = {
    "Linear Gaussian": "linear",
    "XOR": "xor",
    "Make Moons": "moons",
}
DEFAULT_ITERATIONS = 300
DEFAULT_MODULE2_ITERATIONS = 300
DEFAULT_GAMMA = 0.75
DEFAULT_MOMENTUM = 0.9
DEFAULT_RMSPROP_DECAY = 0.9
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.999
DEFAULT_EPSILON = 1e-8
DEFAULT_MODULE2_LEARNING_RATE = 0.45
DEFAULT_MODULE2_L2 = 0.01
DEFAULT_MODULE2_TEST_SIZE = 0.3
DEFAULT_MODULE2_RANDOM_STATE = 42
MODULE2_BATCH_SIZE_OPTIONS = (1, 8, 16, 32, 64, 128, 256)
DEFAULT_MODULE2_LEARNING_RATES = {
    "Mini-batch GD": 0.45,
    "SGD + Momentum": 0.05,
    "RMSProp": 0.03,
    "Adam": 0.05,
}
DEFAULT_LEARNING_RATES = {
    "Quadratic bowl": {
        "Gradient Descent": 0.08,
        "SGD + Momentum": 0.08,
        "RMSProp": 0.01,
        "Adam": 0.01,
    },
    "Rosenbrock": {
        "Gradient Descent": 0.005,
        "SGD + Momentum": 0.003,
        "RMSProp": 0.01,
        "Adam": 0.01,
    },
    "Rastrigin": {
        "Gradient Descent": 0.01,
        "SGD + Momentum": 0.005,
        "RMSProp": 0.02,
        "Adam": 0.02,
    },
    "Ackley": {
        "Gradient Descent": 0.03,
        "SGD + Momentum": 0.02,
        "RMSProp": 0.03,
        "Adam": 0.03,
    },
}
CONTOUR_PLOT_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
    "doubleClick": "reset",
}
DIAGNOSTIC_PLOT_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
    "doubleClick": "reset",
}


@dataclass
class Module1Result:
    x_final: np.ndarray
    trajectory: np.ndarray
    losses: np.ndarray
    grad_norms: np.ndarray


@dataclass
class Module2Result:
    optimizer_name: str
    beta_final: np.ndarray
    trajectory: np.ndarray
    train_losses: np.ndarray
    test_losses: np.ndarray
    train_objectives: np.ndarray
    grad_norms: np.ndarray
    train_accuracy: float
    test_accuracy: float
    data: dict
    l2: float
    batch_size: int
    updates_per_epoch: int


def quadratic_is_convex(params):
    hessian = np.array([[2 * params["A"], params["C"]], [params["C"], 2 * params["B"]]])
    return bool(np.all(np.linalg.eigvalsh(hessian) > 0))


def quadratic_minimum(params):
    if not quadratic_is_convex(params):
        return None

    hessian = np.array([[2 * params["A"], params["C"]], [params["C"], 2 * params["B"]]])
    linear = np.array([params["D"], params["E"]])
    point = -np.linalg.solve(hessian, linear)
    return float(point[0]), float(point[1])


def get_objective(name, quad_params=None):
    # Return value/gradient callables and plotting defaults for one 2D objective.
    if name == "Quadratic bowl":
        params = QUAD_DEFAULT.copy()
        if quad_params:
            params.update({key: float(value) for key, value in quad_params.items()})

        return {
            "value": lambda x, y: quadratic(x, y, **params),
            "grad": lambda p: quadratic_grad(p[0], p[1], **params),
            "start": (-2.0, 2.0),
            "x_range": (-3.0, 3.0),
            "y_range": (-3.0, 3.0),
            "minimum": quadratic_minimum(params),
        }

    if name == "Rosenbrock":
        return {
            "value": rosenbrock,
            "grad": lambda p: rosenbrock_grad(p[0], p[1]),
            "start": (-1.2, 1.0),
            "x_range": (-2.0, 2.0),
            "y_range": (-1.0, 3.0),
            "minimum": (1.0, 1.0),
        }

    if name == "Rastrigin":
        return {
            "value": rastrigin,
            "grad": lambda p: rastrigin_grad(p[0], p[1]),
            "start": (3.5, 3.0),
            "x_range": (-5.12, 5.12),
            "y_range": (-5.12, 5.12),
            "minimum": (0.0, 0.0),
        }

    if name == "Ackley":
        return {
            "value": ackley,
            "grad": lambda p: ackley_grad(p[0], p[1]),
            "start": (2.5, 2.0),
            "x_range": (-5.0, 5.0),
            "y_range": (-5.0, 5.0),
            "minimum": (0.0, 0.0),
        }

    raise ValueError(f"Unknown objective: {name}")


def default_learning_rate(objective_name, optimizer_name):
    # Defaults are conservative enough to show movement without immediate blow-up.
    return DEFAULT_LEARNING_RATES[objective_name][optimizer_name]


def run_module1(
    objective_name,
    optimizer_name,
    x_init,
    iterations,
    learning_rate,
    gamma=DEFAULT_GAMMA,
    momentum=DEFAULT_MOMENTUM,
    rmsprop_decay=DEFAULT_RMSPROP_DECAY,
    beta1=DEFAULT_BETA1,
    beta2=DEFAULT_BETA2,
    epsilon=DEFAULT_EPSILON,
    quad_params=None,
):
    # Pure helper for tests: run one Module 1 experiment without Streamlit.
    obj = get_objective(objective_name, quad_params)
    x0 = np.asarray(x_init, dtype=float)

    if x0.shape != (2,):
        raise ValueError("Module 1 expects a two-dimensional initial point.")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    if optimizer_name == "Gradient Descent":
        x_final, trajectory = standard_GD(obj["grad"], x0, learning_rate, iterations, gamma=gamma)
    elif optimizer_name == "SGD + Momentum":
        x_final, trajectory = SGDM(
            obj["grad"], x0, learning_rate, np.zeros_like(x0), iterations, gamma=gamma, rho=momentum
        )
    elif optimizer_name == "RMSProp":
        x_final, trajectory = RMSProp(
            obj["grad"], x0, iterations, rho=rmsprop_decay, eta=learning_rate, epsilon=epsilon
        )
    elif optimizer_name == "Adam":
        x_final, trajectory = ADAM(obj["grad"], x0, (beta1, beta2), iterations, eta=learning_rate, epsilon=epsilon)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    trajectory = np.asarray(trajectory, dtype=float)
    losses, grad_norms = summarize_path(obj, trajectory)
    return Module1Result(np.asarray(x_final), trajectory, losses, grad_norms)


def run_module2(
    optimizer_name=MODULE2_OPTIMIZER,
    iterations=DEFAULT_MODULE2_ITERATIONS,
    learning_rate=DEFAULT_MODULE2_LEARNING_RATE,
    batch_size=32,
    dataset_mode="linear",
    polynomial_degree=1,
    gamma=DEFAULT_GAMMA,
    momentum=DEFAULT_MOMENTUM,
    rmsprop_decay=DEFAULT_RMSPROP_DECAY,
    beta1=DEFAULT_BETA1,
    beta2=DEFAULT_BETA2,
    epsilon=DEFAULT_EPSILON,
    l2=DEFAULT_MODULE2_L2,
    n_samples=1000,
    test_size=DEFAULT_MODULE2_TEST_SIZE,
    class_sep=2.0,
    noise=0.65,
    label_noise=0.0,
    random_state=DEFAULT_MODULE2_RANDOM_STATE,
):
    # Pure helper for tests: run Module 2 without Streamlit.
    iterations = int(iterations)
    batch_size = int(batch_size)
    polynomial_degree = int(polynomial_degree)
    learning_rate = float(learning_rate)
    gamma = float(gamma)
    momentum = float(momentum)
    rmsprop_decay = float(rmsprop_decay)
    beta1 = float(beta1)
    beta2 = float(beta2)
    epsilon = float(epsilon)
    l2 = float(l2)

    if optimizer_name not in MODULE2_OPTIMIZERS:
        raise ValueError(f"Unknown Module 2 optimizer: {optimizer_name}")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not 1 <= polynomial_degree <= 5:
        raise ValueError("polynomial_degree must be between 1 and 5")
    if not 0.5 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0.5 and 1.0")
    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must be in [0, 1).")
    if not 0.0 <= rmsprop_decay < 1.0:
        raise ValueError("rmsprop_decay must be in [0, 1).")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("Adam beta values must be in [0, 1).")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if l2 < 0:
        raise ValueError("l2 must be non-negative")

    data = make_module2_data(
        n_samples=n_samples,
        test_size=test_size,
        mode=dataset_mode,
        degree=polynomial_degree,
        class_sep=class_sep,
        noise=noise,
        label_noise=label_noise,
        random_state=random_state,
    )
    if batch_size > data["X_train"].shape[0]:
        raise ValueError("batch_size must be no larger than the training set size.")

    batch_grad = lambda beta, X_batch, y_batch: logistic_grad(
        beta, X_batch, y_batch, l2=l2, penalize_intercept=False
    )

    beta_final, trajectory = mini_batch_optimizer(
        batch_grad,
        data["beta_init"],
        optimizer_name,
        iterations,
        data["X_train"],
        data["y_train"],
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
        momentum=momentum,
        rmsprop_decay=rmsprop_decay,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        random_state=random_state,
    )

    trajectory = np.asarray(trajectory, dtype=float)
    train_losses, test_losses, train_objectives, grad_norms = summarize_module2_path(data, trajectory, l2)
    updates_per_epoch = int(np.ceil(data["X_train"].shape[0] / batch_size))
    return Module2Result(
        optimizer_name,
        np.asarray(beta_final, dtype=float),
        trajectory,
        train_losses,
        test_losses,
        train_objectives,
        grad_norms,
        classification_accuracy(beta_final, data["X_train"], data["y_train"]),
        classification_accuracy(beta_final, data["X_test"], data["y_test"]),
        data,
        l2,
        batch_size,
        updates_per_epoch,
    )


def summarize_path(obj, trajectory):
    """Compute diagnostics after the optimizer path is known."""
    losses = []
    grad_norms = []
    # Suppress overflow warnings since some optimizers may diverge with bad hyperparameters.
    with np.errstate(over="ignore", invalid="ignore"):
        for point in trajectory:
            losses.append(obj["value"](point[0], point[1]))
            grad_norms.append(np.linalg.norm(obj["grad"](point)))
    return np.asarray(losses, dtype=float), np.asarray(grad_norms, dtype=float)


def summarize_module2_path(data, trajectory, l2):
    train_losses = []
    test_losses = []
    train_objectives = []
    grad_norms = []

    with np.errstate(over="ignore", invalid="ignore"):
        for beta in trajectory:
            train_losses.append(logistic(beta, data["X_train"], data["y_train"]))
            test_losses.append(logistic(beta, data["X_test"], data["y_test"]))
            train_objectives.append(
                logistic(beta, data["X_train"], data["y_train"], l2=l2, penalize_intercept=False)
            )
            grad_norms.append(
                np.linalg.norm(
                    logistic_grad(beta, data["X_train"], data["y_train"], l2=l2, penalize_intercept=False)
                )
            )

    return (
        np.asarray(train_losses, dtype=float),
        np.asarray(test_losses, dtype=float),
        np.asarray(train_objectives, dtype=float),
        np.asarray(grad_norms, dtype=float),
    )


def classification_accuracy(beta, X, y):
    logits = np.asarray(X, dtype=float) @ np.asarray(beta, dtype=float)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500.0, 500.0)))
    y_true = np.asarray(y, dtype=float).reshape(-1)
    return float(np.mean((probs >= 0.5) == y_true))


def contour_plot(obj, result, x_range, y_range, grid_size):
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    xx, yy = np.meshgrid(x, y)

    with np.errstate(over="ignore", invalid="ignore"):
        z = np.asarray(obj["value"](xx, yy), dtype=float)

    # Cap extreme contour values so the path remains readable on Rosenbrock.
    finite_z = z[np.isfinite(z)]
    if finite_z.size:
        z = np.minimum(z, np.nanpercentile(finite_z, 96))

    path = result.trajectory[np.all(np.isfinite(result.trajectory), axis=1)]
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=x,
            y=y,
            z=z,
            colorscale="Viridis",
            colorbar={"title": "loss", "x": 1.02, "thickness": 14, "len": 0.82},
            contours={"showlabels": False},
            line_smoothing=0.75,
            name="Objective",
        )
    )

    add_contour_grid(fig, x_range, y_range, grid_size)

    if obj["minimum"] is not None:
        fig.add_trace(marker_trace(obj["minimum"], "Known minimum", "#facc15", "star", 15))

    if len(path):
        fig.add_trace(
            go.Scatter(
                x=path[:, 0],
                y=path[:, 1],
                mode="lines+markers",
                line={"color": "#111827", "width": 2},
                marker={
                    "size": 6,
                    "color": np.arange(len(path)),
                    "colorscale": "Greys_r",
                    "showscale": True,
                    "colorbar": {"title": "iter", "x": 1.14, "thickness": 14, "len": 0.82},
                },
                name="Path",
            )
        )
        fig.add_trace(marker_trace(path[0], "Start", "#00BA10", "circle", 12, "#111827"))
        # fig.add_trace(marker_trace(path[-1], "Final", "#ef4444", "x", 13))

    fig.update_layout(
        height=760,
        margin={"l": 60, "r": 115, "t": 45, "b": 55},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        template="plotly_white",
        xaxis_title="x",
        yaxis_title="y",
        dragmode="pan",
        uirevision=f"{x_range}:{y_range}",
    )
    fig.update_xaxes(
        range=list(x_range),
        showgrid=False,
        zeroline=False,
        constrain="domain",
        constraintoward="center",
    )
    fig.update_yaxes(
        range=list(y_range),
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
        constraintoward="center",
    )
    return fig


def add_contour_grid(fig, x_range, y_range, grid_size):
    # Reuse the grid-size slider for the visible overlay, not only contour resolution.
    target_lines = max(4, min(16, int(grid_size / 20)))
    x_grid = nice_grid_values(x_range, target_lines)
    y_grid = nice_grid_values(y_range, target_lines)
    grid_color = "rgba(148, 163, 184, 0.42)"

    if y_grid:
        x_values = []
        y_values = []
        for value in y_grid:
            x_values.extend([x_range[0], x_range[1], None])
            y_values.extend([value, value, None])
        fig.add_trace(grid_trace(x_values, y_values, grid_color))

    if x_grid:
        x_values = []
        y_values = []
        for value in x_grid:
            x_values.extend([value, value, None])
            y_values.extend([y_range[0], y_range[1], None])
        fig.add_trace(grid_trace(x_values, y_values, grid_color))


def nice_grid_values(value_range, target_lines=7):
    start, end = float(value_range[0]), float(value_range[1])
    span = end - start
    if span <= 0:
        return []

    rough_step = span / max(target_lines, 1)
    exponent = np.floor(np.log10(rough_step))
    fraction = rough_step / (10**exponent)
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    step = nice_fraction * (10**exponent)

    first = np.ceil(start / step) * step
    values = np.arange(first, end + step * 0.5, step)
    return [float(value) for value in values if start <= value <= end]


def grid_trace(x_values, y_values, color):
    return go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines",
        line={"color": color, "width": 1},
        hoverinfo="skip",
        showlegend=False,
        name="Grid",
    )


# Utility to plot start/final/known-minimum points; edge_color is used only when requested.
def marker_trace(point, name, color, symbol, size, edge_color=None):
    marker = {"size": size, "color": color, "symbol": symbol}
    if edge_color:
        marker["line"] = {"color": edge_color, "width": 2}

    return go.Scatter(
        x=[point[0]],
        y=[point[1]],
        mode="markers",
        marker=marker,
        name=name,
    )


def loss_plot(result, log_axis=False):
    fig = go.Figure(
        go.Scatter(
            x=np.arange(len(result.losses)),
            y=result.losses,
            mode="lines",
            line={"color": "#2563eb", "width": 2},
            name="Loss",
        )
    )
    fig.update_layout(
        height=380,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        template="plotly_white",
        xaxis_title="iteration",
        yaxis_title="loss",
        dragmode="zoom",
        uirevision=len(result.losses),
    )
    fig.update_xaxes(rangeslider={"visible": True, "thickness": 0.16})
    if log_axis:
        fig.update_yaxes(type="log")
    fig.update_yaxes(fixedrange=True)
    return fig


def module2_loss_plot(result, log_axis=False):
    epochs = np.arange(len(result.trajectory))
    axis_type = "log" if log_axis else "linear"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=result.train_losses,
            mode="lines",
            line={"color": "#2563eb", "width": 2},
            name="Train loss",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=result.test_losses,
            mode="lines",
            line={"color": "#dc2626", "width": 2},
            name="Test loss",
        )
    )
    if result.l2 > 0:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=result.train_objectives,
                mode="lines",
                line={"color": "#64748b", "width": 1.5, "dash": "dot"},
                name="Train objective + L2",
            )
        )

    fig.update_layout(
        height=430,
        margin={"l": 45, "r": 25, "t": 40, "b": 45},
        template="plotly_white",
        xaxis_title="epoch",
        yaxis_title="logistic loss",
        title={
            "text": (
                f"Train vs. Test Loss - {result.optimizer_name} "
                f"(batch={result.batch_size}, {result.updates_per_epoch} updates/epoch, lambda={result.l2:.4g})"
            ),
            "x": 0.02,
            "xanchor": "left",
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        dragmode="zoom",
        uirevision=f"{len(result.trajectory)}:{axis_type}",
    )
    fig.update_xaxes(rangeslider={"visible": True, "thickness": 0.13})
    fig.update_yaxes(type=axis_type)
    return fig


def module2_scatter_boundary_plot(result, boundary_iteration):
    data = result.data
    boundary_iteration = int(np.clip(boundary_iteration, 0, len(result.trajectory) - 1))
    X_train = data["X_train_raw"]
    X_test = data["X_test_raw"]
    y_train = data["y_train"].reshape(-1)
    y_test = data["y_test"].reshape(-1)
    all_features = np.vstack((X_train, X_test))
    x_range, y_range = feature_ranges(all_features)

    fig = go.Figure()
    add_class_scatter(fig, X_train, y_train, "Train", "circle", 0.72)
    add_class_scatter(fig, X_test, y_test, "Test", "diamond", 0.95)

    history_steps = np.unique(np.linspace(0, boundary_iteration, min(6, boundary_iteration + 1), dtype=int))
    for step in history_steps[:-1]:
        add_decision_boundary_trace(
            fig,
            result.trajectory[step],
            data,
            x_range,
            y_range,
            f"Boundary epoch {step}",
            "rgba(100, 116, 139, 0.35)",
            1.4,
            "dot",
            showlegend=False,
        )

    selected_beta = result.trajectory[boundary_iteration]
    add_decision_boundary_trace(
        fig,
        selected_beta,
        data,
        x_range,
        y_range,
        f"Boundary epoch {boundary_iteration}",
        "#111827",
        3,
        "solid",
        showlegend=True,
    )

    fig.update_layout(
        height=560,
        margin={"l": 55, "r": 30, "t": 58, "b": 50},
        template="plotly_white",
        title={
            "text": (
                f"{data['mode'].title()} Data and Degree {data['degree']} Decision Boundary "
                f"({len(selected_beta)} coefficients)"
            ),
            "x": 0.02,
            "xanchor": "left",
        },
        xaxis_title="feature 1",
        yaxis_title="feature 2",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        dragmode="pan",
        uirevision=f"{boundary_iteration}:{result.l2}",
    )
    grid_style = {"showgrid": True, "gridcolor": "rgba(148, 163, 184, 0.35)", "gridwidth": 1}
    fig.update_xaxes(range=list(x_range), zeroline=False, **grid_style)
    fig.update_yaxes(range=list(y_range), zeroline=False, scaleanchor="x", scaleratio=1, **grid_style)
    return fig


def add_class_scatter(fig, features, labels, split_name, symbol, opacity):
    colors = {0.0: "#2563eb", 1.0: "#dc2626"}
    names = {0.0: "class 0", 1.0: "class 1"}
    for label in (0.0, 1.0):
        mask = labels == label
        if not np.any(mask):
            continue
        fig.add_trace(
            go.Scatter(
                x=features[mask, 0],
                y=features[mask, 1],
                mode="markers",
                marker={
                    "size": 8,
                    "color": colors[label],
                    "symbol": symbol,
                    "opacity": opacity,
                    "line": {"color": "white", "width": 0.7},
                },
                name=f"{split_name} {names[label]}",
            )
        )


def add_decision_boundary_trace(fig, beta, data, x_range, y_range, name, color, width, dash, showlegend):
    logits = decision_boundary_logits(beta, data, x_range, y_range)
    if logits is None:
        return

    x_values, y_values, z_values = logits
    fig.add_trace(
        go.Contour(
            x=x_values,
            y=y_values,
            z=z_values,
            contours={"start": 0.0, "end": 0.0, "size": 1.0, "coloring": "lines"},
            colorscale=[[0.0, color], [1.0, color]],
            autocolorscale=False,
            line={"color": color, "width": width, "dash": dash},
            name=name,
            hoverinfo="skip",
            showlegend=showlegend,
            showscale=False,
        )
    )


def decision_boundary_logits(beta, data, x_range, y_range, grid_size=180):
    beta = np.asarray(beta, dtype=float)
    if not np.any(np.abs(beta) > 1e-12):
        return None

    x_values = np.linspace(x_range[0], x_range[1], grid_size)
    y_values = np.linspace(y_range[0], y_range[1], grid_size)
    xx, yy = np.meshgrid(x_values, y_values)
    raw_grid = np.column_stack((xx.ravel(), yy.ravel()))
    design_grid = module2_design_from_raw(raw_grid, data)
    z_values = (design_grid @ beta).reshape(xx.shape)
    if np.nanmin(z_values) > 0.0 or np.nanmax(z_values) < 0.0:
        return None
    return x_values, y_values, z_values


def module2_design_from_raw(raw_features, data):
    standardized = zscore_transform(raw_features, data["feature_mean"], data["feature_scale"])
    return map_feature(standardized, data["degree"])


def feature_ranges(features):
    x_min, y_min = np.min(features, axis=0)
    x_max, y_max = np.max(features, axis=0)
    x_pad = max((x_max - x_min) * 0.12, 0.5)
    y_pad = max((y_max - y_min) * 0.12, 0.5)
    return (float(x_min - x_pad), float(x_max + x_pad)), (float(y_min - y_pad), float(y_max + y_pad))


# Helper to create Streamlit sidebar inputs with consistent formatting and keys.
def sidebar_float(label, value, min_value, max_value, step, key=None):
    decimal_places = max(0, len(f"{step:.10f}".rstrip("0").split(".")[-1]))
    return float(
        st.sidebar.number_input(
            label,
            value=value,
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=key,
            format=f"%.{decimal_places}f",
        )
    )


def sidebar_controls(objective_name, optimizer_name):
    # Collect all UI inputs in one place so main() stays small.
    key = f"{objective_name}_{optimizer_name}".lower().replace(" ", "_").replace("+", "plus")
    quad_params = QUAD_DEFAULT.copy()

    if objective_name == "Quadratic bowl":
        with st.sidebar.expander("Quadratic parameters"):
            for name, value in quad_params.items():
                quad_params[name] = float(st.number_input(name, value=value, step=0.5))
        if not quadratic_is_convex(quad_params):
            st.sidebar.warning("The quadratic Hessian is not positive definite.")

    obj = get_objective(objective_name, quad_params)
    controls = {
        "quad_params": quad_params,
        "iterations": int(st.sidebar.slider("Iterations", 1, 1000, DEFAULT_ITERATIONS)),
        "learning_rate": sidebar_float(
            "Learning rate", default_learning_rate(objective_name, optimizer_name), 1e-6, 1.0, 0.001, f"lr_{key}"
        ),
        "gamma": DEFAULT_GAMMA,
        "momentum": DEFAULT_MOMENTUM,
        "rmsprop_decay": DEFAULT_RMSPROP_DECAY,
        "beta1": DEFAULT_BETA1,
        "beta2": DEFAULT_BETA2,
    }

    if optimizer_name in ("Gradient Descent", "SGD + Momentum"):
        controls["gamma"] = sidebar_float("Learning-rate decay", DEFAULT_GAMMA, 0.5, 1.0, 0.001, f"gamma_{key}")
    if optimizer_name == "SGD + Momentum":
        controls["momentum"] = sidebar_float("Momentum", DEFAULT_MOMENTUM, 0.0, 0.999, 0.01, "momentum")
    if optimizer_name == "RMSProp":
        controls["rmsprop_decay"] = sidebar_float("RMSProp decay", DEFAULT_RMSPROP_DECAY, 0.0, 0.999, 0.01, "rmsprop_decay")
    if optimizer_name == "Adam":
        controls["beta1"] = sidebar_float("Beta 1", DEFAULT_BETA1, 0.0, 0.999, 0.01, "adam_beta1")
        controls["beta2"] = sidebar_float("Beta 2", DEFAULT_BETA2, 0.0, 0.9999, 0.0001, "adam_beta2")

    st.sidebar.subheader("Initial point")
    obj_key = objective_name.lower().replace(" ", "_")
    controls["start_x"] = sidebar_float("Initial x", obj["start"][0], -10.0, 10.0, 0.1, f"start_x_{obj_key}")
    controls["start_y"] = sidebar_float("Initial y", obj["start"][1], -10.0, 10.0, 0.1, f"start_y_{obj_key}")

    st.sidebar.subheader("Plot window")
    controls["x_min"] = sidebar_float("x min", obj["x_range"][0], -20.0, 20.0, 0.5, f"x_min_{obj_key}")
    controls["x_max"] = sidebar_float("x max", obj["x_range"][1], -20.0, 20.0, 0.5, f"x_max_{obj_key}")
    controls["y_min"] = sidebar_float("y min", obj["y_range"][0], -20.0, 20.0, 0.5, f"y_min_{obj_key}")
    controls["y_max"] = sidebar_float("y max", obj["y_range"][1], -20.0, 20.0, 0.5, f"y_max_{obj_key}")
    controls["grid_size"] = int(st.sidebar.slider("Grid size", 60, 240, 160, step=20))
    controls["log_loss"] = st.sidebar.checkbox("Log loss axis", value=False)
    controls["show_table"] = st.sidebar.checkbox("Show trajectory table", value=False)
    return controls


def module2_sidebar_controls():
    st.sidebar.selectbox("Objective", (MODULE2_OBJECTIVE,), key="module2_objective")
    if st.session_state.get("module2_optimizer") not in MODULE2_OPTIMIZERS:
        st.session_state["module2_optimizer"] = MODULE2_OPTIMIZER
    optimizer_name = st.sidebar.selectbox("Optimizer", MODULE2_OPTIMIZERS, key="module2_optimizer")
    optimizer_key = optimizer_name.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")

    st.sidebar.subheader("Data")
    dataset_name = st.sidebar.selectbox("Dataset", MODULE2_DATASETS, key="module2_dataset")
    n_samples = int(st.sidebar.slider("Samples", 100, 3000, 1000, step=100))
    test_size = DEFAULT_MODULE2_TEST_SIZE
    n_test = min(max(int(round(n_samples * test_size)), 1), n_samples - 1)
    n_train = n_samples - n_test
    batch_size_options = tuple(size for size in MODULE2_BATCH_SIZE_OPTIONS if size <= n_train)
    if n_train not in batch_size_options:
        batch_size_options = (*batch_size_options, n_train)
    default_batch_index = batch_size_options.index(32) if 32 in batch_size_options else len(batch_size_options) - 1

    controls = {
        "optimizer_name": optimizer_name,
        "dataset_mode": MODULE2_DATASET_MODES[dataset_name],
        "n_samples": n_samples,
        "class_sep": sidebar_float("Class separation", 2.0, 0.2, 5.0, 0.1, "module2_class_sep"),
        "noise": sidebar_float("Gaussian noise", 0.65, 0.05, 3.0, 0.05, "module2_noise"),
        "label_noise": sidebar_float("Label noise", 0.0, 0.0, 0.45, 0.01, "module2_label_noise"),
        "polynomial_degree": int(st.sidebar.slider("Polynomial Degree", 1, 5, 1, step=1)),
    }

    st.sidebar.subheader("Optimizer")
    batch_size = int(
        st.sidebar.selectbox(
            "Batch size",
            batch_size_options,
            index=default_batch_index,
            format_func=lambda size: f"{size} (full batch)" if size == n_train else str(size),
        )
    )

    controls.update(
        {
            "iterations": int(st.sidebar.slider("Epochs", 10, 1000, DEFAULT_MODULE2_ITERATIONS, step=10)),
            "batch_size": batch_size,
            "learning_rate": sidebar_float(
                "Learning rate",
                DEFAULT_MODULE2_LEARNING_RATES[optimizer_name],
                1e-4,
                5.0,
                0.01,
                f"module2_learning_rate_{optimizer_key}",
            ),
            "gamma": DEFAULT_GAMMA,
            "momentum": DEFAULT_MOMENTUM,
            "rmsprop_decay": DEFAULT_RMSPROP_DECAY,
            "beta1": DEFAULT_BETA1,
            "beta2": DEFAULT_BETA2,
            "l2": sidebar_float("Ridge lambda", DEFAULT_MODULE2_L2, 0.0, 2.0, 0.001, "module2_l2"),
            "log_loss": st.sidebar.checkbox("Log loss axis", value=False, key="module2_log_loss"),
        }
    )
    updates_per_epoch = int(np.ceil(n_train / batch_size))
    controls["updates_per_epoch"] = updates_per_epoch

    if optimizer_name in ("Mini-batch GD", "SGD + Momentum"):
        controls["gamma"] = sidebar_float(
            "Learning-rate decay", DEFAULT_GAMMA, 0.5, 1.0, 0.001, f"module2_gamma_{optimizer_key}"
        )
    if optimizer_name == "SGD + Momentum":
        controls["momentum"] = sidebar_float(
            "Momentum", DEFAULT_MOMENTUM, 0.0, 0.999, 0.01, "module2_momentum"
        )
    if optimizer_name == "RMSProp":
        controls["rmsprop_decay"] = sidebar_float(
            "RMSProp decay", DEFAULT_RMSPROP_DECAY, 0.0, 0.999, 0.01, "module2_rmsprop_decay"
        )
    if optimizer_name == "Adam":
        controls["beta1"] = sidebar_float("Beta 1", DEFAULT_BETA1, 0.0, 0.999, 0.01, "module2_adam_beta1")
        controls["beta2"] = sidebar_float("Beta 2", DEFAULT_BETA2, 0.0, 0.9999, 0.0001, "module2_adam_beta2")

    st.sidebar.subheader("Decision boundary")
    controls["boundary_iteration"] = int(
        st.sidebar.number_input(
            "Boundary epoch",
            min_value=0,
            max_value=controls["iterations"],
            value=controls["iterations"],
            step=1,
        )
    )
    return controls


def show_metrics(result):
    finite_loss = result.losses[np.isfinite(result.losses)]
    finite_grad = result.grad_norms[np.isfinite(result.grad_norms)]
    cols = st.columns(4)
    cols[0].metric("Final x", f"{result.x_final[0]:.4g}")
    cols[1].metric("Final y", f"{result.x_final[1]:.4g}")
    cols[2].metric("Final loss", f"{finite_loss[-1]:.4g}" if finite_loss.size else "nan")
    cols[3].metric("Gradient norm", f"{finite_grad[-1]:.4g}" if finite_grad.size else "nan")


def show_module2_metrics(result, boundary_iteration):
    iteration = int(np.clip(boundary_iteration, 0, len(result.trajectory) - 1))
    beta = result.trajectory[iteration]
    train_loss = result.train_losses[iteration]
    test_loss = result.test_losses[iteration]
    gap = test_loss - train_loss
    test_accuracy = classification_accuracy(beta, result.data["X_test"], result.data["y_test"])
    grad_norm = result.grad_norms[iteration]

    st.caption(f"Metrics at boundary epoch {iteration}")
    cols = st.columns(5)
    cols[0].metric("Train loss", f"{train_loss:.4g}")
    cols[1].metric("Test loss", f"{test_loss:.4g}")
    cols[2].metric("Generalization gap", f"{gap:.4g}")
    cols[3].metric("Test accuracy", f"{test_accuracy:.1%}")
    cols[4].metric("Gradient norm", f"{grad_norm:.4g}" if np.isfinite(grad_norm) else "nan")


def render_module1():
    st.title("Optimization Module 1")
    objective_name = st.sidebar.selectbox("Objective", OBJECTIVES, key="module1_objective")
    optimizer_name = st.sidebar.selectbox("Optimizer", OPTIMIZERS, key="module1_optimizer")
    controls = sidebar_controls(objective_name, optimizer_name)

    if controls["x_min"] >= controls["x_max"] or controls["y_min"] >= controls["y_max"]:
        st.error("Plot bounds must satisfy min < max.")
        return

    result = run_module1(
        objective_name,
        optimizer_name,
        (controls["start_x"], controls["start_y"]),
        controls["iterations"],
        controls["learning_rate"],
        gamma=controls["gamma"],
        momentum=controls["momentum"],
        rmsprop_decay=controls["rmsprop_decay"],
        beta1=controls["beta1"],
        beta2=controls["beta2"],
        quad_params=controls["quad_params"],
    )
    obj = get_objective(objective_name, controls["quad_params"])

    show_metrics(result)
    if not np.all(np.isfinite(result.trajectory)) or not np.all(np.isfinite(result.losses)):
        st.warning("The optimizer produced non-finite values. Lower the learning rate or reduce momentum.")

    # Keep the contour visualization dominant; the loss plot is secondary context.
    left, right = st.columns([3.2, 1.0])
    with left:
        st.plotly_chart(
            contour_plot(
                obj,
                result,
                (controls["x_min"], controls["x_max"]),
                (controls["y_min"], controls["y_max"]),
                controls["grid_size"],
            ),
            use_container_width=True,
            config=CONTOUR_PLOT_CONFIG,
        )

    with right:
        st.plotly_chart(
            loss_plot(result, controls["log_loss"]),
            use_container_width=True,
            config=DIAGNOSTIC_PLOT_CONFIG,
        )
        if controls["show_table"]:
            st.dataframe(
                {
                    "iteration": np.arange(len(result.trajectory)),
                    "x": result.trajectory[:, 0],
                    "y": result.trajectory[:, 1],
                    "loss": result.losses,
                    "grad_norm": result.grad_norms,
                },
                use_container_width=True,
                height=300,
            )


def render_module2():
    st.title("Optimization Module 2")
    controls = module2_sidebar_controls()
    result = run_module2(
        optimizer_name=controls["optimizer_name"],
        iterations=controls["iterations"],
        learning_rate=controls["learning_rate"],
        batch_size=controls["batch_size"],
        dataset_mode=controls["dataset_mode"],
        polynomial_degree=controls["polynomial_degree"],
        gamma=controls["gamma"],
        momentum=controls["momentum"],
        rmsprop_decay=controls["rmsprop_decay"],
        beta1=controls["beta1"],
        beta2=controls["beta2"],
        l2=controls["l2"],
        n_samples=controls["n_samples"],
        class_sep=controls["class_sep"],
        noise=controls["noise"],
        label_noise=controls["label_noise"],
    )

    show_module2_metrics(result, controls["boundary_iteration"])
    if not np.all(np.isfinite(result.trajectory)) or not np.all(np.isfinite(result.train_losses)):
        st.warning("The optimizer produced non-finite values. Lower the learning rate or ridge lambda.")

    left, right = st.columns([1.35, 1.0])
    with left:
        st.plotly_chart(
            module2_scatter_boundary_plot(result, controls["boundary_iteration"]),
            use_container_width=True,
            config=CONTOUR_PLOT_CONFIG,
        )
    with right:
        st.plotly_chart(
            module2_loss_plot(result, controls["log_loss"]),
            use_container_width=True,
            config=DIAGNOSTIC_PLOT_CONFIG,
        )
        st.dataframe(
            {
                "coefficient": result.data["feature_names"],
                "value": result.trajectory[controls["boundary_iteration"]],
                "final_value": result.beta_final,
            },
            use_container_width=True,
            height=150,
        )


def main():
    st.set_page_config(page_title="Optimization Visualization Toolbox", layout="wide")
    module_name = st.sidebar.radio("Module", MODULES)

    if module_name == MODULES[0]:
        render_module1()
    else:
        render_module2()


if __name__ == "__main__":
    main()
