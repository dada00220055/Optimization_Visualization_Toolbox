import numpy as np


def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def _prepare_logistic_inputs(beta, X, y):
    beta = np.asarray(beta, dtype=float)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if beta.ndim != 1:
        raise ValueError("beta must be a one-dimensional parameter vector.")
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional design matrix.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of observations.")
    if X.shape[1] != beta.shape[0]:
        raise ValueError("beta length must match the number of columns in X.")
    if X.shape[0] == 0:
        raise ValueError("X and y must contain at least one observation.")

    labels = set(np.unique(y))
    if labels <= {0.0, 1.0}:
        y01 = y
    else:
        raise ValueError("y must use binary labels encoded as 0/1.")

    return beta, X, y01


def _l2_penalty_vector(beta, penalize_intercept):
    penalty_beta = beta.copy()
    if not penalize_intercept and penalty_beta.size:
        penalty_beta[0] = 0.0
    return penalty_beta


def _validate_l2(l2):
    l2 = float(l2)
    if l2 < 0:
        raise ValueError("l2 must be non-negative.")
    return l2


def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(x, y):
    df_dx = -2 * (1 - x) - 400 * x * (y - x**2)
    df_dy = 200 * (y - x**2)
    return np.array([df_dx, df_dy])


def rastrigin(x, y):
    return 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))


def rastrigin_grad(x, y):
    df_dx = 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
    df_dy = 2 * y + 20 * np.pi * np.sin(2 * np.pi * y)
    return np.array([df_dx, df_dy])


def ackley(x, y):
    radius = np.sqrt(0.5 * (x**2 + y**2))
    cosine = 0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
    return -20 * np.exp(-0.2 * radius) - np.exp(cosine) + np.e + 20


def ackley_grad(x, y):
    radius = np.sqrt(0.5 * (x**2 + y**2))
    if radius == 0:
        radial_grad = np.array([0.0, 0.0])
    else:
        radial_grad = (2 * np.exp(-0.2 * radius) / radius) * np.array([x, y])

    cosine = 0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
    periodic_grad = np.pi * np.exp(cosine) * np.array(
        [np.sin(2 * np.pi * x), np.sin(2 * np.pi * y)]
    )
    return radial_grad + periodic_grad


def quadratic(x, y, A=5, B=5, C=-6, D=0, E=0, F=0):
    return A*(x**2) + B*(y**2) + C*x*y + D*x + E*y + F

def quadratic_grad(x, y, A=5, B=5, C=-6, D=0, E=0, F=0):
    df_dx = 2*A*x + C*y + D
    df_dy = 2*B*y + C*x + E
    return np.array([df_dx, df_dy])


def logistic(beta, X, y, l2=0.0, penalize_intercept=False):
    l2 = _validate_l2(l2)
    beta, X, y01 = _prepare_logistic_inputs(beta, X, y)
    logits = X @ beta
    losses = np.logaddexp(0.0, logits) - y01 * logits

    if l2 == 0.0:
        return float(np.mean(losses))

    penalty_beta = _l2_penalty_vector(beta, penalize_intercept)
    return float(np.mean(losses) + 0.5 * l2 * float(penalty_beta @ penalty_beta))


def logistic_grad(beta, X, y, l2=0.0, penalize_intercept=False):
    l2 = _validate_l2(l2)
    beta, X, y01 = _prepare_logistic_inputs(beta, X, y)
    probs = _sigmoid(X @ beta)
    grad = (X.T @ (probs - y01)) / X.shape[0]

    if l2 == 0.0:
        return grad

    penalty_beta = _l2_penalty_vector(beta, penalize_intercept)
    return grad + l2 * penalty_beta


def ridge_logistic(beta, X, y, l2=1.0, penalize_intercept=False):
    return logistic(beta, X, y, l2=l2, penalize_intercept=penalize_intercept)


def ridge_logistic_grad(beta, X, y, l2=1.0, penalize_intercept=False):
    return logistic_grad(beta, X, y, l2=l2, penalize_intercept=penalize_intercept)
