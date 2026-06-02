import numpy as np

MINI_BATCH_RANDOM_STATE = 42


def standard_GD(func_grad, x_init, alpha, n, gamma=0.75):
    x = x_init.copy()
    trajectory = [x.copy()]
    alpha0 = alpha
    for t in range(1,n+1):
        grad = func_grad(x)
        # Decay learning rate over time
        alpha = alpha0 * (t ** -gamma)
        x = x - alpha * grad
        trajectory.append(x.copy())
    return x, trajectory


def mini_batch_GD(
    func_grad,
    x_init,
    alpha,
    n,
    X,
    y,
    batch_size=32,
    gamma=0.75,
    random_state=MINI_BATCH_RANDOM_STATE,
):
    X = np.asarray(X)
    if X.ndim == 0:
        raise ValueError("X must contain at least one observation.")

    x = np.asarray(x_init, dtype=float).copy()
    n_samples = X.shape[0]
    y_array = np.asarray(y)
    batch_size = int(batch_size)

    if n_samples == 0:
        raise ValueError("X must contain at least one observation.")
    if y_array.shape[0] != n_samples:
        raise ValueError("X and y must contain the same number of observations.")
    if batch_size < 1 or batch_size > n_samples:
        raise ValueError("batch_size must be between 1 and the number of observations.")

    trajectory = [x.copy()]
    alpha0 = alpha
    rng = np.random.default_rng(random_state)
    order = rng.permutation(n_samples)
    cursor = 0

    for t in range(1, n + 1):
        if cursor + batch_size > n_samples:
            order = rng.permutation(n_samples)
            cursor = 0

        batch_idx = order[cursor : cursor + batch_size]
        cursor += batch_size
        X_batch = X[batch_idx]
        grad = func_grad(x, X_batch, y_array[batch_idx])

        if isinstance(grad, tuple):
            grad = grad[1]
        grad = np.asarray(grad, dtype=float)

        alpha = alpha0 * (t ** -gamma)
        x = x - alpha * grad
        trajectory.append(x.copy())

    return x, trajectory


def _validate_mini_batch_inputs(X, y, batch_size):
    X = np.asarray(X)
    if X.ndim == 0:
        raise ValueError("X must contain at least one observation.")

    y_array = np.asarray(y)
    n_samples = X.shape[0]
    batch_size = int(batch_size)

    if n_samples == 0:
        raise ValueError("X must contain at least one observation.")
    if y_array.shape[0] != n_samples:
        raise ValueError("X and y must contain the same number of observations.")
    if batch_size < 1 or batch_size > n_samples:
        raise ValueError("batch_size must be between 1 and the number of observations.")

    return X, y_array, n_samples, batch_size


def _permuted_minibatches(n_samples, batch_size, rng):
    order = rng.permutation(n_samples)
    for start in range(0, n_samples, batch_size):
        yield order[start : start + batch_size]


def mini_batch_optimizer(
    func_grad,
    x_init,
    optimizer_name,
    iterations,
    X,
    y,
    batch_size=32,
    learning_rate=1e-3,
    gamma=0.75,
    momentum=0.9,
    rmsprop_decay=0.9,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    random_state=MINI_BATCH_RANDOM_STATE,
):
    X, y_array, n_samples, batch_size = _validate_mini_batch_inputs(X, y, batch_size)
    x = np.asarray(x_init, dtype=float).copy()
    iterations = int(iterations)
    learning_rate = float(learning_rate)
    gamma = float(gamma)
    momentum = float(momentum)
    rmsprop_decay = float(rmsprop_decay)
    beta1 = float(beta1)
    beta2 = float(beta2)
    epsilon = float(epsilon)

    if iterations < 1:
        raise ValueError("iterations must be at least 1.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    trajectory = [x.copy()]
    rng = np.random.default_rng(random_state)
    velocity = np.zeros_like(x)
    rms_average = np.zeros_like(x)
    adam_m = np.zeros_like(x)
    adam_v = np.zeros_like(x)
    update_step = 0

    for _ in range(iterations):
        for batch_idx in _permuted_minibatches(n_samples, batch_size, rng):
            update_step += 1
            grad = func_grad(x, X[batch_idx], y_array[batch_idx])
            if isinstance(grad, tuple):
                grad = grad[1]
            grad = np.asarray(grad, dtype=float)

            if optimizer_name == "Mini-batch GD":
                alpha = learning_rate * (update_step ** -gamma)
                x = x - alpha * grad
            elif optimizer_name == "SGD + Momentum":
                alpha = learning_rate * (update_step ** -gamma)
                velocity = momentum * velocity - alpha * grad
                x = x + velocity
            elif optimizer_name == "RMSProp":
                rms_average = rmsprop_decay * rms_average + (1 - rmsprop_decay) * grad**2
                alpha = learning_rate / (np.sqrt(rms_average + epsilon))
                x = x - alpha * grad
            elif optimizer_name == "Adam":
                adam_m = beta1 * adam_m + (1 - beta1) * grad
                adam_v = beta2 * adam_v + (1 - beta2) * grad**2
                m_hat = adam_m / (1 - beta1**update_step)
                v_hat = adam_v / (1 - beta2**update_step)
                x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            else:
                raise ValueError(f"Unknown mini-batch optimizer: {optimizer_name}")

        trajectory.append(x.copy())

    return x, trajectory


def SGDM(func_grad, x_init, alpha, v0, n, gamma=0.75, rho=0.9):
    x = x_init.copy()
    trajectory = [x.copy()]
    alpha0 = alpha
    v = v0.copy()
    for t in range(1,n+1):
        grad = func_grad(x)
        alpha = alpha0 * (t ** -gamma)
        v = rho * v - alpha * grad
        x = x + v
        trajectory.append(x.copy())
    return x, trajectory


def RMSProp(func_grad, x_init, n, rho=0.9, eta=1e-3, epsilon=1e-8):
    x = x_init.copy()
    trajectory = [x.copy()]
    EWMA = 0
    for t in range(1,n+1):
        grad = func_grad(x)
        EWMA = rho * EWMA + (1-rho) * grad**2
        alpha = eta / (np.sqrt(EWMA + epsilon))
        x = x - alpha * grad
        trajectory.append(x.copy())
    return x, trajectory


def ADAM(func_grad, x_init, beta, n, eta=1e-3, epsilon=1e-8):
    x = x_init.copy()
    trajectory = [x.copy()]
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    for t in range(1,n+1):
        grad = func_grad(x)
        m = beta[0] * m + (1 - beta[0]) * grad
        v = beta[1] * v + (1 - beta[1]) * grad ** 2
        m_hat = m / (1 - beta[0]**t)
        v_hat = v / (1 - beta[1]**t)
        alpha = eta / (np.sqrt(v_hat) + epsilon)
        x = x - alpha * m_hat
        trajectory.append(x.copy())
    return x, trajectory
    
