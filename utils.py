import numpy as np


def map_feature(X, degree):
    X = np.asarray(X, dtype=float)
    degree = int(degree)

    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must be a two-dimensional array with exactly two feature columns.")
    if degree < 1:
        raise ValueError("degree must be at least 1.")

    x1 = X[:, 0]
    x2 = X[:, 1]
    columns = [np.ones(X.shape[0], dtype=float)]
    for current_degree in range(1, degree + 1):
        columns.append(x1**current_degree)
        columns.append(x2**current_degree)
        for x2_power in range(1, current_degree):
            x1_power = current_degree - x2_power
            columns.append((x1**x1_power) * (x2**x2_power))

    return np.column_stack(columns)


def polynomial_feature_names(degree):
    degree = int(degree)
    if degree < 1:
        raise ValueError("degree must be at least 1.")

    names = ["1"]
    for current_degree in range(1, degree + 1):
        names.append("x1" if current_degree == 1 else f"x1^{current_degree}")
        names.append("x2" if current_degree == 1 else f"x2^{current_degree}")
        for x2_power in range(1, current_degree):
            x1_power = current_degree - x2_power
            x1_name = "x1" if x1_power == 1 else f"x1^{x1_power}"
            x2_name = "x2" if x2_power == 1 else f"x2^{x2_power}"
            names.append(f"{x1_name}*{x2_name}")

    return tuple(names)


def zscore_fit(X_features):
    X_features = np.asarray(X_features, dtype=float)
    if X_features.ndim != 2:
        raise ValueError("X_features must be a two-dimensional array.")

    mean = np.mean(X_features, axis=0)
    scale = np.std(X_features, axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return mean, scale


def zscore_transform(X_features, mean, scale):
    X_features = np.asarray(X_features, dtype=float)
    mean = np.asarray(mean, dtype=float)
    scale = np.asarray(scale, dtype=float)

    if X_features.ndim != 2:
        raise ValueError("X_features must be a two-dimensional array.")
    if X_features.shape[1] != mean.shape[0] or mean.shape != scale.shape:
        raise ValueError("mean and scale must match the feature dimension.")

    return (X_features - mean) / scale


def train_test_split_arrays(X, y, test_size=0.3, random_state=None, shuffle=True):
    # Split arrays into random train and test subsets.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    test_size = float(test_size)

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of observations.")
    if X.shape[0] < 2:
        raise ValueError("At least two observations are required for a train/test split.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    rng = np.random.default_rng(random_state)
    indices = np.arange(X.shape[0])
    if shuffle:
        rng.shuffle(indices)

    n_test = int(round(X.shape[0] * test_size))
    n_test = min(max(n_test, 1), X.shape[0] - 1)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def make_module2_data(
    n_samples=1000,
    test_size=0.3,
    mode="linear",
    degree=1,
    class_sep=2.0,
    noise=0.65,
    label_noise=0.0,
    random_state=42,
):
    """Generate Module 2 binary-classification data.

    Raw 2D features are standardized with training-set Z-score statistics,
    then expanded to polynomial features for logistic regression.
    """
    n_samples = int(n_samples)
    degree = int(degree)
    if n_samples < 4:
        raise ValueError("n_samples must be at least 4.")
    if not 1 <= degree <= 5:
        raise ValueError("degree must be between 1 and 5.")
    if class_sep <= 0:
        raise ValueError("class_sep must be positive.")
    if noise <= 0:
        raise ValueError("noise must be positive.")
    if not 0.0 <= label_noise < 0.5:
        raise ValueError("label_noise must be in [0, 0.5).")

    mode = str(mode).lower()
    rng = np.random.default_rng(random_state)
    if mode == "linear":
        X_raw, y = _make_linear_gaussian_data(n_samples, class_sep, noise, rng)
    elif mode == "xor":
        X_raw, y = _make_xor_data(n_samples, class_sep, noise, rng)
    elif mode in ("moons", "make_moons", "make moons"):
        mode = "moons"
        X_raw, y = _make_moons_data(n_samples, class_sep, noise, rng)
    else:
        raise ValueError("mode must be 'linear', 'xor', or 'moons'.")

    if label_noise > 0:
        flip = rng.random(n_samples) < label_noise
        y[flip] = 1.0 - y[flip]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split_arrays(
        X_raw, y, test_size=test_size, random_state=rng
    )
    feature_mean, feature_scale = zscore_fit(X_train_raw)
    X_train_std = zscore_transform(X_train_raw, feature_mean, feature_scale)
    X_test_std = zscore_transform(X_test_raw, feature_mean, feature_scale)
    X_std = zscore_transform(X_raw, feature_mean, feature_scale)
    X = map_feature(X_std, degree)
    X_train = map_feature(X_train_std, degree)
    X_test = map_feature(X_test_std, degree)
    feature_names = polynomial_feature_names(degree)

    return {
        "X": X,
        "X_raw": X_raw,
        "X_std": X_std,
        "y": y.reshape(-1, 1),
        "X_train": X_train,
        "X_train_raw": X_train_raw,
        "X_train_std": X_train_std,
        "y_train": y_train,
        "X_test": X_test,
        "X_test_raw": X_test_raw,
        "X_test_std": X_test_std,
        "y_test": y_test,
        "beta_init": np.zeros(X.shape[1], dtype=float),
        "feature_names": feature_names,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "degree": degree,
        "mode": mode,
    }


def _make_linear_gaussian_data(n_samples, class_sep, noise, rng):
    class_counts = _balanced_counts(n_samples, 2)
    means = np.array(
        [
            [-class_sep, -class_sep],
            [class_sep, class_sep],
        ],
        dtype=float,
    )

    X_parts = []
    y_parts = []
    for label, count in enumerate(class_counts):
        X_parts.append(rng.normal(loc=means[label], scale=noise, size=(count, 2)))
        y_parts.append(np.full((count, 1), label, dtype=float))

    return np.vstack(X_parts), np.vstack(y_parts)


def _make_xor_data(n_samples, class_sep, noise, rng):
    cluster_counts = _balanced_counts(n_samples, 4)
    means = np.array(
        [
            [-class_sep, -class_sep],
            [class_sep, class_sep],
            [-class_sep, class_sep],
            [class_sep, -class_sep],
        ],
        dtype=float,
    )
    labels = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)

    X_parts = []
    y_parts = []
    for cluster, count in enumerate(cluster_counts):
        X_parts.append(rng.normal(loc=means[cluster], scale=noise, size=(count, 2)))
        y_parts.append(np.full((count, 1), labels[cluster], dtype=float))

    return np.vstack(X_parts), np.vstack(y_parts)


def _make_moons_data(n_samples, class_sep, noise, rng):
    moon_counts = _balanced_counts(n_samples, 2)
    theta_a = rng.uniform(0.0, np.pi, moon_counts[0])
    theta_b = rng.uniform(0.0, np.pi, moon_counts[1])

    moon_a = np.column_stack((np.cos(theta_a), np.sin(theta_a)))
    moon_b = np.column_stack((1.0 - np.cos(theta_b), 0.5 - np.sin(theta_b)))
    X_raw = class_sep * np.vstack((moon_a, moon_b))
    X_raw += rng.normal(scale=noise, size=X_raw.shape)
    y = np.vstack(
        (
            np.zeros((moon_counts[0], 1), dtype=float),
            np.ones((moon_counts[1], 1), dtype=float),
        )
    )
    return X_raw, y


def _balanced_counts(total, groups):
    # Distribute total into groups as evenly as possible, returning an array of counts.
    base = total // groups
    counts = np.full(groups, base, dtype=int)
    counts[: total % groups] += 1
    return counts
