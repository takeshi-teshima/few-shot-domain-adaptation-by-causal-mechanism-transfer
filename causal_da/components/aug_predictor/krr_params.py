import numpy as np
from scipy import stats

DEFAULT_HYPERPARAM_CANDIDATES = {
    # Regularization coeff.
    "alpha": [2.**i for i in range(-10, 10 + 1, 1)],
    # "alpha": [2.**i for i in range(-5, 5 + 1, 1)],  # Lessen the number because this is bottleneck
    # "alpha": [2.**i for i in np.linspace(-3, 0, 3)],
    # "alpha": [2.**i for i in np.arange(-4, 0)],

    # Kernel bandwidth. [None] resorts to heuristic choice (e.g., median heuristic)
    'gamma': [None],
    # 'gamma': np.linspace(0.01, 1.0, 5),
}

loguniform = stats.reciprocal
DEFAULT_HYPERPARAM_DISTRIBUTIONS = {'alpha': loguniform(1e-3, 1e0), 'gamma': loguniform(1e-5, 1)}
