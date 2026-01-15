import numpy as np


def build_tau_hat(countries, shocks):
    n = len(countries)
    idx = {iso: i for i, iso in enumerate(countries)}
    tau = np.ones((n, n), dtype=float)

    for shock in shocks:
        stype = shock.get("type")
        rate = float(shock.get("rate", 0.0))
        target = shock.get("target")
        partner = shock.get("partner")

        if stype == "global_tariff":
            tau += (1 + rate - 1) * (np.ones((n, n)) - np.eye(n))
            continue

        if stype == "import_tariff":
            if target in idx:
                j = idx[target]
                for i in range(n):
                    if i != j:
                        tau[i, j] *= (1 + rate)
            continue

        if stype == "bilateral_tariff":
            if target in idx and partner in idx:
                i = idx[partner]
                j = idx[target]
                tau[i, j] *= (1 + rate)
            continue

        if stype == "rta":
            reduction = max(0.0, min(0.5, rate))
            if target in idx and partner in idx:
                i = idx[partner]
                j = idx[target]
                tau[i, j] *= (1 - reduction)
                tau[j, i] *= (1 - reduction)

    return tau


def solve_ge(flows, theta=5.0, tau_hat=None, max_iter=2000, tol=1e-6, damping=0.4):
    flows = np.asarray(flows, dtype=float)
    n = flows.shape[0]
    if tau_hat is None:
        tau_hat = np.ones((n, n), dtype=float)

    Y = flows.sum(axis=1)
    E = flows.sum(axis=0)
    Y_total = Y.sum()

    phi = np.power(np.maximum(tau_hat, 1e-12), 1 - theta)

    P = np.ones(n)
    Pi = np.ones(n)

    for it in range(max_iter):
        Pi_term = (E / max(Y_total, 1e-12)) * np.power(Pi, theta - 1)
        P_new = np.power(np.maximum(phi @ Pi_term, 1e-12), 1 / (1 - theta))

        P_term = (Y / max(Y_total, 1e-12)) * np.power(P, theta - 1)
        Pi_new = np.power(np.maximum(phi.T @ P_term, 1e-12), 1 / (1 - theta))

        diff = max(np.max(np.abs(P_new - P)), np.max(np.abs(Pi_new - Pi)))
        P = (1 - damping) * P + damping * P_new
        Pi = (1 - damping) * Pi + damping * Pi_new

        if diff < tol:
            break

    scale_i = np.power(P, theta - 1)
    scale_j = np.power(Pi, theta - 1)
    X_cf = (Y[:, None] * E[None, :] / max(Y_total, 1e-12)) * phi * (scale_i[:, None] * scale_j[None, :])

    welfare = 1 / np.maximum(P, 1e-12)

    return {
        "exports_base": Y,
        "imports_base": E,
        "exports_cf": X_cf.sum(axis=1),
        "imports_cf": X_cf.sum(axis=0),
        "flows_base": flows,
        "flows_cf": X_cf,
        "welfare": welfare,
        "converged": diff < tol,
        "iterations": it + 1,
        "max_diff": float(diff),
    }
