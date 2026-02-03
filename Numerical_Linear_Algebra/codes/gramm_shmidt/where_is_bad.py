#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Base operator: 2D Laplacian (Dirichlet), dense for simplicity
# -------------------------
def laplacian_2d_dirichlet(m: int) -> np.ndarray:
    """
    2D Laplacian on an m x m interior grid, 5-point stencil, Dirichlet BC.
    Returns dense matrix A of size n x n with n = m^2.
    """
    T = np.zeros((m, m), dtype=np.float64)
    np.fill_diagonal(T, 2.0)
    np.fill_diagonal(T[1:], -1.0)
    np.fill_diagonal(T[:, 1:], -1.0)

    I = np.eye(m, dtype=np.float64)
    return np.kron(I, T) + np.kron(T, I)


# -------------------------
# Krylov matrix: n x p
# -------------------------
def krylov_matrix(A: np.ndarray, r: np.ndarray, p: int) -> np.ndarray:
    """
    Build Krylov matrix K = [r, A r, A^2 r, ..., A^{p-1} r] with p columns.
    """
    n = A.shape[0]
    if r.shape != (n,):
        raise ValueError(f"r must have shape ({n},), got {r.shape}")
    if p < 1:
        raise ValueError("p must be >= 1")

    K = np.empty((n, p), dtype=np.float64)
    v = r.astype(np.float64, copy=True)
    K[:, 0] = v
    for j in range(1, p):
        v = A @ v
        v = v / np.linalg.norm(v)
        K[:, j] = v 
    return K


# -------------------------
# QR via Gram–Schmidt
# -------------------------
def cgs_qr_projector(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Classical Gram–Schmidt in projector form:
        v <- (I - Q Q^T) a_j
    Works for rectangular A (n x p), returns Q (n x p), R (p x p).
    """
    n, p = A.shape
    Q = np.zeros((n, p), dtype=np.float64)
    R = np.zeros((p, p), dtype=np.float64)

    for j in range(p):
        a = A[:, j].astype(np.float64, copy=False)

        if j == 0:
            v = a.copy()
        else:
            Qprev = Q[:, :j]
            v = a - Qprev @ (Qprev.T @ a)
            R[:j, j] = Qprev.T @ a

        R[j, j] = np.linalg.norm(v)
        if R[j, j] == 0.0:
            raise np.linalg.LinAlgError(f"Breakdown in CGS at column {j}.")
        Q[:, j] = v / R[j, j]

    return Q, R


def mgs_qr(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Modified Gram–Schmidt QR (MGS)."""
    n, p = A.shape
    V = A.astype(np.float64, copy=True)
    Q = np.zeros((n, p), dtype=np.float64)
    R = np.zeros((p, p), dtype=np.float64)

    for i in range(p):
        R[i, i] = np.linalg.norm(V[:, i])
        if R[i, i] == 0.0:
            raise np.linalg.LinAlgError(f"Breakdown in MGS at column {i}.")
        Q[:, i] = V[:, i] / R[i, i]
        for j in range(i + 1, p):
            R[i, j] = np.dot(Q[:, i], V[:, j])
            V[:, j] -= R[i, j] * Q[:, i]

    return Q, R


# -------------------------
# QR via Householder (rectangular)
# -------------------------
def householder_qr(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Dense Householder QR for A (n x p), assumes n >= p.
    Returns Q (n x p) with orthonormal columns, R (p x p) upper-triangular.
    """
    A = A.astype(np.float64, copy=True)
    n, p = A.shape
    if n < p:
        raise ValueError("Householder QR here assumes n >= p.")

    R = A.copy()
    vs = []

    for k in range(p):
        x = R[k:, k]
        normx = np.linalg.norm(x)
        if normx == 0.0:
            vs.append(None)
            continue

        sign = 1.0 if x[0] >= 0.0 else -1.0
        v = x.copy()
        v[0] += sign * normx
        vnorm = np.linalg.norm(v)
        if vnorm == 0.0:
            vs.append(None)
            continue
        v /= vnorm

        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])
        vs.append(v)

    Qfull = np.eye(n, dtype=np.float64)
    for k, v in enumerate(vs):
        if v is None:
            continue
        Qfull[k:, :] -= 2.0 * np.outer(v, v @ Qfull[k:, :])

    Rpp = np.triu(R[:p, :p])
    Qnp = Qfull[:, :p]
    return Qnp, Rpp


# -------------------------
# QR via Givens rotations (rectangular)
# -------------------------
def givens_qr(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Dense Givens QR for A (n x p), assumes n >= p.
    Returns Q (n x p) with orthonormal columns, R (p x p) upper-triangular.
    """
    A = A.astype(np.float64, copy=True)
    n, p = A.shape
    if n < p:
        raise ValueError("Givens QR here assumes n >= p.")

    R = A.copy()
    Q_left = np.eye(n, dtype=np.float64)  # accumulate left rotations; Q = Q_left^T

    for j in range(p):
        for i in range(n - 1, j, -1):
            a = R[i - 1, j]
            b = R[i, j]
            if b == 0.0:
                continue

            r = np.hypot(a, b)
            if r == 0.0:
                continue
            c = a / r
            s = b / r

            # Rotate rows (i-1,i) of R
            row_im1 = R[i - 1, :].copy()
            row_i = R[i, :].copy()
            R[i - 1, :] = c * row_im1 + s * row_i
            R[i, :] = -s * row_im1 + c * row_i

            # Accumulate into Q_left
            qrow_im1 = Q_left[i - 1, :].copy()
            qrow_i = Q_left[i, :].copy()
            Q_left[i - 1, :] = c * qrow_im1 + s * qrow_i
            Q_left[i, :] = -s * qrow_im1 + c * qrow_i

    Rpp = np.triu(R[:p, :p])
    Qnp = Q_left.T[:, :p]
    return Qnp, Rpp


# -------------------------
# Diagnostics
# -------------------------
def orthogonality_error_prefix(Q: np.ndarray, norm_kind: str = "fro") -> np.ndarray:
    """
    For k = 1..p: e_k = || Q_k^T Q_k - I ||, where Q_k = Q[:, :k].
    norm_kind: "fro" or "2".
    """
    _, p = Q.shape
    G = Q.T @ Q
    errs = np.empty(p, dtype=np.float64)
    for k in range(1, p + 1):
        M = G[:k, :k] - np.eye(k, dtype=np.float64)
        errs[k - 1] = np.linalg.norm(M, ord=norm_kind)
    return errs


def cond2_svd(A: np.ndarray) -> float:
    """
    2-norm condition number via singular values: kappa_2 = smax / smin.
    Works for rectangular matrices too.
    """
    s = np.linalg.svd(A, compute_uv=False)
    smin = s[-1]
    if smin == 0.0:
        return np.inf
    return s[0] / smin


def cond_prefix_matrix(A: np.ndarray, norm_kind = 2) -> np.ndarray:
    """
    For k = 1..p: c_k = cond2( A[:, :k] ) via SVD.
    Always valid for rectangular prefixes.
    """
    _, p = A.shape
    conds = np.empty(p, dtype=np.float64)
    for k in range(1, p + 1):
        Ak = A[:, :k]
        conds[k - 1] = cond2_svd(Ak)
    return conds


def run_case(K: np.ndarray, title: str, norm_kind: str = "fro") -> None:
    n, p = K.shape

    # Krylov matrices can be extremely ill-conditioned; cond may overflow.
    try:
        cond_all = np.linalg.cond(K)
        print(f"{title}: shape=({n},{p}), cond_2(K) ≈ {cond_all:.3e}")
    except Exception as e:
        print(f"{title}: shape=({n},{p}), cond_2(K) not computed ({e})")

    # --- QR methods ---
    Q_cgs, _ = cgs_qr_projector(K)
    Q_mgs, _ = mgs_qr(K)
    Q_hh, _ = householder_qr(K)
    Q_gv, _ = givens_qr(K)

    # --- Orthogonality errors ---
    err_cgs = orthogonality_error_prefix(Q_cgs, norm_kind=norm_kind)
    err_mgs = orthogonality_error_prefix(Q_mgs, norm_kind=norm_kind)
    err_hh = orthogonality_error_prefix(Q_hh, norm_kind=norm_kind)
    err_gv = orthogonality_error_prefix(Q_gv, norm_kind=norm_kind)

    k = np.arange(1, p + 1)

    plt.figure()
    plt.semilogy(k, err_cgs, label="CGS (projector)")
    plt.semilogy(k, err_mgs, label="MGS")
    plt.semilogy(k, err_hh, label="Householder")
    plt.semilogy(k, err_gv, label="Givens")
    plt.xlabel("k (number of columns)")
    plt.ylabel(f"||Q_k^T Q_k - I|| ({norm_kind}-norm)")
    plt.title(title + " — orthogonality loss")
    plt.grid(True, which="both")
    plt.legend()

    # --- Condition number of prefixes of original K ---
    conds = cond_prefix_matrix(K, norm_kind="2")

    plt.figure()
    plt.semilogy(k, conds)
    plt.xlabel("k (number of columns)")
    plt.ylabel("cond_2( K[:, :k] )")
    plt.title(title + " — condition number of Krylov prefix")
    plt.grid(True, which="both")


def main():
    # -----------------------------
    # Hardcoded parameters (edit here)
    # -----------------------------
    M_LAPLACE = 10        # n = m^2
    P = 80                # Krylov columns: K is n x P
    SEED = 0
    ORTH_NORM = "fro"     # "fro" or "2" for ||Q_k^T Q_k - I||
    NORMALIZE_R = True
    # -----------------------------

    A = laplacian_2d_dirichlet(M_LAPLACE)
    n = A.shape[0]
    if P > n:
        print(f"Warning: P={P} > n={n}. Krylov subspace dimension cannot exceed n (rank issues likely).")

    rng = np.random.default_rng(SEED)
    r = rng.standard_normal(n).astype(np.float64)
    if NORMALIZE_R:
        r /= np.linalg.norm(r)

    K = krylov_matrix(A, r, P)

    run_case(
        K,
        title=f"Krylov K=[r,Ar,...,A^{{p-1}}r] for 2D Laplacian (m={M_LAPLACE}, n={n}, p={P})",
        norm_kind=ORTH_NORM,
    )

    plt.show()


if __name__ == "__main__":
    main()

