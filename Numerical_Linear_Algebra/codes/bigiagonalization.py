import numpy as np
from typing import Iterable, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting dependency may not be installed
    plt = None


def householder_vector(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Build Householder vector v and scaling beta so that
    (I - beta v v*) x = [+-||x||, 0, ..., 0]^T for real or complex x.
    """
    x = x.astype(complex)
    sigma = np.linalg.norm(x)
    if sigma == 0.0:
        return np.zeros_like(x), 0.0

    sign = x[0] / abs(x[0]) if x[0] != 0 else 1.0
    v = x.copy()
    v[0] += sign * sigma
    beta = 2.0 / np.vdot(v, v)
    return v, beta


def golub_kahan_bidiagonalize(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reduce A to upper bidiagonal form B using Golub-Kahan.

    Returns U, B, V such that A ~ U @ B @ V.H with U and V unitary
    and B upper bidiagonal.
    """
    A = np.array(A, dtype=complex, copy=True)
    m, n = A.shape
    U = np.eye(m, dtype=complex)
    V = np.eye(n, dtype=complex)

    for k in range(min(m, n)):
        # Left reflector to introduce zeros below the diagonal in column k
        v, beta = householder_vector(A[k:, k])
        if beta != 0.0:
            A[k:, :] -= beta * np.outer(v, v.conj() @ A[k:, :])
            U[:, k:] -= beta * np.outer(U[:, k:] @ v, v.conj())

        if k >= n - 1:
            continue

        # Right reflector to introduce zeros right of the super-diagonal in row k
        v, beta = householder_vector(A[k, k + 1 :])
        if beta != 0.0:
            A[:, k + 1 :] -= beta * np.outer(A[:, k + 1 :] @ v, v.conj())
            V[:, k + 1 :] -= beta * np.outer(V[:, k + 1 :] @ v, v.conj())

    # Zero-out tiny roundoff elements to make the bidiagonal structure explicit
    A[np.abs(A) < 1e-12] = 0.0
    return U, A, V


def svd_via_golub_kahan(
    A: np.ndarray, full_matrices: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the SVD of A using Golub-Kahan bidiagonalization followed by
    a (small) dense SVD of the bidiagonal matrix.

    Returns U, s, Vh so that A is approximately U @ np.diag(s) @ Vh.
    """
    U_bi, B, V_bi = golub_kahan_bidiagonalize(A)
    U_b, s, Vt_b = np.linalg.svd(B, full_matrices=full_matrices)

    U = U_bi @ U_b
    Vt = Vt_b @ V_bi.conj().T
    return U, s, Vt


def svd_via_jacobi(
    A: np.ndarray,
    max_sweeps: int = 80,
    tol: float = 1e-14,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Two-sided (one-sided column orthogonalization) complex Jacobi SVD.

    Performs cyclic sweeps across all column pairs until off-diagonal
    products fall below tol or max_sweeps is reached.
    """
    A_work = np.array(A, dtype=complex, copy=True)
    m, n = A_work.shape
    V = np.eye(n, dtype=complex)

    for _ in range(max_sweeps):
        max_corr = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                ap = A_work[:, p]
                aq = A_work[:, q]
                alpha = np.vdot(ap, ap)
                beta = np.vdot(aq, aq)
                gamma = np.vdot(ap, aq)

                if alpha == 0 and beta == 0:
                    continue

                denom = np.sqrt(abs(alpha) * abs(beta))
                corr = 0.0 if denom == 0 else abs(gamma) / denom
                max_corr = max(max_corr, corr)
                if corr <= tol:
                    continue

                G = np.array([[alpha, gamma], [np.conj(gamma), beta]], dtype=complex)
                eigvals, eigvecs = np.linalg.eigh(G)

                # Update columns and right singular vectors
                A_work[:, [p, q]] = A_work[:, [p, q]] @ eigvecs
                V[:, [p, q]] = V[:, [p, q]] @ eigvecs
        if max_corr <= tol:
            break

    s = np.linalg.norm(A_work, axis=0)
    U = np.zeros((m, n), dtype=complex)
    nonzero = s > 0
    U[:, nonzero] = A_work[:, nonzero] / s[nonzero]

    order = np.argsort(-s.real)
    s = s[order]
    U = U[:, order]
    V = V[:, order]
    Vh = V.conj().T
    return U, s, Vh


def reconstruction_error(A: np.ndarray, method: str = "golub-kahan", **method_kwargs) -> float:
    """Relative Frobenius reconstruction error using a chosen SVD method."""
    if method == "golub-kahan":
        U, s, Vh = svd_via_golub_kahan(A, **method_kwargs)
    elif method == "jacobi":
        U, s, Vh = svd_via_jacobi(A, **method_kwargs)
    else:
        raise ValueError(f"Unknown method '{method}'.")

    reconstruction = U @ np.diag(s) @ Vh
    return np.linalg.norm(A - reconstruction) / np.linalg.norm(A)


def orthogonality_error(X: np.ndarray) -> float:
    """Relative Frobenius error of orthonormality for columns of X."""
    I = np.eye(X.shape[1], dtype=complex)
    return np.linalg.norm(X.conj().T @ X - I) / np.linalg.norm(I)


def random_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random unitary matrix via QR."""
    Z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, R = np.linalg.qr(Z)
    # Fix phases to make Q unitary (diagonal of R may have phases)
    diag = np.diag(R)
    phases = np.ones_like(diag)
    nonzero = np.abs(diag) > 0
    phases[nonzero] = diag[nonzero] / np.abs(diag[nonzero])
    Q = Q * phases.conj()
    return Q


def make_ill_conditioned_matrix(
    n: int, condition: float = 1e12, seed: int | None = None
) -> np.ndarray:
    """Construct a random unitary * diag(s) * unitary^H matrix with large condition number."""
    rng = np.random.default_rng(seed)
    s = np.geomspace(1.0, 1.0 / condition, num=n)
    U = random_unitary(n, rng)
    V = random_unitary(n, rng)
    return U @ np.diag(s) @ V.conj().T


def method_metrics(A: np.ndarray, method: str, **kwargs) -> Tuple[float, float]:
    """Return (reconstruction_error, orthogonality_error) for a given method."""
    if method == "golub-kahan":
        U, s, Vh = svd_via_golub_kahan(A, **kwargs)
    elif method == "jacobi":
        U, s, Vh = svd_via_jacobi(A, **kwargs)
    else:
        raise ValueError(f"Unknown method '{method}'.")

    reconstruction = U @ np.diag(s) @ Vh
    recon_err = np.linalg.norm(A - reconstruction) / np.linalg.norm(A)
    ortho_err = max(orthogonality_error(U), orthogonality_error(Vh.conj().T))
    return recon_err, ortho_err


def plot_error_over_sizes(
    sizes: Iterable[int],
    trials: int = 3,
    seed: int = 0,
    method: str = "golub-kahan",
    method_kwargs: dict | None = None,
    save_path: str | None = "codes/reconstruction_error.png",
    label: str | None = None,
    show: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute and plot reconstruction error for random complex matrices of given sizes.
    Returns (sizes_array, errors_array).
    """
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting but is not installed.")

    sizes = np.array(list(sizes), dtype=int)
    errors = []
    mat_rng = np.random.default_rng(seed)

    for n in sizes:
        trial_errs = []
        for _ in range(trials):
            A = mat_rng.standard_normal((n, n)) + 1j * mat_rng.standard_normal((n, n))
            trial_errs.append(reconstruction_error(A, method=method, **(method_kwargs or {})))
        errors.append(np.mean(trial_errs))

    errors = np.array(errors)

    plt.figure(figsize=(6.4, 4))
    method_label = label or method
    plt.plot(sizes, errors, marker="o", label=method_label)
    plt.xlabel("Matrix size (n x n)")
    plt.ylabel("Relative reconstruction error ||A - U diag(s) V^H||_F / ||A||_F")
    plt.title(f"SVD reconstruction error via {method_label}")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved reconstruction error plot to {save_path}")
    elif show:  # pragma: no cover - interactive use
        plt.show()
    plt.close()

    return sizes, errors


def plot_method_comparison(
    sizes: Iterable[int],
    trials: int = 1,
    seed: int = 0,
    golub_kahan_kwargs: dict | None = None,
    jacobi_kwargs: dict | None = None,
    save_path: str = "codes/reconstruction_comparison.png",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Plot reconstruction error for Golub-Kahan vs Jacobi on random complex matrices.
    """
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting but is not installed.")

    sizes = np.array(list(sizes), dtype=int)
    gk_err = []
    jc_err = []
    mat_rng = np.random.default_rng(seed)

    for n in sizes:
        A = mat_rng.standard_normal((n, n)) + 1j * mat_rng.standard_normal((n, n))
        gk_err.append(reconstruction_error(A, method="golub-kahan", **(golub_kahan_kwargs or {})))
        jc_err.append(reconstruction_error(A, method="jacobi", **(jacobi_kwargs or {})))

    gk_err = np.array(gk_err)
    jc_err = np.array(jc_err)

    plt.figure(figsize=(6.8, 4.2))
    plt.plot(sizes, gk_err, marker="o", label="Golub-Kahan")
    plt.plot(sizes, jc_err, marker="s", label="Jacobi")
    plt.xlabel("Matrix size (n x n)")
    plt.ylabel("Relative reconstruction error ||A - U diag(s) V^H||_F / ||A||_F")
    plt.title("SVD reconstruction error vs matrix size")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved comparison plot to {save_path}")
    plt.close()

    return sizes, gk_err, jc_err


def compare_on_ill_conditioned(
    sizes: Iterable[int],
    condition: float = 1e12,
    trials: int = 1,
    seed: int = 0,
    golub_kahan_kwargs: dict | None = None,
    jacobi_kwargs: dict | None = None,
):
    """
    Collect reconstruction and orthogonality errors on ill-conditioned matrices.
    """
    sizes = np.array(list(sizes), dtype=int)
    gk_recon = []
    jc_recon = []
    gk_ortho = []
    jc_ortho = []

    for idx, n in enumerate(sizes):
        gk_r = []
        jc_r = []
        gk_o = []
        jc_o = []
        for t in range(trials):
            mat_seed = seed + 97 * (idx + 1) + t
            A = make_ill_conditioned_matrix(n, condition=condition, seed=mat_seed)
            r, o = method_metrics(A, "golub-kahan", **(golub_kahan_kwargs or {}))
            gk_r.append(r)
            gk_o.append(o)
            r, o = method_metrics(A, "jacobi", **(jacobi_kwargs or {}))
            jc_r.append(r)
            jc_o.append(o)
        gk_recon.append(np.mean(gk_r))
        jc_recon.append(np.mean(jc_r))
        gk_ortho.append(np.mean(gk_o))
        jc_ortho.append(np.mean(jc_o))

    return sizes, np.array(gk_recon), np.array(jc_recon), np.array(gk_ortho), np.array(jc_ortho)


def plot_instability(
    sizes: Iterable[int],
    condition: float = 1e12,
    trials: int = 1,
    seed: int = 0,
    golub_kahan_kwargs: dict | None = None,
    jacobi_kwargs: dict | None = None,
    recon_path: str = "codes/reconstruction_instability.png",
    orth_path: str = "codes/orthogonality_instability.png",
):
    """
    Plot reconstruction and orthogonality errors on ill-conditioned matrices.
    """
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting but is not installed.")

    (
        sizes,
        gk_recon,
        jc_recon,
        gk_ortho,
        jc_ortho,
    ) = compare_on_ill_conditioned(
        sizes,
        condition=condition,
        trials=trials,
        seed=seed,
        golub_kahan_kwargs=golub_kahan_kwargs,
        jacobi_kwargs=jacobi_kwargs,
    )

    plt.figure(figsize=(6.8, 4.2))
    plt.plot(sizes, gk_recon, marker="o", label="Golub-Kahan")
    plt.plot(sizes, jc_recon, marker="s", label="Jacobi")
    plt.xlabel("Matrix size (n x n)")
    plt.ylabel("Rel. reconstruction error ||A - U diag(s) V^H||_F / ||A||_F")
    plt.yscale("log")
    plt.title(f"Ill-conditioned matrices (cond ~ 1e{int(np.log10(condition))})")
    plt.grid(True, alpha=0.4, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(recon_path, dpi=150)
    print(f"Saved reconstruction instability plot to {recon_path}")
    plt.close()

    plt.figure(figsize=(6.8, 4.2))
    plt.plot(sizes, gk_ortho, marker="o", label="Golub-Kahan")
    plt.plot(sizes, jc_ortho, marker="s", label="Jacobi")
    plt.xlabel("Matrix size (n x n)")
    plt.ylabel("Orthogonality error max(||U^H U - I||, ||V^H V - I||)_F / ||I||_F")
    plt.yscale("log")
    plt.grid(True, alpha=0.4, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(orth_path, dpi=150)
    print(f"Saved orthogonality instability plot to {orth_path}")
    plt.close()

    return sizes, gk_recon, jc_recon, gk_ortho, jc_ortho


def report_bad_case(n: int = 80, condition: float = 1e14, seed: int = 2024):
    """
    Print a concrete ill-conditioned matrix example comparing both methods.
    """
    A = make_ill_conditioned_matrix(n, condition=condition, seed=seed)
    gk_recon, gk_ortho = method_metrics(A, "golub-kahan")
    jc_recon, jc_ortho = method_metrics(A, "jacobi", max_sweeps=100)
    print(f"Ill-conditioned matrix n={n}, cond~{condition:.1e}")
    print(f"Golub-Kahan  recon err: {gk_recon:.3e}, orthogonality err: {gk_ortho:.3e}")
    print(f"Jacobi       recon err: {jc_recon:.3e}, orthogonality err: {jc_ortho:.3e}")


def _demo():
    rng = np.random.default_rng(42)
    A = rng.standard_normal((5, 3)) + 1j * rng.standard_normal((5, 3))

    U, s, Vt = svd_via_golub_kahan(A)
    reconstruction = U @ np.diag(s) @ Vt

    print("Input matrix A:\n", A)
    print("\nSingular values:", s)
    print("\nReconstruction error ||A - USigmaV^H||_F:", np.linalg.norm(A - reconstruction))


if __name__ == "__main__":  # pragma: no cover - simple demonstration
    _demo()
    report_bad_case(n=60, condition=1e14, seed=7)
    sizes = np.geomspace(10, 150, num=6, dtype=int)
    plot_instability(
        sizes,
        condition=1e14,
        trials=1,
        seed=3,
        golub_kahan_kwargs={},
        jacobi_kwargs={"max_sweeps": 120, "tol": 1e-14},
        recon_path="codes/reconstruction_instability.png",
        orth_path="codes/orthogonality_instability.png",
    )
