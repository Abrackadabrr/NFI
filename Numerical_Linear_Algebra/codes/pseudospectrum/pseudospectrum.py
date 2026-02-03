import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import schur, solve_triangular


def resolvent_norm_via_schur_power(
    Q: np.ndarray,
    T: np.ndarray,
    z: complex,
    maxit: int = 80,
    tol: float = 1e-6,
    seed: int = 0,
):
    """
    Оценка ||(A - zI)^(-1)||_2 через Шур-разложение A = Q T Q* и степенной метод
    для оператора C = ( (A-zI)^* (A-zI) )^{-1}.

    Идея:
      sigma_min(A-zI)^2 = lambda_min( (A-zI)^*(A-zI) )
      => 1/sigma_min^2 = lambda_max( ((A-zI)^*(A-zI))^{-1} ) = lambda_max(C)

    Степенной метод на C требует применения v -> C v, что делается двумя решением СЛАУ:
      y = (A-zI)^(-H) v  (то есть (A-zI)^* y = v)
      w = (A-zI)^(-1)  y (то есть (A-zI)  w = y)

    Благодаря Шуру решаем треугольные системы:
      A-zI = Q (T - zI) Q*
    """
    n = T.shape[0]
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    v = v / np.linalg.norm(v)

    # Подготовим треугольные матрицы в шур-базисе
    U = T - z * np.eye(n, dtype=T.dtype)  # верхнетреугольная
    L = U.conj().T                         # нижнетреугольная: (T-zI)^*

    # Быстрые решатели через шур-базис:
    # solve (A - zI) x = b  => Q * solve(U, Q* b)
    def solve_M(b):
        qb = Q.conj().T @ b
        xq = solve_triangular(U, qb, lower=False, check_finite=False)
        return Q @ xq

    # solve (A - zI)^* x = b => Q * solve(L, Q* b)
    def solve_MH(b):
        qb = Q.conj().T @ b
        xq = solve_triangular(L, qb, lower=True, check_finite=False)
        return Q @ xq

    # Степенной метод для C
    lam_old = 0.0
    for _ in range(maxit):
        # w = C v = (M^* M)^(-1) v = M^(-1) (M^(-H) v)
        y = solve_MH(v)
        w = solve_M(y)

        nw = np.linalg.norm(w)
        if not np.isfinite(nw) or nw == 0:
            return np.inf  # практически сингулярно или численный развал

        v = w / nw

        # Релеевское частное для C: lambda ≈ v* (C v)
        # (можно использовать уже посчитанное w = C v_prev, но проще пересчитать одно применение)
        y2 = solve_MH(v)
        w2 = solve_M(y2)
        lam = np.vdot(v, w2).real  # должно быть >= 0 (в идеале)

        if lam > 0 and abs(lam - lam_old) / lam < tol:
            lam_old = lam
            break
        lam_old = lam

    if lam_old <= 0 or not np.isfinite(lam_old):
        return np.inf

    # lam_old ≈ 1/sigma_min^2  =>  ||(A-zI)^(-1)||_2 = 1/sigma_min = sqrt(lam_old)
    return float(np.sqrt(lam_old))


def pseudospectrum_resolvent_schur(
    A: np.ndarray,
    eps: float,
    xlim,
    ylim,
    nx: int = 200,
    ny: int = 200,
    maxit: int = 80,
    tol: float = 1e-6,
):
    """
    Считает поле log10(||(A-zI)^(-1)||_2) на сетке, но:
      1) сначала A -> комплексная форма Шура A = Q T Q*
      2) в каждой точке оценивает норму резольвенты через степенной метод,
         используя только треугольные solve в шур-базисе.
    """
    # Комплексная форма Шура (T — верхнетреугольная)
    T, Q = schur(A, output="complex")

    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)

    R = np.empty((ny, nx), dtype=float)
    mask = np.zeros((ny, nx), dtype=bool)
    thr = 1.0 / eps

    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            z = x + 1j * y
            rn = resolvent_norm_via_schur_power(Q, T, z, maxit=maxit, tol=tol, seed=123)
            R[j, i] = rn
            mask[j, i] = (rn >= thr)

    with np.errstate(divide="ignore", invalid="ignore"):
        logR = np.log10(R)

    return xs, ys, R, logR, mask


def demo(n=50, eps=1e-2, nx=220, ny=220):
    # Ненормальная матрица (чтобы псевдоспектр был интереснее)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = A + 2.0 * np.triu(rng.standard_normal((n, n)), 1)

    eigs = np.linalg.eigvals(A)
    pad = 2.0
    xlim = (eigs.real.min() - pad, eigs.real.max() + pad)
    ylim = (eigs.imag.min() - pad, eigs.imag.max() + pad)

    xs, ys, R, logR, mask = pseudospectrum_resolvent_schur(
        A, eps=eps, xlim=xlim, ylim=ylim, nx=nx, ny=ny, maxit=80, tol=1e-6
    )

    X, Y = np.meshgrid(xs, ys)
    level = np.log10(1.0 / eps)

    # ---- График 1: поле нормы резольвенты + контур уровня 1/eps ----
    plt.figure(figsize=(7, 6))
    plt.imshow(
        logR,
        origin="lower",
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        aspect="auto",
    )
    plt.colorbar(label=r"$\log_{10}\|(A - zI)^{-1}\|_2$")
    plt.contour(X, Y, logR, levels=[level], colors="black", linewidths=2)
    plt.scatter(eigs.real, eigs.imag, s=12, marker="x", color="white")
    plt.title(f"Поле нормы резольвенты (Шур + степенной метод), n={n}, ε={eps:g}")
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.tight_layout()
    plt.show()

    # ---- График 2: только точки внутри ε-псевдоспектра ----
    logR_ps = np.where(mask, logR, np.nan)

    plt.figure(figsize=(7, 6))
    plt.imshow(
        logR_ps,
        origin="lower",
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        aspect="auto",
    )
    plt.colorbar(label=r"$\log_{10}\|(A - zI)^{-1}\|_2$")
    plt.scatter(eigs.real, eigs.imag, s=12, marker="x", color="black")
    plt.title(f"ε-псевдоспектр (только точки внутри), ε={eps:g}")
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo(n=50, eps=5e-1, nx=100, ny=100)



