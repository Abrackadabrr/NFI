import numpy as np
import time
import matplotlib.pyplot as plt

import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pyamg


def poisson_2d_matrix(n: int) -> sp.csr_matrix:
    e = np.ones(n)
    T = sp.diags([-e, 4 * e, -e], [-1, 0, 1], shape=(n, n), format="csr")
    S = sp.diags([-e, -e], [-1, 1], shape=(n, n), format="csr")
    I = sp.eye(n, format="csr")
    A = sp.kron(I, T, format="csr") + sp.kron(S, I, format="csr")
    return A.tocsr()


# ============================================================
# 1) Honest timing (no callback / no residuals)
# ============================================================
def honest_time_cg(A, b, tol=1e-8, maxiter=5000, repeats=5):
    _ = spla.cg(A, b, rtol=tol, atol=0.0, maxiter=5)

    times = []
    x_last = None
    info_last = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        x, info = spla.cg(A, b, rtol=tol, atol=0.0, maxiter=maxiter, callback=None)
        dt = time.perf_counter() - t0
        times.append(dt)
        x_last, info_last = x, info

    rrel = np.linalg.norm(b - A @ x_last) / (np.linalg.norm(b) + 1e-30)
    return min(times), float(np.mean(times)), info_last, rrel


def honest_time_mg(A, b, tol=1e-8, maxiter=200, repeats=5):
    t0 = time.perf_counter()
    ml = pyamg.smoothed_aggregation_solver(A)
    setup_time = time.perf_counter() - t0

    _ = ml.solve(b, tol=tol, maxiter=2, cycle="V", residuals=None)

    times = []
    x_last = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        x = ml.solve(b, tol=tol, maxiter=maxiter, cycle="V", residuals=None)
        dt = time.perf_counter() - t0
        times.append(dt)
        x_last = x

    rrel = np.linalg.norm(b - A @ x_last) / (np.linalg.norm(b) + 1e-30)
    return setup_time, min(times), float(np.mean(times)), rrel


# ============================================================
# 2) Residual histories for plot (separate run)
# ============================================================
def residual_history_for_plot(A, b, tol=1e-8, maxiter_cg=5000, maxiter_mg=200):
    bnorm = np.linalg.norm(b) + 1e-30

    # CG residuals via callback (extra A@x per point, but ONLY for plotting)
    r_cg = []

    def cb(xk):
        r = b - A @ xk
        r_cg.append(np.linalg.norm(r) / bnorm)

    x_cg, info_cg = spla.cg(A, b, rtol=tol, atol=0.0, maxiter=maxiter_cg, callback=cb)
    r_cg.append(np.linalg.norm(b - A @ x_cg) / bnorm)

    # MG residuals via pyamg residuals list
    ml = pyamg.smoothed_aggregation_solver(A)
    residuals = []
    x_mg = ml.solve(b, tol=tol, maxiter=maxiter_mg, cycle="V", residuals=residuals)
    r_mg = np.array(residuals, dtype=float) / bnorm
    r_mg = np.append(r_mg, np.linalg.norm(b - A @ x_mg) / bnorm)

    return np.array(r_cg, dtype=float), r_mg


# ============================================================
# 3) Matvec counting WITHOUT passing wrappers into SciPy
# ============================================================
def cg_matvec_count_via_iterations(A, b, tol=1e-8, maxiter=5000):
    """
    SciPy CG uses (approximately) 1 matvec per iteration plus one at init.
    We count iterations from callback (cheap), then estimate matvec.
    """
    iters = 0

    def cb(_xk):
        nonlocal iters
        iters += 1

    x, info = spla.cg(A, b, rtol=tol, atol=0.0, maxiter=maxiter, callback=cb)
    # Typical CG: matvec ~ iters + 1 (init)
    matvec_est = iters + 1

    rrel = np.linalg.norm(b - A @ x) / (np.linalg.norm(b) + 1e-30)
    return iters, matvec_est, info, rrel


def mg_fine_matvec_equivalent(ml, cycles_done):
    """
    Оценка 'fine-level matvec equivalent' для MG.
    В V-цикле на каждом цикле на fine уровне обычно:
      - несколько применений smoother'а (внутри могут быть spmv, но это уже не A@x),
      - 1 раз считается residual r=b-Ax (это точно A@x на fine)
    Поэтому для сопоставления с CG считаем: 1 A@x на цикл на fine.
    => matvec_equiv_fine ≈ cycles_done
    """
    return int(cycles_done)


def mg_cycles_count(A, b, tol=1e-8, maxiter=200):
    ml = pyamg.smoothed_aggregation_solver(A)
    residuals = []
    x = ml.solve(b, tol=tol, maxiter=maxiter, cycle="V", residuals=residuals)
    cycles_done = len(residuals)  # в pyamg residuals обычно по итерациям/циклам
    rrel = np.linalg.norm(b - A @ x) / (np.linalg.norm(b) + 1e-30)
    return cycles_done, mg_fine_matvec_equivalent(ml, cycles_done), rrel


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    n = 128
    tol = 1e-8

    A = poisson_2d_matrix(n)
    b = np.ones(A.shape[0], dtype=float)

    # ---- honest time ----
    cg_best, cg_mean, cg_info, cg_rrel = honest_time_cg(A, b, tol=tol, maxiter=5000, repeats=5)
    mg_setup, mg_best, mg_mean, mg_rrel = honest_time_mg(A, b, tol=tol, maxiter=200, repeats=5)

    # ---- matvec counts (separate runs) ----
    cg_iters, cg_matvec, cg_info2, cg_rrel2 = cg_matvec_count_via_iterations(A, b, tol=tol, maxiter=5000)
    mg_cycles, mg_matvec_equiv_fine, mg_rrel2 = mg_cycles_count(A, b, tol=tol, maxiter=200)

    print("=== ЧЕСТНОЕ ВРЕМЯ (без callback/residuals) ===")
    print(f"CG: best={cg_best:.6f}s mean={cg_mean:.6f}s info={cg_info} final_rel_res={cg_rrel:.3e}")
    print(f"MG: setup={mg_setup:.6f}s best_solve={mg_best:.6f}s mean_solve={mg_mean:.6f}s "
          f"final_rel_res={mg_rrel:.3e} total(best)={(mg_setup + mg_best):.6f}s")

    print("\n=== MATVEC (отдельный прогон) ===")
    print(f"CG: iterations={cg_iters}, matvec_est≈{cg_matvec} (info={cg_info2}, final_rel_res={cg_rrel2:.3e})")
    print(f"MG: cycles={mg_cycles}, fine-level matvec equiv≈{mg_matvec_equiv_fine}, final_rel_res={mg_rrel2:.3e}")

    # ---- plot residual histories ----
    r_cg, r_mg = residual_history_for_plot(A, b, tol=tol, maxiter_cg=5000, maxiter_mg=200)

    plt.figure()
    plt.semilogy(np.arange(1, len(r_cg) + 1), r_cg, label="CG (SciPy) — итерации")
    plt.semilogy(np.arange(1, len(r_mg) + 1), r_mg, label="MG V-cycle (PyAMG) — V-циклы")
    plt.xlabel("Номер итерации / цикла")
    plt.ylabel("Относительная невязка ||r_k|| / ||b||")
    plt.title(f"Сходимость (отдельный прогон для графика): n={n}x{n}, tol={tol}")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()

