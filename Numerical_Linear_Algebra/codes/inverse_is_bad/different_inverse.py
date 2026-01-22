import numpy as np
import math

def hilbert(n):
    """Матрица Гильберта n x n"""
    return np.array([[1/(i+j+1) for j in range(n)] for i in range(n)])


def hilbert_inverse(n):
    """
    Точная обратная матрица Гильберта с использованием math.comb
    """
    H_inv = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            # (-1)^(i+j)
            sign = 1 if (i + j) % 2 == 0 else -1

            # (i+j+1)
            factor = i + j + 1

            # C(n+i, n-j-1)
            c1 = math.comb(n + i, n - j - 1)

            # C(n+j, n-i-1)
            c2 = math.comb(n + j, n - i - 1)

            # C(i+j, i)²
            c3 = math.comb(i + j, i)
            c3_sq = c3 * c3

            # Итоговая формула
            H_inv[i, j] = sign * factor * c1 * c2 * c3_sq

    return H_inv

n = int(input())
H = hilbert(n)
H_inv_exact = hilbert_inverse(n)
H_inv_numerical = np.linalg.inv(H)

np.set_printoptions(precision=1)

# Простое сравнение
print(f"Матрица Гильберта {n}x{n}")
print(f"cond(H) = {np.linalg.cond(H):.2e}")
print(f"Относительная ошибка inv: {np.linalg.norm(H_inv_exact - H_inv_numerical)/np.linalg.norm(H_inv_numerical):.2e}")

# Через SVD
H_inv_svd = np.linalg.pinv(H)  # Псевдообратная через SVD (с регуляризацией)
print(f"Относительная ошибка svd: {np.linalg.norm(H_inv_exact - H_inv_svd)/np.linalg.norm(H_inv_exact):.2e}")

# Через QR
Q, R = np.linalg.qr(H)
H_inv_qr = np.linalg.solve(R, Q.T)
print(f"Относительная ошибка qr: {np.linalg.norm(H_inv_exact - H_inv_qr)/np.linalg.norm(H_inv_exact):.2e}")

# Сравнение с единичной матрицей
print(np.linalg.norm(H_inv_exact @ H - np.eye(n)))
print(np.linalg.norm(H_inv_numerical @ H - np.eye(n)))
print(np.linalg.norm(H_inv_svd @ H - np.eye(n)))
print(np.linalg.norm(H_inv_qr @ H - np.eye(n)))
