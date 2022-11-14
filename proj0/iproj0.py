from typing import Tuple
import numpy as np
from scipy.linalg import expm, norm, inv
import matplotlib.pyplot as plt

def ieulerstep(A : np.ndarray, uold : np.ndarray, h : float) -> np.ndarray:
    # y_n+1 = y_n + h * f(t_n+1, y_n+1)
    # f(t_n+1, y_n+1) = A * y_n+1
    # (I - h*A) * y_n+1 = y_n
    # y_n+1 = (I - h*A)^-1 * y_n
	return inv(np.identity(len(A)) - h * A) @ uold

def ieulerint(A : np.ndarray, y0 : np.ndarray, t0 : float, tf : float, N : int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	tgrid = np.linspace(t0, tf, N + 1)
	ygrid = np.zeros((N + 1, len(A)))
	# egrid = np.zeros((N + 1, len(A)))
	egrid = np.zeros(N + 1)
	h = (tf - t0) / (N + 1)
	approx = y0
	ygrid[0] = approx
	egrid[0] = 0
	for i in range(N):
		approx = ieulerstep(A, approx, h)
		ygrid[i + 1] = approx
		egrid[i + 1] = norm(approx - y0 * expm(A * (tgrid[i] - t0)))

	return tgrid, ygrid, egrid

def ierrVSh(A : np.ndarray, y0 : np.ndarray, t0 : float, tf : float) -> None:
	ns = []
	errs = []
	for k in range(15):
		N = 2 ** k
		tg, ys, err = ieulerint(A, y0, t0, tf, N)
		ns.append(N)
		errs.append(norm(err[-1]))
	# print(ns)
	# print(errs)
	plt.loglog(ns, errs)
	plt.xlabel("N (2**k), log-scale")
	plt.ylabel("error, log-scale")
	plt.show()

def ierrVShTime(A : np.ndarray, y0 : np.ndarray, t0 : float, tf : float, N : int) -> None:
	times, ys, errs = ieulerint(A, y0, t0, tf, N)
	# print(times)
	# print(errs)
	plt.plot(times, errs)
	plt.yscale('log')
	plt.xlabel("t")
	plt.ylabel("error (relative), log-scale")
	plt.show()

def test() -> None:
	A = np.array([[-1, 100], [0, -3]])
	# lambda0 = 1
	# A = np.array([lambda0])
	t0 = 0
	tf = 10
	y0 = np.array([1, 1])
	N = 10000

	# ts, appr, err = eulerint(A, y0, t0, tf, N)
	# print(appr)
	# print(err)
	# ierrVSh(A, y0, t0, tf)
	ierrVShTime(A, y0, t0, tf, N)

if __name__ == '__main__':
	test()
