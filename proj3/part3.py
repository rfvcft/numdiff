import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.linalg import norm
from scipy.sparse import diags
from numpy.linalg import solve
import math

def convdif(u, Fdx, I):
    return solve(I-Fdx, (I+Fdx) @ u)

def convdifSolver():
    N = 300
    M = 300
    tend = 1
    a = 0.001
    d = 1
    print("Pe:", abs(a/d))
    xgrid = np.linspace(0, 1, N)

    # Build our matrix
    dx = 1/(N+1)
    dt = 1/(M+1)
    T = diags([1, -2, 1], [-1, 0, 1], shape=(N, N)).toarray()
    T[0][-1] = 1
    T[-1][0] = 1
    S = diags([1, 0, -1], [-1, 0, 1], shape=(N, N)).toarray()
    S[0][-1] = 1
    S[-1][0] = -1
    Fdx = dt/(2*dx) * (d/dx * T - a * S)
    # print("T:", T)
    # print("S:", S)
    # print("Fdx:", Fdx)
    I = np.identity(N)

    # g = lambda x: x*(1-x)
    # g = lambda x: np.power(np.sin(2 * math.pi * x), 2)
    # g = lambda x: np.sin(3 * math.pi * x)
    g = lambda x: np.exp(-100 * np.power((x - 0.5), 2))
    U = np.zeros((M+1, N))
    u = g(xgrid)
    for i in range(M + 1):
        U[i] = u
        u = convdif(u, Fdx, I)
    convdifVsh(N, M, tend, U)

def RMSVsh(N, M, tend, U):
    tt = np.linspace(0, tend, M+1)
    xx = [math.sqrt(1/(N+1)) * norm(u) for u in U]
    plt.plot(tt, xx)
    plt.show()


def convdifVsh(N, M, tend, U):
    xx = np.linspace(0, 1, N)
    tt = np.linspace(0, tend, M + 1)
    T, X = np.meshgrid(tt, xx)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, T, np.transpose(U), cmap=cm.rainbow)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    plt.show()

if __name__ == '__main__':
   convdifSolver()
