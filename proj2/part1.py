import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import norm
import math
import matplotlib.pyplot as plt


def twopBVP(fvec, alpha, beta, L, N):
    h = L / (N + 1)
    T = diags([1, -2, 1], [-1, 0, 1], shape=(N, N))
    fvec *= h**2
    fvec[0] += -alpha
    fvec[-1] += -beta
    y = spsolve(T.tocsc(), fvec)
    y = np.insert(y, 0, alpha)
    y = np.append(y, beta)
    return y

def errVSh():
    lambda0 = 3
    f = lambda t: lambda0**2 * np.exp(lambda0 * t)
    y = lambda t: np.exp(lambda0 * t)
    L = 1
    errs = []
    ns = []
    tst = []
    for k in range(2, 15):
        N = 2 ** k
        tgrid = np.linspace(0, L, N + 2)
        fvec = f(tgrid[1:-1])
        ygrid = twopBVP(fvec, y(0), y(L), L, N)
        h = L / (N+1)
        err = math.sqrt(h) * norm(ygrid - y(tgrid)) # using RMS-norm
        errs.append(err)
        ns.append(N)
        tst.append(N ** -2)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure()
    plt.loglog(ns, errs)
    plt.loglog(ns, tst)
    plt.xlabel("$\displaystyle N (= 2^k, k = 2\ldots 14)$, log-scale")
    plt.ylabel("RMS-error, log-scale")
    plt.title("""Error vs number of steps for $\displaystyle y'' = \lambda^2 e^{\lambda t}$,
                $\lambda = 3$ (blue)\nReference line is $N$ vs $N^{-2}$ (orange)""")

    plt.figure()
    plt.plot(tgrid, ygrid)
    plt.plot(tgrid, y(tgrid))
    plt.xlabel("$t$")
    plt.ylabel("$y(t)$")
    plt.title("""Numerical calculation (blue) and true function (orange) of
                $y(t) = e^{\lambda t}$, $\lambda = 3$ for $N=2^{14}$""")
    plt.show()

def beamSolver():
    N = 999
    L = 10
    E = 1.9 * 10**11
    I = lambda x: 10**-3 * (3 - 2 * np.cos((math.pi * x) / L)**12)
    q = -50 * 10**3 * np.ones(N)
    tgrid = np.linspace(0, L, N + 2)
    M = twopBVP(q, 0, 0, L, N)
    fvec = M[1:-1] / (E * I(tgrid[1:-1]))
    u = twopBVP(fvec, 0, 0, L, N)
    print(u[500]) # -0.011741059085879973
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure()
    plt.plot(tgrid, M)
    plt.xlabel("L [m]")
    plt.ylabel("M [Nm]")
    plt.figure()
    plt.plot(tgrid, u)
    plt.xlabel("L [m]")
    plt.ylabel("u [m]")
    plt.show()

if __name__ == '__main__':
    errVSh()
    # beamSolver()

