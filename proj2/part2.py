import numpy as np
from numpy.linalg import eig
from scipy.sparse import diags
from scipy.linalg import norm
import math
import matplotlib.pyplot as plt

def SLeig(N):
    L = 1
    h = L / (N + 1)
    T = diags([1, -2, 1], [-1, 0, 1], shape=(N, N))
    T = T.toarray()
    # Set correct values in the double derivate matrix
    # corresponding to homogenous neuman conditions
    # 0 = beta = (y_n-1 - 4y_n + 3y_n+1)/2h
    # ie. y_n+1 = 1/3(4y_n - y_n-1)
    T[-1][-2] = 2/3
    T[-1][-1] = -2/3
    return eig(T/h**2)

def SLplot():
    ns = []
    M = 3
    errs = [[] for _ in range(M)]
    for k in range(2, 10):
        N = 2 ** k
        eigs, modes = SLeig(N)
        tuples = sorted(zip(eigs, modes), key=lambda x: abs(x[0]))
        eigs, modes = [t[0] for t in tuples], [t[1] for t in tuples]
        for j in range(M):
            theoretical = -((2*j + 1) * math.pi)** 2 / 4
            errs[j].append(norm(eigs[j] - theoretical))
        ns.append(N)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, axs = plt.subplots(3, 1)
    for i in range(M):
        axs[i].loglog(ns, errs[i])
        axs[i].loglog(ns, np.float_power(ns, -2)) # help line at slope -2
        axs[i].set_ylabel("$\lambda_{%d} = -\\frac{(%d \pi)^2}{4L}$" %(i+1, 2*i+1))
    fig.suptitle("""Errors of the computed eigenvalues against the number of
                    interior grid points $N$ (blue)\ntogether with a help-line
                    with slope -2 (orange), log-log scale""")
    plt.show()

def endpoints(vec):
    vec = np.insert(vec, 0, 0)
    y_final = 1/3 * (4 * vec[-1] - vec[-2])
    return np.append(vec, y_final)

def SL499():
    M = 3
    eigs, modes = SLeig(499)
    tgrid = np.linspace(0, 1, 501)
    sorted_indices = np.argsort(-eigs)[:M]
    n_largest = [(eigs[i], modes[:, i]) for i in sorted_indices]
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    for i, (eig, mode) in enumerate(n_largest):
        print(eig)
        plt.plot(tgrid, endpoints(mode)) # plot with alpha at first index
    plt.xlabel("$x$")
    plt.ylabel("$y(x)$")
    plt.title("First three eigenmodes for $y'' = \lambda y$ with $y(0) = 0, y'(1) = 0$.")
    plt.show()

def pad(vec):
    vec = np.insert(vec, 0, 0)
    return np.append(vec, 0)

def shrodinger(N, vVec):
    h = 1 / (N+1)
    M = 10
    xgrid = np.linspace(0, 1, N + 2)
    T = diags([1, -2, 1], [-1, 0, 1], shape=(N, N))
    V = diags(vVec, 0, shape=(N, N))
    A = (T/h**2-V).toarray()
    eigs, waves = eig(A)
    sorted_indices = np.argsort(-eigs)[:M]
    n_largest = [(eigs[i], pad(waves[:, i])) for i in sorted_indices]
    # Plotting
    fig, axs = plt.subplots(1, 2)
    cm = plt.cm.jet(np.linspace(0, 1, M))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    for i in range(2):
        axs[i].set_prop_cycle('color', list(cm))
        axs[i].plot(xgrid, pad(vVec), 'k--')
    for (E, wave) in n_largest:
        axs[0].plot(xgrid, 1000 * wave - E)
        axs[1].plot(xgrid, 16000 * np.power(np.abs(wave), 2) - E)
    fig.suptitle("""Wave function and probability densisty corresponding to
                    $V(x) = 800\sin(2\pi x)^2$""")
    axs[0].set_ylabel("$\psi_k + E_k$ and $V(x)$ (dashed)")
    axs[1].set_ylabel("$|\psi_k|^2 + E_k$ and $V(x)$ (dashed)")
    plt.jet()
    plt.show()

def shrod():
    N = 498
    # V = lambda x: [0 for _ in range(len(x))]
    # V = lambda x: 700 * (0.5 - np.abs(x - 0.5))
    # V = lambda x: 800 * np.power(np.sin(math.pi * x), 2)
    V = lambda x: 800 * np.power(np.sin(2 * math.pi * x), 2)
    xgrid = np.linspace(0, 1, N)
    shrodinger(N, V(xgrid))

if __name__ == "__main__":
    # SLplot()
    # SL499()
    shrod()
