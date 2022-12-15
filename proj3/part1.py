import numpy as np
from numpy.linalg import solve
from scipy.sparse import diags
import math
import matplotlib.pyplot as plt
from matplotlib import cm

def eulerstep(Tdx, uold, dt):
    return uold + dt * Tdx @ uold

def TRstep(Tdx, uold, dt):
    T = dt/2 * Tdx
    I = np.identity(uold.size)
    return solve(I - T, (I + T) @ uold)

def pad(vec): # COPYRIGHTED BY ME - NOBODY ELSE USE THIS WITHOUT PERMISSION
    vec = np.insert(vec, 0, 0)
    return np.append(vec, 0)

def diffusion():
    tend = 1/2
    N = 99
    M = 10000
    # N = 99, M = 9990, eulerstep right on stability
    # N = 100, M = 100, TRstep creates spikes
    dt = tend/(M+1)
    dx = 1/(N+1)
    xgrid = np.linspace(0, 1, N)
    T = 1/dx**2 * diags([1, -2, 1], [-1, 0, 1], shape=(N, N)).toarray()
    # g = lambda x: x*(1-x)
    # g = lambda x: np.power(np.sin(2 * math.pi * x), 2)
    g = lambda x: np.sin(math.pi * x)
    U = np.zeros((M+1, N+2))
    u = g(xgrid)
    for i in range(M + 1):
        U[i] = pad(u)
        u = eulerstep(T, u, dt)
        # u = TRstep(T, u, dt)
    diffVsh(N, M, tend, U)

def diffVsh(N, M, tend, U):
    xx = np.linspace(0, 1, N + 2)
    tt = np.linspace(0, tend, M + 1)
    T, X = np.meshgrid(tt, xx)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, T, np.transpose(U), cmap=cm.rainbow)
    plt.show()


if __name__ == '__main__':
    diffusion()
