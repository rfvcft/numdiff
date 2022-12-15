import numpy as np
from scipy.sparse import diags
from numpy.linalg import solve
import matplotlib.pyplot as plt
from matplotlib import cm

def LW(u, dt):
    N = u.size
    dx = 1/(N+1)
    unew = np.zeros(N)
    for i in range(N):
        ux = (u[i-1] - u[(i+1) % N]) / (2*dx)
        uxx = (u[i-1] - 2*u[i] + u[(i+1) % N]) / dx**2
        unew[i] = u[i] - dt*u[i]*ux + dt**2/2*(2 * u[i] * ux**2 + u[i]**2 * uxx)
    return unew

def TRstep(Tdx, uold, dt, d):
    T = d * dt/2 * Tdx
    I = np.identity(uold.size)
    return solve(I - T, LW(uold, dt) + T @ uold)

def burgers():
    N = 250
    M = 1000
    d = 0.01
    tend = 1
    dt = tend/(M+1)
    dx = 1 / N
    xgrid = np.linspace(0, 1, N)

    T = diags([1, -2, 1], [-1, 0, 1], shape=(N, N)).toarray()
    T[0][-1] = 1
    T[-1][0] = 1

    # g = lambda x: np.sin(math.pi * x)
    g = lambda x: 10**0.5 * np.exp(-150 * np.power((x - 0.5), 2))
    # g = lambda x: 1.1*10**-1 * np.sin(3 * math.pi * x)
    U = np.zeros((M+1, N))
    u = g(xgrid)
    for i in range(M + 1):
        U[i] = u
        u = TRstep(1/dx**2 * T, u, dt, d)
    burgersVsh(N, M, tend, U)

def burgersVsh(N, M, tend, U):
    xx = np.linspace(0, 1, N)
    tt = np.linspace(0, tend, M + 1)
    T, X = np.meshgrid(tt, xx)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, T, np.transpose(U), cmap=cm.rainbow)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    plt.show()

if __name__ == '__main__':
    burgers()
