import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.linalg import norm
import math

def LaxWen(u, amu):
    N = u.size
    # lw = np.array([amu/2 * (1+amu), 1 - amu^2, -amu/2 * (1-amu)])
    unew = np.zeros(N)
    for i in range(N):
        # us = np.array([u[i-1], u[i], u[(i+1) % N]])
        unew[i] = (amu/2 * (1+amu))*u[i-1] + (1 - amu**2)*u[i] + (-amu/2 * (1-amu))*u[(i+1) % N]
        # unew[i] = np.dot(lw, us)
    return unew

def LaxWenSolver():
    N = 1000
    M = 1000
    # kaosar vid N=109, M=100
    tend = 5
    a = 0.2
    dt = tend/(M+1)
    dx = 1/(N+1)
    amu = a * dt / dx
    xgrid = np.linspace(0, 1, N)
    print("amu:", amu)
    # g = lambda x: x*(1-x)
    # g = lambda x: np.power(np.sin(2 * math.pi * x), 2)
    # g = lambda x: np.sin(3 * math.pi * x)
    g = lambda x: np.exp(-100 * np.power((x - 0.5), 2))
    U = np.zeros((M+1, N))
    u = g(xgrid)
    for i in range(M + 1):
        U[i] = u
        u = LaxWen(u, amu)
    lwVsh(N, M, tend, U)
    RMSVsh(N, M, tend, U)

def RMSVsh(N, M, tend, U):
    tt = np.linspace(0, tend, M+1)
    xx = [math.sqrt(1/(N+1)) * norm(u) for u in U]
    plt.plot(tt, xx)
    plt.show()


def lwVsh(N, M, tend, U):
    xx = np.linspace(0, 1, N)
    tt = np.linspace(0, tend, M + 1)
    T, X = np.meshgrid(tt, xx)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, T, np.transpose(U), cmap=cm.rainbow)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    plt.show()

if __name__ == '__main__':
    LaxWenSolver()
