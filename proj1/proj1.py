import numpy as np
from scipy.linalg import norm, expm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def RK4step(f, told : float, uold : np.ndarray, h : float):
    y1 = f(told, uold)
    y2 = f(told + h/2, uold + h/2 * y1)
    y3 = f(told + h/2, uold + h/2 * y2)
    y4 = f(told + h, uold + h * y3)
    return uold + h/6 * (y1 + 2*y2 + 2*y3 + y4)

def RK34step(f, told, uold, h):
    y1 = f(told, uold)
    y2 = f(told + h/2, uold + h/2 * y1)
    y3 = f(told + h/2, uold + h/2 * y2)
    z3 = f(told + h, uold - h * y1 + 2*h * y2)
    y4 = f(told + h, uold + h * y3)
    ynew = uold + h/6 * (y1 + 2*y2 + 2*y3 + y4)
    # znew = uold + h/6 * (y1 + 4*y2 + z3)
    err = norm(h/6 * (2*y2 + z3 - 2*y3 - y4))
    return ynew, err

def newstep(tol, err, errold, hold, k):
    return (tol / err) ** (2/(3*k)) * (tol / errold) ** (-1/(3*k)) * hold

def adaptiveRK34(f, t0, tf, y0, tol):
    h0 = (abs(tf - t0) * tol ** 0.25) / (100 * (1 + norm(f(t0, y0))))  * 10**-1 #<- activate this for large mu:s in van der Pol
    y = y0
    h = h0
    t = t0
    errold = tol
    ys = [y0]
    ts = [t0]
    while t + h < tf:
        y, err = RK34step(f, t, y, h)
        t += h
        h = newstep(tol, err, errold, h, 4)
        ys.append(y)
        ts.append(t)
    y, err = RK34step(f, t, y, tf - t)
    ys.append(y)
    ts.append(tf)
    return ts, ys

def RK4int(f, y0, t0, tf, N):
    tgrid = np.linspace(t0, tf, N + 1)
    ygrid = np.zeros(N + 1)
    egrid = np.zeros(N + 1)
    h = (tf - t0) / (N)
    approx = y0
    ygrid[0] = approx
    egrid[0] = 0
    for i in range(N):
        approx = RK4step(f, t0, approx, h)
        ygrid[i + 1] = approx
        egrid[i + 1] = norm(approx - y0 * expm(f(t0, tgrid[i + 1] - t0)))
    return tgrid, ygrid, egrid

def errVSh() -> None:
    lambda0 = 1
    y0 = 1
    t0 = 0
    tf = 1
    f = lambda _, u: lambda0 * u
    tst = []
    ns = []
    errs = []
    for k in range(10):
        N = 2 ** k
        tg, ys, err = RK4int(f, y0, t0, tf, N)
        h = (tf - t0) / N
        ns.append(h)
        errs.append(norm(err[-1]))
        tst.append(h ** 4)

    plt.loglog(ns, errs)
    plt.loglog(ns, tst)
    plt.xlabel("N (2**k), log-scale")
    plt.ylabel("error, log-scale")
    plt.show()

def volterra():
    t0 = 0
    tf = 200
    tol = 10 ** -6
    volterra = lambda _, u: np.array([3 * u[0] - 9 * u[0]*u[1], 15 * u[0]*u[1] - 15 * u[1]])
    y0 = np.array([1, 1])
    t, y = adaptiveRK34(volterra, t0, tf, y0, tol)
    y1 = list(map(lambda x: x[0], y))
    y2 = list(map(lambda x: x[1], y))
    plt.plot(t, y)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.show()

def van_der_Pol(mu):
    tol = 10 ** -6
    van_der_Pol = lambda _, u: np.array([u[1], mu * (1 - u[0]**2) * u[1] - u[0]])
    y0 = np.array([2, 0])
    t0 = 0
    # tf = 0.7 * mu
    tf = 2 * mu

    t, y = adaptiveRK34(van_der_Pol, t0, tf, y0, tol)

    # van_der_Plot(t, y)

    return len(t)

def van_der_Pol_cheat(mu):
    van_der_Pol = lambda _, u: np.array([u[1], mu * (1 - u[0]**2) * u[1] - u[0]])
    y0 = [2, 0]
    t0 = 0
    # tf = 0.7 * mu
    tf = 2 * mu
    # print("mu:", mu)
    sol = solve_ivp(van_der_Pol, [t0, tf], y0, method='BDF')
    # print("y0:", y0)

    # van_der_Plot(sol.t, list(zip(sol.y[0], sol.y[1])))
    return sol.y.shape[1]

def van_der_Plot(t, y):
    y1 = list(map(lambda x: x[0], y))
    y2 = list(map(lambda x: x[1], y))
    # print(y)
    # print(y1)
    plt.figure()
    plt.plot(t, y2)
    plt.xlabel("t")
    plt.ylabel("y2")

    plt.figure()
    plt.plot(y1, y2)
    plt.xlabel("y1")
    plt.ylabel("y2")

    plt.show()


def test() -> None:
    # volterra()
    mus = [10, 15, 22, 33, 47, 68, 100, 150, 220, 330, 470, 680, 1000, 10000]
    # mus = [1000]
    mus2 = [2**n for n in range(70)]
    ns = []
    ns2 = []
    for mu in mus:
        n = van_der_Pol_cheat(mu)
        # print(n)
        # ns2.append(van_der_Pol(mu))
        ns.append(n)
    # van_der_Pol(10)
    # van_der_Pol_cheat(100)
    plt.loglog(mus, ns)
    # plt.loglog(mus, ns2)
    # plt.xscale('log')
    plt.xlabel("mu")
    plt.ylabel("N")
    plt.show()
    # errVSh()


if __name__ == '__main__':
    test()
