import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
import matplotlib.animation as animation
import time

def F(u, dx, dt, nu, u_old):
    f = np.zeros_like(u)
    for i in range(1, len(u) - 1):
        derivx = (u_old[i + 1] - u_old[i - 1]) / (2 * dx)
        derivxx = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx**2
        f[i] = (u[i] - u_old[i]) / dt + u[i] * derivx - nu * derivxx
    return f
def J(u, dx, dt, nu,u_old):
    N = len(u)
    J = np.zeros((N - 2, N - 2))
    diag = 1 / dt + 2 * nu / dx**2
    sup = -nu / dx**2
    inf= sup
    for i in range(N - 2):
        if i > 0:
            J[i, i - 1] = sup
        J[i, i] = diag+(u_old[i + 1] - u_old[i - 1]) / (2 * dx)

        if i < N - 3:
            J[i, i + 1] = inf
    return J
def newton(u, u_old, dx, dt, nu, TOL, N,t1):
    t0= time.time()
    errors, residuals = [], []
    for k in range(1,N):
        Fx = F(u, dx, dt, nu, u_old)
        Jx = J(u, dx, dt, nu,u_old)
        s = np.linalg.solve(Jx, -Fx[1:-1])
        u[1:-1] += s
        errors.append(np.linalg.norm(s))
        residuals.append(np.linalg.norm(Fx))
        if errors[-1] < TOL:
            break
    t1= t1+ time.time()-t0
    return u, errors, residuals,t1, k
def broyden(u,u_old, dx, dt, nu, TOL, N,t1):
    t0= time.time()
    Fx = F(u, dx, dt, nu, u_old)
    Jx= J(u,dx,dt,nu,u_old)
    A = np.linalg.inv(Jx)
    s = -A @ Fx[1:-1]
    u[1:-1] += s
    errors, residuals = [np.linalg.norm(s)], [np.linalg.norm(Fx)]
    
    for k in range(1, N):
        Fx_new = F(u, dx, dt, nu, u_old)
        y = Fx_new[1:-1] - Fx[1:-1]
        z = -A @ y
        p = -s @ z
        uv = s @ A
        A += (1 / p) * np.outer(s + z, uv)
        s = -A @ Fx_new[1:-1]
        u[1:-1] += s
        errors.append(np.linalg.norm(s))
        residuals.append(np.linalg.norm(Fx_new))
        Fx = Fx_new
        if errors[-1] < TOL:
            break
    t1= t1+ time.time() -t0
    return u, errors, residuals,t1, k
def steffensen(u,u_old,dx,dt,nu,TOL,N,t1):
    
    pass
def empirical_order(errors):
    orders = []
    for k in range(2, len(errors)):
        ekm1, ek, ekp1 = errors[k-2:k+1]
        p = np.log(ekp1 / ek) / np.log(ek / ekm1)
        orders.append(p)
    return orders
def solve(method):
    u = u0.copy()
    U = np.zeros((Nt, Nx))
    U[0, :] = u
    error_log = []
    residual_log = []
    t1=0
    iter=0
    total_iter=0
    for n in range(1, Nt):
        u_old = u.copy()
        u, errors, residuals,t1,iter = method(u.copy(), u_old, dx, dt, nu, TOL, MAX_ITER,t1)
        total_iter= total_iter+iter
        U[n, :] = u
        if n == Nt-1:  # Only store last timestep convergence for plotting (worst case scenario)
            error_log = errors
            residual_log = residuals
    return U, error_log, residual_log,t1,total_iter
def plot_convergence(errors1, res1, errors2, res2):
    plt.figure(figsize=(10, 5))
    plt.semilogy(errors1, label='Broyden Step Norm')
    plt.semilogy(res1, label='Broyden Residual')
    plt.semilogy(errors2, '--', label='Newton Step Norm')
    plt.semilogy(res2, '--', label='Newton Residual')
    plt.xlabel('Iteration')
    plt.ylabel('Norm')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
def print_order(orders, method_name, digits=1):
    print(f"\nEmpirical Order of Convergence ({method_name}):")
    for i, p in enumerate(orders, start=2):  # because p_k uses e_{k-1}, e_k, e_{k+1}
        print(f"  Iter {i}: p ≈ {p:.{digits}f}")
def animate_solution(U, method_name):
    fig, ax = plt.subplots()
    line, = ax.plot(x, U[0, :])
    ax.set_xlim(0, L)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"{method_name} Method Solution Evolution")

    def update(frame):
        line.set_ydata(U[frame, :])
        ax.set_title(f"{method_name} t = {frame * dt:.2f}")
        return line,

    ani = animation.FuncAnimation(fig, update, frames=range(0, Nt, max(1, Nt // 200)), interval=30, blit=True)
    plt.show()
def plot_all_snapshots(U, x, dt, method_name="Method", steps=None):
    if steps is None:
        # default: pick ~5 evenly spaced frames including first and last
        steps = np.linspace(0, U.shape[0] - 1, 5, dtype=int)

    plt.figure(figsize=(10, 6))
    for n in steps:
        plt.plot(x, U[n, :], label=f't = {n * dt:.2f}')
    
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.title(f'{method_name}: Snapshots at Various Time Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{method_name}_snapshots.png")
    plt.show()
def plot_all_differences(U1, U2, x, dt, method1="Broyden", method2="Newton", steps=None):
    if steps is None:
        steps = np.linspace(0, U1.shape[0] - 1, 5, dtype=int)

    plt.figure(figsize=(10, 6))
    for n in steps:
        diff = U1[n, :] - U2[n, :]
        plt.plot(x, diff, label=f't = {n * dt:.2f}')
    
    plt.xlabel('x')
    plt.ylabel(f'{method1} - {method2}')
    plt.title(f'Difference Between {method1} and {method2} at Various Time Steps')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'diff_{method1}_{method2}.png')
    plt.show()

# Parameters
L, Nx = 1, 200
T, dt = 0.5, 0.05
nu = 0.01
x = np.linspace(0, L, Nx)
dx = L / (Nx - 1)
Nt = int(T / dt) + 1  
TOL = 1e-15
MAX_ITER = 30

# Initial condition
u0 = np.sin(np.pi * x)
u0[0] = u0[-1] = 0

U_broyden, broyden_errors, broyden_residuals,timeB,iterB= solve(broyden)
U_newton, newton_errors, newton_residuals,timeN,iterN = solve(newton)
print(f"Max difference: {np.max(np.abs(U_broyden - U_newton)):.2e} \nIterations Newton: {int(iterN)} \nIterations Broyden: {int(iterB)}")
print(f"Broyden {round((timeB)*1000,2)} miliseconds vs Newton: {round((timeN)*1000,2)} miliseconds. Broyden is {round((timeN)/(timeB)*100-100 ,2)} % faster")
plot_convergence(broyden_errors, broyden_residuals, newton_errors, newton_residuals)
print_order(empirical_order(broyden_errors), "Broyden")
print_order(empirical_order(newton_errors), "Newton")
plot_all_snapshots(U_broyden,x,dt,"Broyden")
plot_all_snapshots(U_newton,x,dt,"Newton")
plot_all_differences(U_broyden, U_newton, x, dt)
