import numpy as np
import matplotlib.pyplot as plt
import time

# --- Bratu residual and Jacobian ---
def F_bratu(u, dx, λ):
    """
    Compute the interior residuals F_i = (u_{i-1}-2u_i+u_{i+1})/dx^2 + λ e^{u_i}.
    Returns an array of length N-2 (interior points only).
    """
    N = len(u)
    f = np.zeros(N-2)
    for i in range(1, N-1):
        f[i-1] = (u[i-1] - 2*u[i] + u[i+1]) / dx**2 + λ * np.exp(u[i])
    return f

def J_bratu(u, dx, λ):
    """
    Build the (N-2)x(N-2) Jacobian matrix for the Bratu problem.
    J[i,i]   = -2/dx^2 + λ e^{u_{i+1}}
    J[i,i-1] =  1/dx^2
    J[i,i+1] =  1/dx^2
    """
    N = len(u)-2
    J = np.zeros((N, N))
    diag = -2.0/dx**2 + λ * np.exp(u[1:-1])
    off  =  1.0/dx**2
    for i in range(N):
        J[i, i] = diag[i]
        if i > 0:   J[i, i-1] = off
        if i < N-1: J[i, i+1] = off
    return J

# --- Newton and Broyden solvers (as in your template) ---
def newton(u, dx, λ, TOL, MAX_ITERS):
    t0 = time.time()
    errors, residuals = [], []
    for k in range(1, MAX_ITERS+1):
        F = F_bratu(u, dx, λ)
        J = J_bratu(u, dx, λ)
        s = np.linalg.solve(J, -F)
        u[1:-1] += s
        err = np.linalg.norm(s)
        res = np.linalg.norm(F)
        errors.append(err)
        residuals.append(res)
        if err +res< TOL:
            break
    return u, errors, residuals, time.time()-t0, k

def broyden(u, dx, λ, TOL, MAX_ITERS):
    t0 = time.time()
    F = F_bratu(u, dx, λ)
    J = J_bratu(u, dx, λ)
    A = np.linalg.inv(J)
    s = -A.dot(F)
    u[1:-1] += s
    errors, residuals = [np.linalg.norm(s)], [np.linalg.norm(F)]
    for k in range(1, MAX_ITERS+1):
        F_new = F_bratu(u, dx, λ)
        y = F_new - F
        z = -A.dot(y)
        p = -s.dot(z)
        uv = s.dot(A)
        A += np.outer(s + z, uv) / p
        s = -A.dot(F_new)
        u[1:-1] += s
        err = np.linalg.norm(s)
        res = np.linalg.norm(F_new)
        errors.append(err)
        residuals.append(res)
        F = F_new
        if err +res < TOL:
            break
    return u, errors, residuals, time.time()-t0, k

# --- Empirical order of convergence ---
def empirical_order(errors):
    orders = []
    for i in range(2, len(errors)):
        e0, e1, e2 = errors[i-2], errors[i-1], errors[i]
        orders.append(np.log(e2/e1) / np.log(e1/e0))
    return orders

# --- Driver to run one solve ---
def run_solve(method, u0, dx, λ, TOL, MAX_ITERS):
    u = u0.copy()
    u, errs, ress, t_cpu, nit = method(u, dx, λ, TOL, MAX_ITERS)
    return u, errs, ress, t_cpu, nit

# --- Plotting utilities ---
def plot_convergence(e1, r1, e2, r2):
    e1 = np.array(e1)
    r1 = np.array(r1)
    e2 = np.array(e2)
    r2 = np.array(r2)
    plt.semilogy(e1 +r1, '-o', label='Broyden ‖Δu‖')
    #plt.semilogy(r1, '-x', label='Broyden ‖F‖')
    plt.semilogy(e2+ r2, '--o', label='Newton ‖Δu‖')
    #plt.semilogy(r2, '--x', label='Newton ‖F‖')
    plt.xlabel('Iteration')
    plt.ylabel('Norm')
    plt.title('Bratu: Convergence Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

def print_order(orders, name):
    print(f"\nEmpirical orders ({name}):")
    for i,p in enumerate(orders, start=2):
        print(f"  iter {i}: p ≈ {p:.2f}")

# --- Main ---
if __name__ == '__main__':
    # Parameters
    L, N = 1.0, 100          # domain [0,1] with N+1 points
    dx = L / N
    λ = 3.51382                 # try 1.0, 3.0, 6.0
    TOL = 1e-4
    MAX_ITERS = 100

    # Initial guess: zero interior + boundary zeros
    u0 = np.zeros(N+1)
    u0[0] = u0[-1] = 0

    # Run both methods
    U_b, eb, rb, tb, kb = run_solve(broyden, u0, dx, λ, TOL, MAX_ITERS)
    U_n, en, rn, tn, kn = run_solve(newton, u0, dx, λ, TOL, MAX_ITERS)

    # Print summary
    print(f"λ = {λ}")
    print(f" Newton: {kn} iters, {tn*1000:.1f} ms")
    print(f" Broyden: {kb} iters, {tb*1000:.1f} ms")
    speedup = (tn/tb - 1)*100
    print(f" Broyden ~ {speedup:.1f}% faster\n")

    # Convergence plots
    plot_convergence(eb, rb, en, rn)

    # Empirical orders
    print_order(empirical_order(rb), "Broyden")
    print_order(empirical_order(rn), "Newton")
 
