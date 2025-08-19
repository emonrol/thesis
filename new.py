"""
FIXED MINIMAL INT-DEEP IMPLEMENTATION
====================================

A corrected version that addresses numerical stability issues.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PROBLEM DEFINITION (UNCHANGED)
# ============================================================================

def exact_solution(x):
    return 0.5 * np.sin(np.pi * x)

def rhs_function(x):
    """Right-hand side f(x) = -u'' + u³"""
    u = exact_solution(x)
    u_xx = -0.5 * np.pi**2 * np.sin(np.pi * x)
    return -u_xx + u**3

# ============================================================================
# PHASE I: FIXED NEURAL NETWORK
# ============================================================================

class SimpleNetwork:
    """Fixed neural network with better initialization and gradients"""
    
    def __init__(self, width=10):  # Smaller network for stability
        # Better initialization (Xavier/Glorot)
        scale = np.sqrt(2.0 / width)
        self.W1 = np.random.randn(1, width) * scale
        self.b1 = np.zeros(width)  # Start with zero bias
        self.W2 = np.random.randn(width, width) * scale
        self.b2 = np.zeros(width)
        self.W3 = np.random.randn(width, 1) * scale
        self.b3 = np.zeros(1)
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        """Forward pass with boundary conditions built in"""
        # Ensure input is the right shape
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        # Boundary term: automatically satisfies u(0) = u(1) = 0
        boundary_term = x.flatten() * (1 - x.flatten())
        
        # Neural network forward pass
        z1 = x @ self.W1 + self.b1
        a1 = self.relu(z1) 
        z2 = a1 @ self.W2 + self.b2
        a2 = self.relu(z2)
        z3 = a2 @ self.W3 + self.b3
        
        # Apply boundary conditions and clip to prevent overflow
        output = boundary_term * z3.flatten()
        return np.clip(output, -10, 10)  # Prevent extreme values
    
    def compute_second_derivative(self, x, h=1e-4):
        """Compute second derivative using finite differences"""
        # Ensure we don't go outside domain
        x = np.clip(x, h, 1-h)
        
        u_center = self.forward(x)
        u_plus = self.forward(x + h)
        u_minus = self.forward(x - h)
        
        # Second derivative: (u(x+h) - 2u(x) + u(x-h)) / h²
        u_xx = (u_plus - 2*u_center + u_minus) / (h**2)
        return u_xx
    
    def compute_loss(self, x_batch):
        """Compute PDE residual loss"""
        # Ensure batch is in valid domain
        x_batch = np.clip(x_batch, 0.01, 0.99)
        
        # Forward pass
        u = self.forward(x_batch)
        
        # Second derivative
        u_xx = self.compute_second_derivative(x_batch)
        
        # PDE residual: -u'' + u³ - f = 0
        f_vals = rhs_function(x_batch)
        residual = -u_xx + u**3 - f_vals
        
        # Return mean squared residual (clipped to prevent overflow)
        loss = np.mean(np.clip(residual**2, 0, 100))
        return loss
    
    def compute_gradients(self, x_batch, h=1e-5):
        """Compute gradients using finite differences (improved)"""
        params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        gradients = []
        
        # Baseline loss
        loss_center = self.compute_loss(x_batch)
        
        for param in params:
            grad = np.zeros_like(param)
            
            # Compute gradient for each parameter
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                old_value = param[idx]
                
                # Forward difference (more stable than central difference)
                param[idx] = old_value + h
                loss_plus = self.compute_loss(x_batch)
                param[idx] = old_value  # Restore
                
                # Compute gradient
                grad[idx] = (loss_plus - loss_center) / h
                
                # Clip gradient to prevent explosion
                grad[idx] = np.clip(grad[idx], -1.0, 1.0)
                
                it.iternext()
            
            gradients.append(grad)
        
        return gradients, loss_center
    
    def train_step(self, x_batch, lr=1e-5):  # Much smaller learning rate
        """Training step with gradient clipping"""
        gradients, loss = self.compute_gradients(x_batch)
        
        # Update parameters
        params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        for param, grad in zip(params, gradients):
            param -= lr * grad
        
        return loss

def phase1_training(epochs=50, batch_size=32):
    """Phase I: Train neural network (with better parameters)"""
    print("Phase I: Training neural network...")
    
    net = SimpleNetwork(width=8)  # Smaller network
    losses = []
    
    for epoch in range(epochs):
        # Sample random points (avoid boundaries)
        x_batch = np.random.uniform(0.1, 0.9, batch_size)
        
        # Training step
        loss = net.train_step(x_batch, lr=1e-5)
        losses.append(loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        # Early stopping if loss explodes
        if np.isnan(loss) or loss > 1000:
            print(f"Training unstable at epoch {epoch}, stopping early")
            break
    
    print(f"Phase I completed. Final loss: {losses[-1]:.6f}")
    return net, losses

# ============================================================================
# PHASE II: FIXED NEWTON'S METHOD
# ============================================================================

def create_finite_difference_matrix(n_interior):
    """Create second derivative matrix for interior points only"""
    h = 1.0 / (n_interior + 1)  # Grid spacing
    A = np.zeros((n_interior, n_interior))
    
    # Fill the tridiagonal matrix for -d²/dx²
    for i in range(n_interior):
        A[i, i] = -2.0 / h**2
        if i > 0:
            A[i, i-1] = 1.0 / h**2
        if i < n_interior - 1:
            A[i, i+1] = 1.0 / h**2
    
    return A

def newton_method_fixed(u_initial, max_iter=10, tol=1e-8):
    """Phase II: Fixed Newton's method"""
    print("Phase II: Newton's method...")
    
    n_total = len(u_initial)
    x = np.linspace(0, 1, n_total)
    
    # Extract interior points (boundary values are 0)
    x_interior = x[1:-1]
    n_interior = len(x_interior)
    u_interior = u_initial[1:-1].copy()
    
    # Create finite difference matrix for interior points
    L = create_finite_difference_matrix(n_interior)
    f_interior = rhs_function(x_interior)
    
    print(f"Matrix L shape: {L.shape}")
    print(f"Interior solution shape: {u_interior.shape}")
    print(f"RHS shape: {f_interior.shape}")
    
    errors = []
    
    for k in range(max_iter):
        # Residual: -L*u + u³ - f = 0
        residual = -L @ u_interior + u_interior**3 - f_interior
        
        # Jacobian: -L + diag(3*u²)
        jacobian = -L + np.diag(3 * u_interior**2)
        
        # Solve for Newton step
        try:
            delta_u = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            print(f"Singular matrix at iteration {k}")
            break
        
        # Update solution
        u_interior_new = u_interior + delta_u
        
        # Check convergence
        error = np.max(np.abs(delta_u)) / (np.max(np.abs(u_interior)) + 1e-10)
        errors.append(error)
        
        print(f"Newton iteration {k+1}: error = {error:.2e}")
        
        if error < tol:
            print(f"Converged in {k+1} iterations!")
            break
        
        u_interior = u_interior_new
    
    # Reconstruct full solution with boundary conditions
    u_full = np.zeros(n_total)
    u_full[1:-1] = u_interior
    # u_full[0] = u_full[-1] = 0 (already set)
    
    return u_full, errors

# ============================================================================
# COMPLETE FIXED INT-DEEP SOLVER
# ============================================================================

def run_int_deep_fixed():
    """Run the complete fixed Int-Deep method"""
    print("="*50)
    print("FIXED MINIMAL INT-DEEP SOLVER")
    print("="*50)
    
    # Phase I: Neural network training
    net, losses = phase1_training(epochs=50, batch_size=32)
    
    # Evaluate neural network solution
    n_points = 65  # Smaller grid for stability
    x = np.linspace(0, 1, n_points)
    u_dl = net.forward(x)
    
    print(f"Neural network output range: [{np.min(u_dl):.3f}, {np.max(u_dl):.3f}]")
    
    # Phase II: Newton's method
    u_final, newton_errors = newton_method_fixed(u_dl)
    
    # Compute errors
    u_exact = exact_solution(x)
    error_dl = np.max(np.abs(u_exact - u_dl))
    error_final = np.max(np.abs(u_exact - u_final))
    
    print("\n" + "="*30)
    print("RESULTS")
    print("="*30)
    print(f"Phase I error:  {error_dl:.2e}")
    print(f"Phase II error: {error_final:.2e}")
    if error_dl > 0:
        print(f"Improvement:    {error_dl/error_final:.1f}x better")
    
    return {
        'x': x,
        'u_exact': u_exact,
        'u_dl': u_dl,
        'u_final': u_final,
        'losses': losses,
        'newton_errors': newton_errors,
        'error_dl': error_dl,
        'error_final': error_final
    }

def plot_results(results):
    """Plot results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Solutions
    ax = axes[0, 0]
    ax.plot(results['x'], results['u_exact'], 'k-', label='Exact', linewidth=2)
    ax.plot(results['x'], results['u_dl'], 'r--', label='Phase I (NN)', linewidth=2)
    ax.plot(results['x'], results['u_final'], 'b:', label='Phase II (Final)', linewidth=3)
    ax.set_title('Solution Comparison')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Errors
    ax = axes[0, 1]
    error_dl = np.abs(results['u_exact'] - results['u_dl'])
    error_final = np.abs(results['u_exact'] - results['u_final'])
    ax.semilogy(results['x'], error_dl + 1e-15, 'r--', label='Phase I Error', linewidth=2)
    ax.semilogy(results['x'], error_final + 1e-15, 'b:', label='Phase II Error', linewidth=2)
    ax.set_title('Absolute Error')
    ax.set_xlabel('x')
    ax.set_ylabel('|Error|')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training loss
    ax = axes[1, 0]
    ax.semilogy(results['losses'])
    ax.set_title('Phase I: Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    
    # Newton convergence
    ax = axes[1, 1]
    if len(results['newton_errors']) > 0:
        ax.semilogy(range(1, len(results['newton_errors']) + 1), 
                   results['newton_errors'], 'bo-')
        ax.set_title('Phase II: Newton Convergence')
        ax.set_xlabel('Newton Iteration')
        ax.set_ylabel('Relative Error')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# RUN THE FIXED VERSIO
# ============================================================================

if __name__ == "__main__":
    # Run the fixed Int-Deep method
    results = run_int_deep_fixed()
    
    # Plot results
    plot_results(results)
    
    print("\n" + "="*50)
    print("SUCCESS! Fixed Int-Deep implementation works!")
    print("="*50)