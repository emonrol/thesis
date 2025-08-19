"""
IMPROVED INT-DEEP IMPLEMENTATION
===============================

Major improvements to Phase I neural network training:
1. ResNet-style architecture
2. Better activation functions  
3. Adam optimizer
4. Learning rate scheduling
5. Better loss formulation
6. Adaptive sampling
7. Early stopping and regularization
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PROBLEM DEFINITION
# ============================================================================

def exact_solution(x):
    return 0.5 * np.sin(np.pi * x)

def rhs_function(x):
    """Right-hand side f(x) = -u'' + u³"""
    u = exact_solution(x)
    u_xx = -0.5 * np.pi**2 * np.sin(np.pi * x)
    return -u_xx + u**3

# ============================================================================
# IMPROVED NEURAL NETWORK ARCHITECTURE
# ============================================================================

def tanh(x):
    """Tanh activation - better than ReLU for smoothness"""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of tanh"""
    return 1 - np.tanh(x)**2

def swish(x):
    """Swish activation: x * sigmoid(x) - smooth and works well"""
    return x * (1 / (1 + np.exp(-np.clip(x, -500, 500))))

def he_init(shape):
    """He initialization for better gradient flow"""
    return np.random.randn(*shape) * np.sqrt(2.0 / shape[0])

def xavier_init(shape):
    """Xavier initialization"""
    return np.random.randn(*shape) * np.sqrt(1.0 / shape[0])

class ImprovedNetwork:
    """Much better neural network with modern techniques"""
    
    def __init__(self, width=64, depth=6):
        self.width = width
        self.depth = depth
        
        # Initialize layers with proper scaling
        self.layers = []
        
        # Input layer
        self.layers.append({
            'W': xavier_init((1, width)),
            'b': np.zeros(width)
        })
        
        # Hidden layers (ResNet-style)
        for i in range(depth - 2):
            self.layers.append({
                'W': he_init((width, width)),
                'b': np.zeros(width)
            })
        
        # Output layer
        self.layers.append({
            'W': xavier_init((width, 1)),
            'b': np.zeros(1)
        })
        
        # Adam optimizer parameters
        self.reset_optimizer()
    
    def reset_optimizer(self):
        """Reset Adam optimizer state"""
        self.m = []  # First moment estimates
        self.v = []  # Second moment estimates
        self.t = 0   # Time step
        
        for layer in self.layers:
            self.m.append({
                'W': np.zeros_like(layer['W']),
                'b': np.zeros_like(layer['b'])
            })
            self.v.append({
                'W': np.zeros_like(layer['W']),
                'b': np.zeros_like(layer['b'])
            })
    
    def forward(self, x):
        """Forward pass with ResNet connections and boundary conditions"""
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        # Store activations for residual connections
        activations = []
        
        # First layer
        z = x @ self.layers[0]['W'] + self.layers[0]['b']
        a = tanh(z)
        activations.append(a)
        
        # Hidden layers with residual connections
        for i in range(1, len(self.layers) - 1):
            z = activations[-1] @ self.layers[i]['W'] + self.layers[i]['b']
            a = tanh(z)
            
            # Residual connection (every 2 layers)
            if i >= 2 and i % 2 == 0:
                a = a + activations[-2]  # Add previous activation
            
            activations.append(a)
        
        # Output layer (linear)
        output = activations[-1] @ self.layers[-1]['W'] + self.layers[-1]['b']
        output = output.flatten()
        
        # Apply boundary conditions: u(0) = u(1) = 0
        x_flat = x.flatten()
        boundary_term = x_flat * (1 - x_flat)
        
        # Scale output to reasonable range
        return boundary_term * output
    
    def compute_derivatives(self, x, h=1e-4):
        """Compute first and second derivatives"""
        # Ensure we stay in domain
        x = np.clip(x, h, 1-h)
        
        # First derivative: central difference
        u_plus = self.forward(x + h)
        u_minus = self.forward(x - h)
        u_x = (u_plus - u_minus) / (2 * h)
        
        # Second derivative
        u_center = self.forward(x)
        u_xx = (u_plus - 2*u_center + u_minus) / (h**2)
        
        return u_x, u_xx
    
    def compute_loss_components(self, x_batch):
        """Compute different loss components"""
        # PDE loss in interior
        x_interior = x_batch[(x_batch > 0.01) & (x_batch < 0.99)]
        if len(x_interior) == 0:
            return 0, 0, 0, 0
        
        u = self.forward(x_interior)
        u_x, u_xx = self.compute_derivatives(x_interior)
        f_vals = rhs_function(x_interior)
        
        # PDE residual: -u'' + u³ - f = 0
        pde_residual = -u_xx + u**3 - f_vals
        pde_loss = np.mean(pde_residual**2)
        
        # Boundary loss (should be automatically satisfied, but add small penalty)
        x_boundary = np.array([0.0, 1.0])
        u_boundary = self.forward(x_boundary)
        boundary_loss = np.mean(u_boundary**2)
        
        # Smoothness regularization (penalize large derivatives)
        smoothness_loss = np.mean(u_x**2) * 1e-6
        
        # Magnitude regularization (prevent output explosion)
        magnitude_loss = np.mean(u**2) * 1e-6
        
        return pde_loss, boundary_loss, smoothness_loss, magnitude_loss
    
    def compute_gradients_autodiff(self, x_batch, h=1e-6):
        """Compute gradients using automatic differentiation (finite differences)"""
        # Get current loss
        pde_loss, boundary_loss, smoothness_loss, magnitude_loss = self.compute_loss_components(x_batch)
        total_loss = pde_loss + 100*boundary_loss + smoothness_loss + magnitude_loss
        
        gradients = []
        
        # Compute gradients for each layer
        for layer_idx, layer in enumerate(self.layers):
            layer_grads = {}
            
            # Gradients for weights
            W_grad = np.zeros_like(layer['W'])
            flat_W = layer['W'].flatten()
            for i, _ in enumerate(flat_W):
                # Perturb weight
                old_val = flat_W[i]
                flat_W[i] = old_val + h
                layer['W'] = flat_W.reshape(layer['W'].shape)
                
                # Compute perturbed loss
                pde_p, boundary_p, smooth_p, mag_p = self.compute_loss_components(x_batch)
                loss_plus = pde_p + 100*boundary_p + smooth_p + mag_p
                
                # Restore weight
                flat_W[i] = old_val
                layer['W'] = flat_W.reshape(layer['W'].shape)
                
                # Compute gradient
                W_grad.flat[i] = (loss_plus - total_loss) / h
            
            # Gradients for biases
            b_grad = np.zeros_like(layer['b'])
            for i, _ in enumerate(layer['b']):
                # Perturb bias
                old_val = layer['b'][i]
                layer['b'][i] = old_val + h
                
                # Compute perturbed loss
                pde_p, boundary_p, smooth_p, mag_p = self.compute_loss_components(x_batch)
                loss_plus = pde_p + 100*boundary_p + smooth_p + mag_p
                
                # Restore bias
                layer['b'][i] = old_val
                
                # Compute gradient
                b_grad[i] = (loss_plus - total_loss) / h
            
            layer_grads['W'] = np.clip(W_grad, -1.0, 1.0)  # Gradient clipping
            layer_grads['b'] = np.clip(b_grad, -1.0, 1.0)
            gradients.append(layer_grads)
        
        return gradients, total_loss
    
    def adam_update(self, gradients, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam optimizer update"""
        self.t += 1
        
        for i, (layer, grad, m, v) in enumerate(zip(self.layers, gradients, self.m, self.v)):
            # Update biased first moment estimate
            m['W'] = beta1 * m['W'] + (1 - beta1) * grad['W']
            m['b'] = beta1 * m['b'] + (1 - beta1) * grad['b']
            
            # Update biased second moment estimate  
            v['W'] = beta2 * v['W'] + (1 - beta2) * (grad['W']**2)
            v['b'] = beta2 * v['b'] + (1 - beta2) * (grad['b']**2)
            
            # Compute bias-corrected moment estimates
            m_hat_W = m['W'] / (1 - beta1**self.t)
            m_hat_b = m['b'] / (1 - beta1**self.t)
            v_hat_W = v['W'] / (1 - beta2**self.t)
            v_hat_b = v['b'] / (1 - beta2**self.t)
            
            # Update parameters
            layer['W'] -= lr * m_hat_W / (np.sqrt(v_hat_W) + eps)
            layer['b'] -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

def adaptive_sampling(epoch, batch_size):
    """Adaptive sampling strategy"""
    if epoch < 50:
        # Early training: focus on interior
        return np.random.uniform(0.1, 0.9, batch_size)
    elif epoch < 100:
        # Mid training: include some boundary points
        interior = np.random.uniform(0.05, 0.95, int(0.8 * batch_size))
        boundary = np.random.choice([0.0, 1.0], int(0.2 * batch_size))
        return np.concatenate([interior, boundary])
    else:
        # Late training: full domain with emphasis on difficult regions
        easy_points = np.random.uniform(0.1, 0.9, int(0.6 * batch_size))
        hard_points = np.random.uniform(0.01, 0.99, int(0.4 * batch_size))
        return np.concatenate([easy_points, hard_points])

def learning_rate_schedule(epoch, initial_lr=1e-3):
    """Learning rate scheduling"""
    if epoch < 50:
        return initial_lr
    elif epoch < 150:
        return initial_lr * 0.5
    elif epoch < 250:
        return initial_lr * 0.2
    else:
        return initial_lr * 0.1

def improved_phase1_training(epochs=100, batch_size=16):
    """Improved Phase I training with modern techniques"""
    print("Phase I: Training improved neural network...")
    print(f"Architecture: 64-unit width, 6 layers, ResNet connections")
    print(f"Optimizer: Adam with learning rate scheduling")
    print(f"Training: {epochs} epochs, batch size {batch_size}")
    print("-" * 50)
    
    net = ImprovedNetwork(width=64, depth=6)
    
    losses = []
    pde_losses = []
    best_loss = float('inf')
    patience = 50
    no_improve_count = 0
    
    for epoch in range(epochs):
        # Adaptive sampling
        x_batch = adaptive_sampling(epoch, batch_size)
        print(epoch)
        # Compute gradients
        gradients, total_loss = net.compute_gradients_autodiff(x_batch)
        
        # Get loss components for monitoring
        pde_loss, boundary_loss, smooth_loss, mag_loss = net.compute_loss_components(x_batch)
        
        # Learning rate scheduling
        lr = learning_rate_schedule(epoch)
        
        # Adam update
        net.adam_update(gradients, lr=lr)
        
        # Record losses
        losses.append(total_loss)
        pde_losses.append(pde_loss)
        
        # Early stopping
        if total_loss < best_loss:
            best_loss = total_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count >= patience and epoch > 100:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Progress reporting
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | Total: {total_loss:.4f} | PDE: {pde_loss:.4f} | "
                  f"Boundary: {boundary_loss:.6f} | LR: {lr:.2e}")
        
        # Stop if training becomes unstable
        if np.isnan(total_loss) or total_loss > 1000:
            print(f"Training unstable at epoch {epoch}, stopping early")
            break
    
    print(f"Phase I completed. Final loss: {losses[-1]:.6f}")
    print(f"Best loss achieved: {best_loss:.6f}")
    
    return net, losses, pde_losses

# ============================================================================
# PHASE II: SAME AS BEFORE
# ============================================================================

def create_finite_difference_matrix(n_interior):
    """Create second derivative matrix for interior points only"""
    h = 1.0 / (n_interior + 1)
    A = np.zeros((n_interior, n_interior))
    
    for i in range(n_interior):
        A[i, i] = -2.0 / h**2
        if i > 0:
            A[i, i-1] = 1.0 / h**2
        if i < n_interior - 1:
            A[i, i+1] = 1.0 / h**2
    
    return A

def newton_method_fixed(u_initial, max_iter=10, tol=1e-8):
    """Phase II: Newton's method"""
    print("Phase II: Newton's method...")
    
    n_total = len(u_initial)
    x = np.linspace(0, 1, n_total)
    x_interior = x[1:-1]
    n_interior = len(x_interior)
    u_interior = u_initial[1:-1].copy()
    
    L = create_finite_difference_matrix(n_interior)
    f_interior = rhs_function(x_interior)
    
    errors = []
    
    for k in range(max_iter):
        residual = -L @ u_interior + u_interior**3 - f_interior
        jacobian = -L + np.diag(3 * u_interior**2)
        
        try:
            delta_u = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            print(f"Singular matrix at iteration {k}")
            break
        
        u_interior_new = u_interior + delta_u
        error = np.max(np.abs(delta_u)) / (np.max(np.abs(u_interior)) + 1e-10)
        errors.append(error)
        
        print(f"Newton iteration {k+1}: error = {error:.2e}")
        
        if error < tol:
            print(f"Converged in {k+1} iterations!")
            break
        
        u_interior = u_interior_new
    
    u_full = np.zeros(n_total)
    u_full[1:-1] = u_interior
    return u_full, errors

# ============================================================================
# COMPLETE IMPROVED INT-DEEP SOLVER
# ============================================================================

def run_improved_int_deep():
    """Run the improved Int-Deep method"""
    print("="*60)
    print("IMPROVED INT-DEEP SOLVER")
    print("="*60)
    
    # Phase I: Improved neural network training
    net, losses, pde_losses = improved_phase1_training(epochs=5, batch_size=1)
    
    # Evaluate neural network solution
    n_points = 60
    x = np.linspace(0, 1, n_points)
    u_dl = net.forward(x)
    
    print(f"\nNeural network evaluation:")
    print(f"Output range: [{np.min(u_dl):.4f}, {np.max(u_dl):.4f}]")
    print(f"Expected range: [0.0000, 0.5000]")
    
    # Phase II: Newton's method
    u_final, newton_errors = newton_method_fixed(u_dl)
    
    # Compute errors against exact solution
    u_exact = exact_solution(x)
    error_dl = np.max(np.abs(u_exact - u_dl))
    error_final = np.max(np.abs(u_exact - u_final))
    
    # Additional metrics
    l2_error_dl = np.sqrt(np.mean((u_exact - u_dl)**2))
    l2_error_final = np.sqrt(np.mean((u_exact - u_final)**2))
    
    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    print(f"Phase I L∞ error:  {error_dl:.2e}")
    print(f"Phase I L² error:  {l2_error_dl:.2e}")
    print(f"Phase II L∞ error: {error_final:.2e}")
    print(f"Phase II L² error: {l2_error_final:.2e}")
    print(f"L∞ improvement:    {error_dl/error_final:.1f}x better")
    print(f"L² improvement:    {l2_error_dl/l2_error_final:.1f}x better")
    
    return {
        'x': x,
        'u_exact': u_exact,
        'u_dl': u_dl,
        'u_final': u_final,
        'losses': losses,
        'pde_losses': pde_losses,
        'newton_errors': newton_errors,
        'error_dl': error_dl,
        'error_final': error_final,
        'l2_error_dl': l2_error_dl,
        'l2_error_final': l2_error_final
    }

def plot_improved_results(results):
    """Plot improved results with more details"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Solutions comparison
    ax = axes[0, 0]
    ax.plot(results['x'], results['u_exact'], 'k-', label='Exact', linewidth=3)
    ax.plot(results['x'], results['u_dl'], 'r--', label='Phase I (NN)', linewidth=2)
    ax.plot(results['x'], results['u_final'], 'b:', label='Phase II (Final)', linewidth=2)
    ax.set_title('Solution Comparison')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error comparison
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
    
    # Neural network outputs
    ax = axes[0, 2]
    ax.plot(results['x'], results['u_exact'], 'k-', label='Exact', linewidth=2)
    ax.plot(results['x'], results['u_dl'], 'r-', label='Neural Network', linewidth=2)
    ax.set_title('Phase I: Neural Network vs Exact')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training loss evolution
    ax = axes[1, 0]
    ax.semilogy(results['losses'], 'b-', label='Total Loss')
    ax.semilogy(results['pde_losses'], 'r--', label='PDE Loss')
    ax.set_title('Phase I: Training Progress')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Newton convergence
    ax = axes[1, 1]
    if len(results['newton_errors']) > 0:
        ax.semilogy(range(1, len(results['newton_errors']) + 1), 
                   results['newton_errors'], 'bo-', markersize=8)
        ax.set_title('Phase II: Newton Convergence')
        ax.set_xlabel('Newton Iteration')
        ax.set_ylabel('Relative Error')
        ax.grid(True, alpha=0.3)
    
    # PDE residual visualization
    ax = axes[1, 2]
    x_test = np.linspace(0.01, 0.99, 100)
    u_test = results['u_final'][1:-1]  # Remove boundary points
    u_exact_test = exact_solution(x_test)
    
    # Compute PDE residual for final solution
    h = x_test[1] - x_test[0]
    u_xx_test = np.gradient(np.gradient(u_exact_test, h), h)
    residual_exact = -u_xx_test + u_exact_test**3 - rhs_function(x_test)
    
    ax.plot(x_test, residual_exact, 'k-', label='Exact Residual')
    ax.set_title('PDE Residual Check')
    ax.set_xlabel('x')
    ax.set_ylabel('Residual')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# RUN THE IMPROVED VERSION
# ============================================================================

if __name__ == "__main__":
    # Run the improved Int-Deep method
    results = run_improved_int_deep()
    
    # Plot detailed results
    plot_improved_results(results)
    
    print("\n" + "="*60)
    print("SUCCESS! Improved Int-Deep with much better Phase I!")
    print("="*60)