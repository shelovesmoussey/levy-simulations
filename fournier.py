import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 1. Missing Function: Generates the stable process increments
def stable_increments(alpha, dt, size, scale):
    U = np.random.uniform(-np.pi/2, np.pi/2, size)
    W = np.random.exponential(1, size)
    X = (np.sin(alpha * U) / (np.cos(U) ** (1/alpha))) * \
        ((np.cos(U - alpha * U) / W) ** ((1 - alpha) / alpha))
    return scale * (dt**(1/alpha)) * X

# 2. Scale calibration for the Levy measure nu(dz) = |z|^-(1+alpha)
def get_fournier_scale(alpha):
    from scipy.special import gamma as gamma_func
    # Constant to align standard stable generator with paper's nu(dz)
    scale_pow_alpha = (2 * gamma_func(1 - alpha) * np.cos(np.pi * alpha / 2)) / (1 - alpha)
    return np.abs(scale_pow_alpha)**(1/alpha)

# 3. Euler Simulation
def simulate_sde(sigma_func, alpha, T, n, M, epsilon=None, method='gaussian'):
    dt = T / n
    X = np.ones(M) # Start at X0 = 1.0 
    scale = get_fournier_scale(alpha)
    
    if epsilon:
        # Theoretical variance of neglected jumps
        sigma_eps_sq = (2.0 * (epsilon**(2.0 - alpha))) / (2.0 - alpha)
        std_eps = np.sqrt(sigma_eps_sq * dt)

    for _ in range(n):
        if epsilon is None:
            dW = stable_increments(alpha, dt, M, scale)
        else:
            lam = (2.0 * (epsilon**(-alpha))) / alpha
            num_jumps = np.random.poisson(lam * dt, M)
            total_jumps = np.sum(num_jumps)
            
            large_jumps = np.zeros(M)
            if total_jumps > 0:
                u = np.random.uniform(0, 1, total_jumps)
                signs = np.random.choice([-1, 1], total_jumps)
                jumps = signs * epsilon * (1.0 - u)**(-1.0/alpha)
                indices = np.repeat(np.arange(M), num_jumps)
                np.add.at(large_jumps, indices, jumps)
            
            if method == 'gaussian':
                # Large jumps + Gaussian noise scaled appropriately
                dW = large_jumps + (np.random.normal(0, std_eps, M) * scale)
            else:
                # Neglecting small jumps entirely
                dW = large_jumps
                
        X = X + sigma_func(X) * dW
    return X

# 4. Comparison and Plotting
def run_comparison():
    alpha, T, n, M = 1.5, 1.0, 1000, 40000
    sigma1 = lambda y: (1 + y**2) / (1 + y**4)
    eps = 0.1 # Large enough to see the mismatch on the left
    
    print("Simulating True vs. Approximations...")
    V_true = simulate_sde(sigma1, alpha, T, n, M)
    V_neg = simulate_sde(sigma1, alpha, T, n, M, epsilon=eps, method='neglect')
    V_gauss = simulate_sde(sigma1, alpha, T, n, M, epsilon=eps, method='gaussian')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(rf"Fournier Figure 1 Replication ($\epsilon = {eps}$)", fontsize=14)
    
    x_range = np.linspace(0, 6, 200)

    # Plotting Neglected (Left)
    ax1.hist(V_true, bins=100, range=(0, 6), density=True, color='white', edgecolor='lightgray', label='True')
    kde_neg = gaussian_kde(V_neg[(V_neg > -2) & (V_neg < 10)], bw_method=0.1)
    ax1.plot(x_range, kde_neg(x_range), color='black', lw=1.5)
    ax1.set_title("Neglecting Small Jumps (Shift Visible)")
    
    # Plotting Gaussian (Right)
    ax2.hist(V_true, bins=100, range=(0, 6), density=True, color='white', edgecolor='lightgray', label='True')
    kde_gauss = gaussian_kde(V_gauss[(V_gauss > -2) & (V_gauss < 10)], bw_method=0.1)
    ax2.plot(x_range, kde_gauss(x_range), color='black', lw=1.5)
    ax2.set_title("Gaussian Approximation (Correct Fit)")
    
    plt.show()

if __name__ == "__main__":
    run_comparison()