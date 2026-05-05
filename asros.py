import numpy as np
import matplotlib.pyplot as plt

def asmussen_rosinski_levy(alpha, T, n, epsilon):
    """
    Simulates a Levy process using Asmussen-Rosinski (2001) approximation.
    Small jumps (< epsilon) replaced by Brownian Motion.
    """
    dt = T / n
    t_axis = np.linspace(0, T, n+1)
    
    # 1. SMALL JUMP APPROXIMATION (Gaussian part)
    # For nu(dz) = |z|^-(1+alpha), the variance sigma^2(eps) is:
    # integral_{-eps}^{eps} z^2 * |z|^-(1+alpha) dz = (2 * eps^(2-alpha)) / (2-alpha)
    sigma_eps_sq = (2.0 * epsilon**(2.0 - alpha)) / (2.0 - alpha)
    std_eps = np.sqrt(sigma_eps_sq * dt)
    gaussian_increments = np.random.normal(0, std_eps, n)
    
    # 2. LARGE JUMP SIMULATION (Poisson part)
    # Intensity lambda = integral_{|z|>eps} nu(dz) = (2 * eps^-alpha) / alpha
    lam = (2.0 * epsilon**(-alpha)) / alpha
    num_jumps = np.random.poisson(lam * T)
    
    jump_times = np.random.uniform(0, T, num_jumps)
    # Jump sizes for |z| > eps: Inverse transform sampling
    u = np.random.uniform(0, 1, num_jumps)
    signs = np.random.choice([-1, 1], num_jumps)
    jump_sizes = signs * epsilon * (1 - u)**(-1.0/alpha)
    
    # Discretize jumps onto the time grid
    poisson_increments = np.zeros(n)
    indices = np.floor(jump_times / dt).astype(int)
    # Ensure indices don't exceed grid
    indices = np.clip(indices, 0, n-1)
    np.add.at(poisson_increments, indices, jump_sizes)
    
    # 3. COMBINE
    # Path = Cumsum(Small Jumps + Large Jumps)
    increments = gaussian_increments + poisson_increments
    path = np.concatenate(([0], np.cumsum(increments)))
    
    return t_axis, path

# Parameters
alpha = 1.5
T = 1.0
n = 1000
eps = 0.1

t, path = asmussen_rosinski_levy(alpha, T, n, eps)

plt.figure(figsize=(10, 5))
plt.plot(t, path, label=f'Asmussen-Rosinski (eps={eps})', color='blue', lw=1)
plt.title(f"Simulation of alpha-stable process (alpha={alpha})")
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.grid(True)
plt.legend()
plt.show()