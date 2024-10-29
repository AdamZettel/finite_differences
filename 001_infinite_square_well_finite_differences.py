import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0         # Length of the well (arbitrary units)
N = 1000        # Number of points for discretization
dx = L / (N - 1) # Spatial step size
hbar = 1.0      # Planck's constant (reduced, in arbitrary units)
m = 1.0         # Mass of the particle (arbitrary units)

# Discretized x-axis
x = np.linspace(0, L, N)

# Finite difference matrix for the second derivative
diagonal = np.full(N, -2.0) / dx**2
off_diagonal = np.full(N - 1, 1.0) / dx**2
H = np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)

# Solve the eigenvalue problem
energies, wavefunctions = np.linalg.eigh(-H * hbar**2 / (2 * m))

# Analytical energies for comparison
n_values = np.arange(1, 5)  # Compare first 4 levels
analytical_energies = [(n**2 * np.pi**2 * hbar**2) / (2 * m * L**2) for n in n_values]

# Ensure wavefunctions start positive for consistent plotting
for i in range(len(wavefunctions[0])):
    if wavefunctions[0, i] < 0:
        wavefunctions[:, i] = -wavefunctions[:, i]

# Plot computed wavefunctions and energies
num_levels = 4
plt.figure(figsize=(10, 6))
for i in range(num_levels):
    plt.plot(x, wavefunctions[:, i], label=f"n={i+1}, Computed E={energies[i]:.2f}, Analytical E={analytical_energies[i]:.2f}")

plt.title("Wavefunctions in the Infinite Square Well")
plt.xlabel("Position x")
plt.ylabel("Wavefunction ψ(x)")
plt.legend()
plt.show()

# Print both computed and analytical energy levels
print("Computed vs Analytical Energy Levels:")
for i in range(num_levels):
    print(f"Level {i+1}: Computed E = {energies[i]:.4f}, Analytical E = {analytical_energies[i]:.4f}")
