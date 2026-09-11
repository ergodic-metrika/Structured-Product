import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class BearCertificate:
    name: str = "03732B Yuanta TAIEX Bear"
    K: float = 50122.62
    H: float = 47877.91
    r: float = 0.017

def simulate_knockout(contract, S0, sigma, T, n_paths=50_000, seed=42):
    if T <= 0:
        return 1.0 if S0 >= contract.H else 0.0

    n_steps = max(int(T * 252), 1)
    dt = T / n_steps
    drift = (contract.r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths, n_steps))
    logS = np.log(S0) + np.cumsum(drift + vol * Z, axis=1)
    S_paths = np.exp(logS)

    return np.any(S_paths >= contract.H, axis=1).mean()

# ============================================================
# Daily Inputs (change these)
# ============================================================
current_S = 46054      # Latest TAIEX
current_T = 166 / 365     # Remaining time
sigma = 0.32            # Volatility
# ============================================================

contract = BearCertificate()

# Calculate KO probabilities across a range
S_range = np.linspace(contract.H * 0.85, contract.H * 1.05, 35)
ko_probs = [simulate_knockout(contract, S, sigma, current_T) for S in S_range]

# Calculate KO probability at your current level
ko_current = simulate_knockout(contract, current_S, sigma, current_T)

# ---- Single Plot ----
plt.close('all')
fig, ax = plt.subplots(figsize=(10, 6))

# Main curve
ax.plot(S_range, ko_probs, 'b-o', lw=2, markersize=5, label='KO Probability')

# Barrier and Strike lines
ax.axvline(contract.H, color='red', ls='--', lw=1.8, label=f'Barrier = {contract.H:,.0f}')
ax.axvline(contract.K, color='green', ls='--', lw=1.8, label=f'Strike = {contract.K:,.0f}')

# Star indicating your current position
ax.plot(current_S, ko_current, 
        marker='*', markersize=22, color='orange', 
        markeredgecolor='black', markeredgewidth=0.8,
        label=f'You are here (S={current_S:,.0f})', zorder=5)

ax.set_title(f'Knock-out Probability vs Index Level\n'
             f'{contract.name}  |  Vol = {sigma:.0%}  |  T = {current_T*365:.0f} days')
ax.set_xlabel('TAIEX Level')
ax.set_ylabel('Probability of Knock-out')
ax.set_ylim(-0.02, 1.05)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()