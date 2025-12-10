import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import odeint
import scipy.optimize
import sympy as sp
import pandas as pd
import copy

try:
    import casadi
except:
    pass

try:
    import do_mpc
except:
    pass

try:
    import pybounds
except:
    pass

import sys
import requests
import importlib

def import_local_or_github(package_name, function_name=None, directory=None, giturl=None):
    try:
        if directory is not None:
            if directory not in sys.path:
                sys.path.append(directory)
        package = importlib.import_module(package_name)
        if function_name is not None:
            function = getattr(package, function_name)
            return function
        else:
            return package
    except:
        if giturl is None:
            giturl = 'https://raw.githubusercontent.com/Ademola-Olorunnisola/Nonlinear_and_Data_Driven_Estimation/main/Utility/' + str(package_name) + '.py'
        r = requests.get(giturl)
        print('Fetching from: ')
        print(r)
        with open(package_name+'.py', 'w') as f:
            f.write(r.text)
        f.close()
        package = importlib.import_module(package_name)
        if function_name is not None:
            function = getattr(package , function_name)
            return function
        else:
            return package

tb_simulations = import_local_or_github('tb_simulations', directory='../Utility')
plot_tme = import_local_or_github('plot_utility', 'plot_tme', directory='../Utility')

f = tb_simulations.F().f
h_a = tb_simulations.H('h_ivr').h
h_b = tb_simulations.H('h_all_svir').h
h_c = tb_simulations.H('h_all_with_params').h

# Choose measurement function
h = h_c

print("Measurement names:", h(None, None, return_measurement_names=True))

############################################################################################
# IMPROVED CONTROL STRATEGIES FOR BETTER OBSERVABILITY
############################################################################################

def u_func_original(x_vec, t):
    """Original simple threshold-based controller"""
    I = x_vec[2]
    I_threshold = 0.005
    if I > I_threshold:
        alpha = 0.01
        kappa = 0.3
    else:
        alpha = 0.005
        kappa = 0.1
    return np.array([alpha, kappa])


def u_func_sinusoidal(x_vec, t, freq_alpha=0.05, freq_kappa=0.08):
    """
    Sinusoidal control inputs - provides persistent excitation
    This helps with observability by continuously varying the inputs
    """
    I = x_vec[2]
    
    # Base levels
    alpha_base = 0.007
    kappa_base = 0.2
    
    # Add sinusoidal variations
    alpha = alpha_base + 0.003 * np.sin(2 * np.pi * freq_alpha * t / 365)
    kappa = kappa_base + 0.1 * np.sin(2 * np.pi * freq_kappa * t / 365)
    
    # React to high infections
    if I > 0.005:
        alpha += 0.005
        kappa += 0.1
    
    # Keep within bounds
    alpha = np.clip(alpha, 0, 0.05)
    kappa = np.clip(kappa, 0, 0.5)
    
    return np.array([alpha, kappa])


def u_func_pulse_train(x_vec, t, period=60, pulse_width=15):
    """
    Pulse train control - alternates between high and low control
    Good for distinguishing system dynamics
    """
    I = x_vec[2]
    
    # Determine if we're in a pulse
    cycle_position = t % period
    in_pulse = cycle_position < pulse_width
    
    if in_pulse:
        alpha = 0.015  # High vaccination during pulse
        kappa = 0.4    # High social distancing during pulse
    else:
        alpha = 0.003  # Low vaccination between pulses
        kappa = 0.05   # Low social distancing between pulses
    
    # Emergency response to very high infections
    if I > 0.01:
        alpha = 0.02
        kappa = 0.5
    
    return np.array([alpha, kappa])


def u_func_chirp(x_vec, t, t_max=365):
    """
    Chirp signal - frequency increases over time
    Excellent for system identification and observability
    """
    I = x_vec[2]
    
    # Chirp parameters: frequency increases from f0 to f1
    f0 = 0.02  # Start frequency (cycles per year)
    f1 = 0.2   # End frequency (cycles per year)
    
    # Linear chirp
    freq = f0 + (f1 - f0) * t / t_max
    
    # Base levels with chirp modulation
    alpha_base = 0.008
    kappa_base = 0.2
    
    alpha = alpha_base + 0.004 * np.sin(2 * np.pi * freq * t / 365)
    kappa = kappa_base + 0.15 * np.sin(2 * np.pi * freq * t / 365 + np.pi/4)  # Phase shift
    
    # React to infections
    if I > 0.005:
        alpha += 0.003
        kappa += 0.08
    
    # Keep within bounds
    alpha = np.clip(alpha, 0, 0.05)
    kappa = np.clip(kappa, 0, 0.5)
    
    return np.array([alpha, kappa])


def u_func_prbs(x_vec, t, switch_time=20, seed=42):
    """
    Pseudo-Random Binary Sequence (PRBS)
    Excellent for system identification - random but deterministic
    """
    np.random.seed(seed)
    
    # Generate sequence at start
    n_switches = int(365 / switch_time) + 1
    if not hasattr(u_func_prbs, 'alpha_seq'):
        u_func_prbs.alpha_seq = np.random.choice([0.003, 0.008, 0.015], size=n_switches)
        u_func_prbs.kappa_seq = np.random.choice([0.05, 0.2, 0.35], size=n_switches)
    
    # Get current values
    idx = min(int(t / switch_time), n_switches - 1)
    alpha = u_func_prbs.alpha_seq[idx]
    kappa = u_func_prbs.kappa_seq[idx]
    
    # Emergency override
    I = x_vec[2]
    if I > 0.01:
        alpha = 0.02
        kappa = 0.5
    
    return np.array([alpha, kappa])


def u_func_multisine(x_vec, t):
    """
    Multi-sine input - sum of multiple sinusoids at different frequencies
    Excellent for exciting multiple modes of the system
    """
    I = x_vec[2]
    
    # Multiple frequencies (in cycles per year)
    freqs_alpha = [0.03, 0.07, 0.13, 0.19]
    freqs_kappa = [0.04, 0.09, 0.15, 0.21]
    
    # Base levels
    alpha_base = 0.008
    kappa_base = 0.2
    
    # Sum multiple sinusoids
    alpha_osc = sum(0.001 * np.sin(2 * np.pi * f * t / 365) for f in freqs_alpha)
    kappa_osc = sum(0.03 * np.sin(2 * np.pi * f * t / 365) for f in freqs_kappa)
    
    alpha = alpha_base + alpha_osc
    kappa = kappa_base + kappa_osc
    
    # React to infections
    if I > 0.005:
        alpha += 0.004
        kappa += 0.1
    
    # Keep within bounds
    alpha = np.clip(alpha, 0, 0.05)
    kappa = np.clip(kappa, 0, 0.5)
    
    return np.array([alpha, kappa])


def u_func_stepped(x_vec, t, n_steps=6):
    """
    Stepped control - changes levels every ~60 days
    Good for comparing different operating conditions
    """
    I = x_vec[2]
    
    # Define control levels for each step
    step_duration = 365 / n_steps
    step = int(t / step_duration)
    
    alpha_levels = [0.003, 0.006, 0.012, 0.009, 0.015, 0.007]
    kappa_levels = [0.1, 0.15, 0.3, 0.25, 0.35, 0.2]
    
    alpha = alpha_levels[min(step, n_steps-1)]
    kappa = kappa_levels[min(step, n_steps-1)]
    
    # Emergency override
    if I > 0.01:
        alpha = 0.02
        kappa = 0.5
    
    return np.array([alpha, kappa])


############################################################################################
# SIMULATION FUNCTION
############################################################################################

def simulate_with_control_strategy(u_func, strategy_name, x0, t_sim):
    """Run simulation with given control strategy"""
    
    def f_ode(x_vec, t):
        u_vec = u_func(x_vec, t)
        return f(x_vec, u_vec)
    
    print(f"\n{'='*60}")
    print(f"Running simulation: {strategy_name}")
    print(f"{'='*60}")
    
    result = odeint(f_ode, x0, t_sim)
    
    x_sim = {
        'S': result[:, 0],
        'V': result[:, 1],
        'I': result[:, 2],
        'R': result[:, 3],
        'beta': result[:, 4],
        'sigma': result[:, 5]
    }
    
    u_sim = {'alpha': [], 'kappa': []}
    for i in range(len(t_sim)):
        u = u_func(result[i], t_sim[i])
        u_sim['alpha'].append(u[0])
        u_sim['kappa'].append(u[1])
    
    return x_sim, u_sim


############################################################################################
# MAIN SIMULATION
############################################################################################

N = 223000000
x0 = np.array([
    52309000/N,      # S
    158330000/N,     # V
    361000/N,        # I
    12000000/N,      # R
    0.00003,         # beta
    0.85             # sigma
])

t_sim = np.arange(0, 365, 1.0)

# Dictionary of control strategies
control_strategies = {
    'original': (u_func_original, "Original Threshold-Based"),
    'sinusoidal': (u_func_sinusoidal, "Sinusoidal (Persistent Excitation)"),
    'pulse_train': (u_func_pulse_train, "Pulse Train"),
    'chirp': (u_func_chirp, "Chirp (Frequency Sweep)"),
    'prbs': (u_func_prbs, "PRBS (Pseudo-Random Binary)"),
    'multisine': (u_func_multisine, "Multi-Sine"),
    'stepped': (u_func_stepped, "Stepped Levels")
}

# SELECT STRATEGY HERE - Change this to try different strategies
SELECTED_STRATEGY = 'chirp'  # Options: 'original', 'sinusoidal', 'pulse_train', 'chirp', 'prbs', 'multisine', 'stepped'

u_func_selected, strategy_name = control_strategies[SELECTED_STRATEGY]
x_sim, u_sim = simulate_with_control_strategy(u_func_selected, strategy_name, x0, t_sim)

# Plot state trajectories
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes = axes.flatten()

state_names = ['S', 'V', 'I', 'R', 'beta', 'sigma']
for i, state in enumerate(state_names):
    axes[i].plot(t_sim, x_sim[state], 'b-', linewidth=2)
    axes[i].set_xlabel('Time (days)')
    axes[i].set_ylabel(state)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_title(f'{state} - {strategy_name}')

plt.tight_layout()
plt.savefig('/home/claude/state_trajectories.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot control inputs
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

axes[0].plot(t_sim, u_sim['alpha'], 'g-', linewidth=2)
axes[0].set_xlabel('Time (days)')
axes[0].set_ylabel('Alpha (vaccination rate)')
axes[0].grid(True, alpha=0.3)
axes[0].set_title(f'Control: Alpha - {strategy_name}')

axes[1].plot(t_sim, u_sim['kappa'], 'r-', linewidth=2)
axes[1].set_xlabel('Time (days)')
axes[1].set_ylabel('Kappa (social distancing)')
axes[1].grid(True, alpha=0.3)
axes[1].set_title(f'Control: Kappa - {strategy_name}')

plt.tight_layout()
plt.savefig('/home/claude/control_inputs.png', dpi=150, bbox_inches='tight')
plt.show()

############################################################################################
# OBSERVABILITY ANALYSIS
############################################################################################

print("\n" + "="*60)
print("OBSERVABILITY ANALYSIS")
print("="*60)

x_sim_df = pd.DataFrame(x_sim)
u_sim_df = pd.DataFrame(u_sim)

state_names = f(None, None, return_state_names=True)
input_names = ['alpha', 'kappa']
measurement_names = h(None, None, return_measurement_names=True)

simulator = pybounds.Simulator(
    f,
    h,
    dt=1.0,
    state_names=state_names,
    input_names=input_names,
    measurement_names=measurement_names,
    mpc_horizon=10
)

w = 7  # window size

# Construct observability matrix in sliding windows
SEOM = pybounds.SlidingEmpiricalObservabilityMatrix(simulator, t_sim, x_sim, u_sim, w=w, eps=1e-4)
O_sliding = SEOM.get_observability_matrix()

n_window = len(O_sliding)
print(f"\nNumber of windows: {n_window}")

# Define measurement noise
measurement_names = h(None, None, return_measurement_names=True)
measurement_noise_stds = {name: 1.0 for name in measurement_names}
measurement_noise_vars = {key: val**2 for key, val in measurement_noise_stds.items()}

# Choose sensors and states
o_sensors = h(None, None, return_measurement_names=True)
o_states = ['S', 'V', 'I', 'R', 'beta', 'sigma']
window_size = 7
o_time_steps = np.arange(0, window_size, step=1)
o_measurement_noise_vars = {key: measurement_noise_vars[key] for key in o_sensors}

# Compute Fisher information
SFO = pybounds.SlidingFisherObservability(
    SEOM.O_df_sliding, 
    time=SEOM.t_sim, 
    lam=1e-8, 
    R=o_measurement_noise_vars,
    states=o_states, 
    sensors=o_sensors, 
    time_steps=o_time_steps, 
    w=None
)

EV_aligned = SFO.get_minimum_error_variance()
EV_no_nan = EV_aligned.fillna(method='bfill').fillna(method='ffill')

# Print summary statistics
print("\n" + "="*60)
print("OBSERVABILITY SUMMARY")
print("="*60)
print(f"Strategy: {strategy_name}")
print(f"\nMean Minimum Error Variance:")
for state in o_states:
    mean_ev = EV_no_nan[state].mean()
    print(f"  {state}: {mean_ev:.6e}")

print(f"\nMedian Minimum Error Variance:")
for state in o_states:
    median_ev = EV_no_nan[state].median()
    print(f"  {state}: {median_ev:.6e}")

# Plot observability results
states = list(SFO.FO[0].O.columns)
n_state = len(states)
fig, ax = plt.subplots(n_state, 2, figsize=(12, n_state*2), dpi=150, sharex=True)
ax = np.atleast_2d(ax)
cmap = 'inferno_r'

min_ev = np.min(EV_no_nan.iloc[:, 2:].values)
max_ev = np.max(EV_no_nan.iloc[:, 2:].values)
log_tick_high = int(np.ceil(np.log10(max_ev)))
log_tick_low = int(np.floor(np.log10(min_ev)))
cnorm = mpl.colors.LogNorm(10**log_tick_low, 10**log_tick_high)

from matplotlib.ticker import FuncFormatter
def fmt(x, pos):
    return f'{x:.2e}'

for n, state_name in enumerate(states):
    # Left panel: state trajectory colored by observability
    pybounds.colorline(t_sim, x_sim[state_name], EV_no_nan[state_name].values,
                       ax=ax[n, 0], cmap=cmap, norm=cnorm)
    
    # Right panel: minimum error variance over time
    pybounds.colorline(t_sim, EV_no_nan[state_name].values, EV_no_nan[state_name].values,
                       ax=ax[n, 1], cmap=cmap, norm=cnorm)
    
    # Colorbar
    cax = ax[n, -1].inset_axes([1.03, 0.0, 0.04, 1.0])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap), cax=cax,
                        ticks=np.logspace(log_tick_low, log_tick_high,
                                         log_tick_high-log_tick_low + 1))
    cbar.set_label('min. EV: ' + state_name, rotation=270, fontsize=7, labelpad=8)
    cbar.ax.tick_params(labelsize=6)
    
    # Left panel formatting
    x_vals = x_sim[state_name]
    x_range = np.max(x_vals) - np.min(x_vals)
    if x_range < 1e-4:
        x_mean = np.mean(x_vals)
        padding = max(x_range * 0.5, 1e-5)
        ax[n, 0].set_ylim(x_mean - padding, x_mean + padding)
    else:
        x_max = np.max(x_vals)
        x_min = np.min(x_vals)
        padding = x_range * 0.1
        ax[n, 0].set_ylim(x_min - padding, x_max + padding)
    ax[n, 0].set_ylabel('state: ' + state_name, fontsize=7)
    ax[n, 0].yaxis.set_major_formatter(FuncFormatter(fmt))
    ax[n, 0].grid(True, alpha=0.3)
    
    # Right panel formatting
    ax[n, 1].set_ylim(10**log_tick_low, 10**log_tick_high)
    ax[n, 1].set_yscale('log')
    ax[n, 1].set_ylabel('min. EV: ' + state_name, fontsize=7)
    ax[n, 1].set_yticks(np.logspace(log_tick_low, log_tick_high,
                                    log_tick_high-log_tick_low + 1))
    ax[n, 1].grid(True, alpha=0.3, which='both')

for a in ax.flat:
    a.tick_params(axis='both', labelsize=6)
    a.set_xlabel('time (days)', fontsize=7)
    offset = t_sim[-1] * 0.05
    a.set_xlim(-offset, t_sim[-1] + offset)

fig.suptitle(f'Observability Analysis - {strategy_name}', fontsize=10, y=0.995)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.4)
plt.savefig('/home/claude/observability_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n{'='*60}")
print("Analysis complete! Check the plots above.")
print(f"{'='*60}")
