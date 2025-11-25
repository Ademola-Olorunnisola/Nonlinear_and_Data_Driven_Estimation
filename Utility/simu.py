# -*- coding: utf-8 -*-
"""tb_sveir_simulations.py - SVEIR Model with Beta and Gamma Estimation - CORRECTED"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pybounds

############################################################################################
# Set some global parameters
############################################################################################
Lambda = 9.04e-5    # Recruitment rate per day
mu = 4.3e-5         # Mortality rate per day
sigma = 0.8         # Vaccine efficacy parameter (fixed: 0.8 means 20% efficacy)
epsilon = 0.033     # Progression rate from E to I (fixed: ~30 day latent period)
N = 223000000       # Total population

############################################################################################
# Continuous time dynamics function
############################################################################################
class F(object):
    def __init__(self):
        pass

    def f(self, x_vec, u_vec, Lambda=Lambda, mu=mu, sigma=sigma, epsilon=epsilon, 
          N=N, return_state_names=False):
        """
        Continuous time dynamics function for TB SVEIR model with PROPORTIONS.
        Includes Exposed (E) compartment and estimates beta and gamma.
        Epsilon is now a FIXED parameter.

        Parameters:
        x_vec : array-like, shape (7,)
            State vector [s, v, e, i, r, beta, gamma] (lowercase = proportions)
        u_vec : array-like, shape (2,)
            Control vector [alpha, kappa]
            alpha: vaccination rate
            kappa: social distancing effectiveness (0=no distancing, 1=full distancing)
        Lambda : float, default 9.04e-5
            Recruitment rate per day
        mu : float, default 4.3e-5
            Mortality rate per day
        sigma : float, default 0.8
            Vaccine efficacy parameter (fixed)
        epsilon : float, default 0.033
            Progression rate from E to I (fixed, ~30 day latent period)
        N : float, default 223000000
            Total population (for scaling)

        Returns:
        x_dot : numpy array, shape (7,)
            Time derivative of state vector
        """
        if return_state_names:
            return ['S', 'V', 'E', 'I', 'R', 'beta', 'gamma']

        # Extract state variables (proportions)
        s = x_vec[0]
        v = x_vec[1]
        e = x_vec[2]
        i = x_vec[3]
        r = x_vec[4]
        beta = x_vec[5]
        gamma = x_vec[6]  # Recovery rate

        # Extract control inputs
        alpha = u_vec[0]  # vaccination rate
        kappa = u_vec[1]  # social distancing effectiveness

        # f0 component: drift dynamics (no controls)
        # FIXED: Removed N from transmission - using proportions only
        f0_contribution = np.array([
            Lambda / N - beta * s * i - mu * s,
            -sigma * beta * v * i - mu * v,
            beta * s * i + sigma * beta * v * i - epsilon * e - mu * e,
            epsilon * e - gamma * i - mu * i,
            gamma * i - mu * r,
            0,
            0
        ])

        # f1 component: multiplied by control alpha (vaccination)
        f1_contribution = alpha * np.array([
            -s,
            s,
            0,
            0,
            0,
            0,
            0
        ])

        # f2 component: multiplied by control kappa (social distancing)
        # FIXED: Removed N from transmission - using proportions only
        f2_contribution = kappa * np.array([
            beta * s * i,
            sigma * beta * v * i,
            -beta * s * i - sigma * beta * v * i,
            0,
            0,
            0,
            0
        ])

        # Combined dynamics
        x_dot_vec = f0_contribution + f1_contribution + f2_contribution

        return x_dot_vec


############################################################################################
# Continuous time measurement functions
############################################################################################
class H(object):
    def __init__(self, measurement_option):
        self.measurement_option = measurement_option

    def h(self, x_vec, u_vec, return_measurement_names=False):
        h_func = self.__getattribute__(self.measurement_option)
        return h_func(x_vec, u_vec, return_measurement_names=return_measurement_names)

    def h_reported(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 1: y = [I] (Infected population only)
        Returns absolute counts
        """
        if return_measurement_names:
            return ['I_absolute']

        i = x_vec[3]  # proportion
        y_vec = np.array([i * N])  # convert to absolute
        return y_vec

    def h_incidence(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 2: y = [I, R]^T (Infected and Recovered populations)
        Returns absolute counts
        """
        if return_measurement_names:
            return ['I_absolute', 'R_absolute']

        i = x_vec[3]
        r = x_vec[4]
        y_vec = np.array([i * N, r * N])
        return y_vec

    def h_eir(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 3: y = [E, I, R]^T (Exposed, Infected, and Recovered)
        Returns absolute counts
        """
        if return_measurement_names:
            return ['E_absolute', 'I_absolute', 'R_absolute']

        e = x_vec[2]
        i = x_vec[3]
        r = x_vec[4]
        y_vec = np.array([e * N, i * N, r * N])
        return y_vec

    def h_all_sveir(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 4: y = [S, V, E, I, R]^T (All five compartments)
        Returns absolute counts
        """
        if return_measurement_names:
            return ['S_absolute', 'V_absolute', 'E_absolute', 'I_absolute', 'R_absolute']

        s = x_vec[0]
        v = x_vec[1]
        e = x_vec[2]
        i = x_vec[3]
        r = x_vec[4]
        y_vec = np.array([s * N, v * N, e * N, i * N, r * N])
        return y_vec

    def h_ei(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 5: y = [E, I]^T (Exposed and Infected)
        Good for observing latent period dynamics
        Returns absolute counts
        """
        if return_measurement_names:
            return ['E_absolute', 'I_absolute']

        e = x_vec[2]
        i = x_vec[3]
        y_vec = np.array([e * N, i * N])
        return y_vec

    def h_all_with_params(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 6: y = [S, V, E, I, R, beta, gamma]^T
        All compartments plus estimated parameters
        Returns absolute counts for compartments
        """
        if return_measurement_names:
            return ['S_absolute', 'V_absolute', 'E_absolute', 'I_absolute', 'R_absolute', 
                    'beta', 'gamma']

        s = x_vec[0]
        v = x_vec[1]
        e = x_vec[2]
        i = x_vec[3]
        r = x_vec[4]
        beta = x_vec[5]
        gamma = x_vec[6]
        y_vec = np.array([s * N, v * N, e * N, i * N, r * N, beta, gamma])
        return y_vec

    def h_with_infection_flow(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 7: y = [S, V, E, I, R, total_incidence]^T
        Includes total infection flow (entry to E) for better beta observability
        total_incidence = β*s*i*N + σ*β*v*i*N (absolute count per day)
        """
        if return_measurement_names:
            return ['S_absolute', 'V_absolute', 'E_absolute', 'I_absolute', 'R_absolute', 
                    'total_incidence']

        s = x_vec[0]
        v = x_vec[1]
        e = x_vec[2]
        i = x_vec[3]
        r = x_vec[4]
        beta = x_vec[5]
        
        total_incidence = (beta * s * i + sigma * beta * v * i) * N
        y_vec = np.array([s * N, v * N, e * N, i * N, r * N, total_incidence])
        return y_vec

    def h_with_progression_flow(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 8: y = [S, V, E, I, R, progression]^T
        Includes progression flow (E to I)
        progression = ε*e*N (absolute count per day)
        """
        if return_measurement_names:
            return ['S_absolute', 'V_absolute', 'E_absolute', 'I_absolute', 'R_absolute', 
                    'progression']

        s = x_vec[0]
        v = x_vec[1]
        e = x_vec[2]
        i = x_vec[3]
        r = x_vec[4]
        
        progression = epsilon * e * N
        y_vec = np.array([s * N, v * N, e * N, i * N, r * N, progression])
        return y_vec

    def h_with_recovery_flow(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 9: y = [S, V, E, I, R, recoveries]^T
        Includes recovery flow for better gamma observability
        recoveries = γ*i*N (absolute count per day)
        """
        if return_measurement_names:
            return ['S_absolute', 'V_absolute', 'E_absolute', 'I_absolute', 'R_absolute', 
                    'recoveries']

        s = x_vec[0]
        v = x_vec[1]
        e = x_vec[2]
        i = x_vec[3]
        r = x_vec[4]
        gamma = x_vec[6]
        
        recoveries = gamma * i * N
        y_vec = np.array([s * N, v * N, e * N, i * N, r * N, recoveries])
        return y_vec

    def h_with_two_flows(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 10: y = [S, V, E, I, R, total_incidence, recoveries]^T
        Infection and recovery flows - helps identify beta and gamma
        """
        if return_measurement_names:
            return ['S_absolute', 'V_absolute', 'E_absolute', 'I_absolute', 'R_absolute',
                    'total_incidence', 'recoveries']

        s = x_vec[0]
        v = x_vec[1]
        e = x_vec[2]
        i = x_vec[3]
        r = x_vec[4]
        beta = x_vec[5]
        gamma = x_vec[6]
        
        total_incidence = (beta * s * i + sigma * beta * v * i) * N
        recoveries = gamma * i * N
        y_vec = np.array([s * N, v * N, e * N, i * N, r * N, total_incidence, recoveries])
        return y_vec

    def h_with_all_flows(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 11: y = [S, V, E, I, R, total_incidence, progression, recoveries]^T
        BEST - All three major flows
        """
        if return_measurement_names:
            return ['S_absolute', 'V_absolute', 'E_absolute', 'I_absolute', 'R_absolute',
                    'total_incidence', 'progression', 'recoveries']

        s = x_vec[0]
        v = x_vec[1]
        e = x_vec[2]
        i = x_vec[3]
        r = x_vec[4]
        beta = x_vec[5]
        gamma = x_vec[6]
        
        total_incidence = (beta * s * i + sigma * beta * v * i) * N
        progression = epsilon * e * N
        recoveries = gamma * i * N
        y_vec = np.array([s * N, v * N, e * N, i * N, r * N, total_incidence, progression, recoveries])
        return y_vec

    def h_comprehensive(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 12: y = [S, V, E, I, R, unvax_incidence, vax_incidence, progression, recoveries]^T
        Most comprehensive - separates all infection types and flows
        """
        if return_measurement_names:
            return ['S_absolute', 'V_absolute', 'E_absolute', 'I_absolute', 'R_absolute',
                    'unvax_incidence', 'vax_incidence', 'progression', 'recoveries']

        s = x_vec[0]
        v = x_vec[1]
        e = x_vec[2]
        i = x_vec[3]
        r = x_vec[4]
        beta = x_vec[5]
        gamma = x_vec[6]
        
        unvax_incidence = beta * s * i * N
        vax_incidence = sigma * beta * v * i * N
        progression = epsilon * e * N
        recoveries = gamma * i * N
        y_vec = np.array([s * N, v * N, e * N, i * N, r * N, unvax_incidence, vax_incidence, progression, recoveries])
        return y_vec


############################################################################################
# TB simulation
############################################################################################
def simulate_tb(f, h, tsim_length=365, dt=1.0, measurement_names=None,
                setpoint=None, rterm_alpha=1e-4, rterm_kappa=1e-4, x0=None):
    """
    Simulate TB disease model with MPC control

    Parameters:
    -----------
    f : function
        Dynamics function
    h : function
        Measurement function
    tsim_length : float
        Total simulation time in days
    dt : float
        Time step in days
    measurement_names : list
        Names of measurements
    setpoint : dict
        Desired trajectories for states
    rterm_alpha : float
        Control input penalty for vaccination
    rterm_kappa : float
        Control input penalty for social distancing
    x0 : array-like
        Initial conditions [S, V, E, I, R, beta, gamma]

    Returns:
    --------
    t_sim, x_sim, u_sim, y_sim, simulator
    """
    # Set state and input names
    state_names = f(None, None, return_state_names=True)
    input_names = ['alpha', 'kappa']  # vaccination and social distancing

    # Choose the measurement function
    if measurement_names is None:
        try:
            measurement_names = h(None, None, return_measurement_names=True)
        except:
            raise ValueError('Need to provide measurement_names as a list of strings')

    # Initialize simulator
    simulator = pybounds.Simulator(f, h, dt=dt, state_names=state_names,
                                   input_names=input_names, measurement_names=measurement_names,
                                   mpc_horizon=int(10/dt))

    # Define the time horizon
    tsim = np.arange(0, tsim_length, step=dt)
    NA = np.zeros_like(tsim)

    # Define default setpoint if not provided
    if setpoint is None:
        # Vaccination setpoint: Ramp up to 80% over 180 days
        V_target = 0.80 * N
        V_setpoint = np.minimum(V_target * (tsim / 180), V_target)

        # Infection setpoint: Decrease infections exponentially
        if x0 is not None:
            I_initial = x0[3]
            E_initial = x0[2]
        else:
            I_initial = 361000
            E_initial = 100000
        
        I_target = 0.001 * N
        I_setpoint = I_initial * np.exp(-tsim / 200)
        E_setpoint = E_initial * np.exp(-tsim / 200)

        setpoint = {
            'S': NA,
            'V': V_setpoint,
            'E': E_setpoint,
            'I': I_setpoint,
            'R': NA,
            'beta': 0.3 * np.ones_like(tsim),
            'gamma': 0.00555 * np.ones_like(tsim),
        }

    # Update the simulator set-point
    simulator.update_dict(setpoint, name='setpoint')

    # Define MPC cost function
    cost_V = (simulator.model.x['V'] - simulator.model.tvp['V_set']) ** 2
    cost_E = (simulator.model.x['E'] - simulator.model.tvp['E_set']) ** 2
    cost_I = (simulator.model.x['I'] - simulator.model.tvp['I_set']) ** 2
    cost = 10 * cost_I + 5 * cost_E + 10 * cost_V

    # Set cost function
    simulator.mpc.set_objective(mterm=cost, lterm=cost)

    # Set input penalty
    simulator.mpc.set_rterm(alpha=rterm_alpha, kappa=rterm_kappa)

    # Set bounds on states and controls
    simulator.mpc.bounds['lower', '_x', 'S'] = 0.0
    simulator.mpc.bounds['upper', '_x', 'S'] = N
    simulator.mpc.bounds['lower', '_x', 'V'] = 0.0
    simulator.mpc.bounds['upper', '_x', 'V'] = N
    simulator.mpc.bounds['lower', '_x', 'E'] = 0.0
    simulator.mpc.bounds['upper', '_x', 'E'] = N
    simulator.mpc.bounds['lower', '_x', 'I'] = 0.0
    simulator.mpc.bounds['upper', '_x', 'I'] = N
    simulator.mpc.bounds['lower', '_x', 'R'] = 0.0
    simulator.mpc.bounds['upper', '_x', 'R'] = N
    simulator.mpc.bounds['lower', '_x', 'beta'] = 0.0
    simulator.mpc.bounds['upper', '_x', 'beta'] = 1.0
    simulator.mpc.bounds['lower', '_x', 'gamma'] = 0.0
    simulator.mpc.bounds['upper', '_x', 'gamma'] = 0.1  # Recovery rate bound

    simulator.mpc.bounds['lower', '_u', 'alpha'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'alpha'] = 0.5
    simulator.mpc.bounds['lower', '_u', 'kappa'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'kappa'] = 1.0

    # Run simulation using MPC
    t_sim, x_sim, u_sim, y_sim, = simulator.simulate(x0=x0, u=None, mpc=True, return_full_output=True)

    return t_sim, x_sim, u_sim, y_sim, simulator


############################################################################################
# Example usage
############################################################################################
if __name__ == "__main__":
    # Define initial conditions [S, V, E, I, R, beta, gamma]
    E_initial = 100000  # Initial exposed population
    I_initial = 361000
    V_initial = 158330000
    R_initial = 12000000
    S_initial = N - V_initial - E_initial - I_initial - R_initial
    
    x0 = np.array([
        S_initial,      # S
        V_initial,      # V
        E_initial,      # E (exposed)
        I_initial,      # I
        R_initial,      # R
        0.3,            # beta (transmission rate)
        0.00555         # gamma (recovery rate)
    ])

    # Create dynamics object
    f_obj = F()

    print("="*80)
    print("TESTING ALL MEASUREMENT OPTIONS (SVEIR: Beta and Gamma Estimation)")
    print(f"Fixed parameters: epsilon = {epsilon:.6f} (~{1/epsilon:.1f} day latent period)")
    print("="*80)

    # Test all measurement options
    measurement_options = [
        ('h_reported', 'Measurement 1: I only'),
        ('h_incidence', 'Measurement 2: I + R'),
        ('h_ei', 'Measurement 3: E + I'),
        ('h_eir', 'Measurement 4: E + I + R'),
        ('h_all_sveir', 'Measurement 5: S + V + E + I + R'),
        ('h_all_with_params', 'Measurement 6: SVEIR + beta + gamma'),
        ('h_with_infection_flow', 'Measurement 7: SVEIR + infection flow (good for beta)'),
        ('h_with_progression_flow', 'Measurement 8: SVEIR + progression flow (ε*E)'),
        ('h_with_recovery_flow', 'Measurement 9: SVEIR + recovery flow (good for gamma)'),
        ('h_with_two_flows', 'Measurement 10: SVEIR + infection + recovery ⭐ BEST'),
        ('h_with_all_flows', 'Measurement 11: SVEIR + all 3 flows'),
        ('h_comprehensive', 'Measurement 12: SVEIR + separated flows (most detailed)')
    ]

    results = {}

    for option_name, description in measurement_options:
        print(f"\n{description}")
        print("-" * 60)
        
        h_obj = H(measurement_option=option_name)
        measurement_names = h_obj.h(None, None, return_measurement_names=True)
        print(f"Measurements: {measurement_names}")
        
        try:
            t_sim, x_sim, u_sim, y_sim, simulator = simulate_tb(
                f_obj.f, h_obj.h, tsim_length=365, dt=1.0, x0=x0
            )
            results[option_name] = {
                't': t_sim,
                'x': x_sim,
                'u': u_sim,
                'y': y_sim,
                'simulator': simulator,
                'measurements': measurement_names
            }
            print(f"✓ Simulation successful")
            print(f"  Final E: {x_sim['E'][-1]:.0f}")
            print(f"  Final I: {x_sim['I'][-1]:.0f}")
            print(f"  Final V: {x_sim['V'][-1]:.0f}")
            print(f"  Final beta: {x_sim['beta'][-1]:.4f}")
            print(f"  Final gamma: {x_sim['gamma'][-1]:.6f}")
        except Exception as e:
            print(f"✗ Simulation failed: {str(e)}")

    print("\n" + "="*80)
    print("SUMMARY: SVEIR Model - Estimating beta and gamma (epsilon FIXED)")
    print("="*80)
    print("\nAvailable measurement options for empirical observability:")
    print("  BASIC (compartments only):")
    print("    - h_reported:  I only (poor observability)")
    print("    - h_ei:        E + I (better, shows latent dynamics)")
    print("    - h_incidence: I + R")
    print("    - h_eir:       E + I + R")
    print("    - h_all_sveir: S + V + E + I + R (all compartments)")
    print("\n  ADVANCED (with flows for parameter identification):")
    print("    - h_all_with_params:       SVEIR + beta + gamma")
    print("    - h_with_infection_flow:   SVEIR + infection flow (β*S*I)")
    print("    - h_with_progression_flow: SVEIR + progression flow (ε*E, ε fixed)")
    print("    - h_with_recovery_flow:    SVEIR + recovery flow (γ*I)")
    print("    - h_with_two_flows:        SVEIR + infection + recovery ⭐ BEST")
    print("    - h_with_all_flows:        SVEIR + all 3 flows")
    print("    - h_comprehensive:         SVEIR + separated infection types + all flows")
    print("\n  KEY INSIGHTS:")
    print("    - Infection flow β*S*I + σ*β*V*I → helps identify β")
    print("    - Recovery flow γ*I → helps identify γ")
    print("    - Progression flow ε*E → depends only on E (ε is FIXED)")
    print("    - Measuring E helps with overall dynamics but ε not estimated")
    print("    - Best: SVEIR + infection flow + recovery flow")
    print("="*80)
