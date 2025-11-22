# -*- coding: utf-8 -*-
"""tb_simulations.py - Updated to estimate beta and gamma instead of sigma"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pybounds

############################################################################################
# Set some global parameters
############################################################################################
Lambda = 9.04e-5    # Recruitment rate per day
mu = 4.3e-5         # Mortality rate per day
sigma = 0.8         # Vaccine efficacy (now fixed parameter)
N = 223000000       # Total population

############################################################################################
# Continuous time dynamics function
############################################################################################
class F(object):
    def __init__(self):
        pass

    def f(self, x_vec, u_vec, Lambda=Lambda, mu=mu, sigma=sigma, return_state_names=False):
        """
        Continuous time dynamics function for TB SVIR model.
        Now estimates beta and gamma as state variables.

        Parameters:
        x_vec : array-like, shape (6,)
            State vector [S, V, I, R, beta, gamma]
        u_vec : array-like, shape (2,)
            Control vector [alpha, kappa]
            alpha: vaccination rate
            kappa: social distancing effectiveness (0=no distancing, 1=full distancing)
        Lambda : float, default 9.04e-5
            Recruitment rate per day
        mu : float, default 4.3e-5
            Mortality rate per day
        sigma : float, default 0.8
            Vaccine efficacy (fixed parameter)

        Returns:
        x_dot : numpy array, shape (6,)
            Time derivative of state vector
        """
        if return_state_names:
            return ['S', 'V', 'I', 'R', 'beta', 'gamma']

        # Extract state variables
        S = x_vec[0]
        V = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]
        beta = x_vec[4]
        gamma = x_vec[5]  # Now a state variable

        # Extract control inputs
        alpha = u_vec[0]  # vaccination rate
        kappa = u_vec[1]  # social distancing effectiveness

        # f0 component: drift dynamics (no controls)
        f0_contribution = np.array([Lambda - beta * S * I - mu * S,
                                     -sigma * beta * V * I - mu * V,
                                     beta*S*I + sigma*beta*V*I - gamma*I - mu*I,
                                     gamma*I - mu*R,
                                     0,
                                     0])

        # f1 component: multiplied by control alpha (vaccination)
        f1_contribution = alpha * np.array([-S,
                                             S,
                                             0,
                                             0,
                                             0,
                                             0])

        # f2 component: multiplied by control kappa (social distancing)
        # Social distancing reduces transmission: effective beta becomes beta*(1-kappa)
        f2_contribution = kappa * np.array([beta*S*I,
                                             sigma*beta*V*I,
                                             -beta*S*I - sigma*beta*V*I,
                                             0,
                                             0,
                                             0])

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
        """
        if return_measurement_names:
            return ['I_absolute']

        I = x_vec[2]
        y_vec = np.array([I])
        return y_vec

    def h_incidence(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 2: y = [I, R]^T (Infected and Recovered populations)
        """
        if return_measurement_names:
            return ['I_absolute', 'R_absolute']

        I = x_vec[2]
        R = x_vec[3]
        y_vec = np.array([I, R])
        return y_vec

    def h_ivr(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 3: y = [I, V, R]^T (Infected, Vaccinated, and Recovered)
        """
        if return_measurement_names:
            return ['I_absolute', 'V_absolute', 'R_absolute']

        I = x_vec[2]
        V = x_vec[1]
        R = x_vec[3]
        y_vec = np.array([I, V, R])
        return y_vec

    def h_all_svir(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 4: y = [S, V, I, R]^T (All four compartments)
        """
        if return_measurement_names:
            return ['S_absolute', 'V_absolute', 'I_absolute', 'R_absolute']

        S = x_vec[0]
        V = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]
        y_vec = np.array([S, V, I, R])
        return y_vec

    def h_is(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 5: y = [I, S]^T (Infected and Susceptible)
        """
        if return_measurement_names:
            return ['I_absolute', 'S_absolute']

        I = x_vec[2]
        S = x_vec[0]
        y_vec = np.array([I, S])
        return y_vec

    def h_iv(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 6: y = [I, V]^T (Infected and Vaccinated)
        """
        if return_measurement_names:
            return ['I_absolute', 'V_absolute']

        I = x_vec[2]
        V = x_vec[1]
        y_vec = np.array([I, V])
        return y_vec

    def h_all_with_params(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 7: y = [S, V, I, R, beta, gamma]^T
        All compartments plus parameters
        FULL observability - includes transmission rate and recovery rate
        """
        if return_measurement_names:
            return ['S_absolute', 'V_absolute', 'I_absolute', 'R_absolute', 'beta', 'gamma']

        S = x_vec[0]
        V = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]
        beta = x_vec[4]
        gamma = x_vec[5]
        y_vec = np.array([S, V, I, R, beta, gamma])
        return y_vec

    def h_with_total_incidence(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 8: y = [S, V, I, R, total_incidence]^T
        Includes total incidence rate for better beta observability
        total_incidence = β*S*I + σ*β*V*I
        """
        if return_measurement_names:
            return ['S_absolute', 'V_absolute', 'I_absolute', 'R_absolute', 'total_incidence']

        S = x_vec[0]
        V = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]
        beta = x_vec[4]
        total_incidence = beta*S*I + sigma*beta*V*I
        y_vec = np.array([S, V, I, R, total_incidence])
        return y_vec

    def h_with_recovery_flow(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 9: y = [S, V, I, R, recoveries]^T
        Includes recovery flow for better gamma observability
        recoveries = γ*I
        """
        if return_measurement_names:
            return ['S_absolute', 'V_absolute', 'I_absolute', 'R_absolute', 'recoveries']

        S = x_vec[0]
        V = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]
        gamma = x_vec[5]
        recoveries = gamma*I
        y_vec = np.array([S, V, I, R, recoveries])
        return y_vec

    def h_with_flows(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 10: y = [S, V, I, R, total_incidence, recoveries]^T
        BEST for beta and gamma observability - separates infection and recovery flows
        total_incidence = β*S*I + σ*β*V*I (depends on β)
        recoveries = γ*I (depends on γ)
        """
        if return_measurement_names:
            return ['S_absolute', 'V_absolute', 'I_absolute', 'R_absolute', 
                    'total_incidence', 'recoveries']

        S = x_vec[0]
        V = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]
        beta = x_vec[4]
        gamma = x_vec[5]
        
        total_incidence = beta*S*I + sigma*beta*V*I
        recoveries = gamma*I
        y_vec = np.array([S, V, I, R, total_incidence, recoveries])
        return y_vec

    def h_with_unvax_incidence(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 11: y = [S, V, I, R, unvax_incidence, recoveries]^T
        Alternative measurement - focuses on unvaccinated infections + recoveries
        unvax_incidence = β*S*I (depends on β only, cleaner than total)
        recoveries = γ*I (depends on γ)
        """
        if return_measurement_names:
            return ['S_absolute', 'V_absolute', 'I_absolute', 'R_absolute',
                    'unvax_incidence', 'recoveries']

        S = x_vec[0]
        V = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]
        beta = x_vec[4]
        gamma = x_vec[5]
        
        unvax_incidence = beta*S*I
        recoveries = gamma*I
        y_vec = np.array([S, V, I, R, unvax_incidence, recoveries])
        return y_vec

    def h_comprehensive(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 12: y = [S, V, I, R, unvax_incidence, vax_incidence, recoveries]^T
        Most comprehensive - all major flows including breakthrough infections
        """
        if return_measurement_names:
            return ['S_absolute', 'V_absolute', 'I_absolute', 'R_absolute',
                    'unvax_incidence', 'vax_incidence', 'recoveries']

        S = x_vec[0]
        V = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]
        beta = x_vec[4]
        gamma = x_vec[5]
        
        unvax_incidence = beta*S*I
        vax_incidence = sigma*beta*V*I
        recoveries = gamma*I
        y_vec = np.array([S, V, I, R, unvax_incidence, vax_incidence, recoveries])
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
        Initial conditions [S, V, I, R, beta, gamma]

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
            I_initial = x0[2]
        else:
            I_initial = 361000
        I_target = 0.001 * N
        I_setpoint = I_initial * np.exp(-tsim / 200)

        setpoint = {
            'S': NA,
            'V': V_setpoint,
            'I': I_setpoint,
            'R': NA,
            'beta': 0.3 * np.ones_like(tsim),
            'gamma': 0.00555 * np.ones_like(tsim),
        }

    # Update the simulator set-point
    simulator.update_dict(setpoint, name='setpoint')

    # Define MPC cost function
    cost_V = (simulator.model.x['V'] - simulator.model.tvp['V_set']) ** 2
    cost_I = (simulator.model.x['I'] - simulator.model.tvp['I_set']) ** 2
    cost = 10 * cost_I + 10 * cost_V

    # Set cost function
    simulator.mpc.set_objective(mterm=cost, lterm=cost)

    # Set input penalty
    simulator.mpc.set_rterm(alpha=rterm_alpha, kappa=rterm_kappa)

    # Set bounds on states and controls
    simulator.mpc.bounds['lower', '_x', 'S'] = 0.0
    simulator.mpc.bounds['upper', '_x', 'S'] = N
    simulator.mpc.bounds['lower', '_x', 'V'] = 0.0
    simulator.mpc.bounds['upper', '_x', 'V'] = N
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
    # Define initial conditions [S, V, I, R, beta, gamma]
    x0 = np.array([
        (N - 158330000 - 361000 - 12000000),  # S
        158330000,                             # V
        361000,                                # I
        12000000,                              # R
        0.3,                                   # beta (transmission rate)
        0.00555                                # gamma (recovery rate)
    ])

    # Create dynamics object
    f_obj = F()

    print("="*80)
    print("TESTING ALL MEASUREMENT OPTIONS (Beta and Gamma Estimation)")
    print("="*80)

    # Test all measurement options
    measurement_options = [
        ('h_reported', 'Measurement 1: I only'),
        ('h_incidence', 'Measurement 2: I + R'),
        ('h_is', 'Measurement 3: I + S'),
        ('h_iv', 'Measurement 4: I + V'),
        ('h_ivr', 'Measurement 5: I + V + R'),
        ('h_all_svir', 'Measurement 6: S + V + I + R'),
        ('h_all_with_params', 'Measurement 7: S + V + I + R + beta + gamma'),
        ('h_with_total_incidence', 'Measurement 8: SVIR + total incidence'),
        ('h_with_recovery_flow', 'Measurement 9: SVIR + recoveries (good for gamma)'),
        ('h_with_flows', 'Measurement 10: SVIR + incidence + recoveries ⭐ BEST'),
        ('h_with_unvax_incidence', 'Measurement 11: SVIR + unvax incidence + recoveries'),
        ('h_comprehensive', 'Measurement 12: SVIR + all flows')
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
            print(f"  Final I: {x_sim['I'][-1]:.0f}")
            print(f"  Final V: {x_sim['V'][-1]:.0f}")
            print(f"  Final beta: {x_sim['beta'][-1]:.4f}")
            print(f"  Final gamma: {x_sim['gamma'][-1]:.6f}")
        except Exception as e:
            print(f"✗ Simulation failed: {str(e)}")

    print("\n" + "="*80)
    print("SUMMARY: Estimating beta (transmission) and gamma (recovery)")
    print("="*80)
    print("\nAvailable measurement options for empirical observability:")
    print("  BASIC (compartments only):")
    print("    - h_reported:  I only (poor observability)")
    print("    - h_is:        I + S")
    print("    - h_iv:        I + V")
    print("    - h_incidence: I + R (good for gamma through dR/dt)")
    print("    - h_ivr:       I + V + R")
    print("    - h_all_svir:  S + V + I + R")
    print("\n  ADVANCED (with parameters or flows):")
    print("    - h_all_with_params:      SVIR + beta + gamma")
    print("    - h_with_total_incidence: SVIR + total incidence (good for beta)")
    print("    - h_with_recovery_flow:   SVIR + recoveries (good for gamma)")
    print("    - h_with_flows:           SVIR + incidence + recoveries ⭐ BEST")
    print("    - h_with_unvax_incidence: SVIR + unvax incidence + recoveries")
    print("    - h_comprehensive:        SVIR + all flows")
    print("\n  KEY INSIGHT:")
    print("    - Recovery flow γ*I helps identify gamma")
    print("    - Infection flow β*S*I helps identify beta")
    print("    - Measuring R gives dR/dt = γ*I (indirect gamma observability)")
    print("="*80)
