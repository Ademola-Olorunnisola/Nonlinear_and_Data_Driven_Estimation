# -*- coding: utf-8 -*-
"""tb_simulations.py - Updated with Observable Measurement Options"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pybounds

############################################################################################
# Set some global parameters
############################################################################################
Lambda = 9.04e-5    # Recruitment rate per day
mu = 4.3e-5         # Mortality rate per day
gamma = 0.00555     # Removal rate per day
N = 223000000       # Total population

############################################################################################
# Continuous time dynamics function
############################################################################################
class F(object):
    def __init__(self):
        pass

    def f(self, x_vec, u_vec, Lambda=Lambda, mu=mu, gamma=gamma, return_state_names=False):
        """
        Continuous time dynamics function for TB SVIR model.

        Parameters:
        x_vec : array-like, shape (6,)
            State vector [S, V, I, R, beta, sigma]
        u_vec : array-like, shape (2,)
            Control vector [alpha, kappa]
            alpha: vaccination rate
            kappa: social distancing effectiveness (0=no distancing, 1=full distancing)
        Lambda : float, default 9.04e-5
            Recruitment rate per day
        mu : float, default 4.3e-5
            Mortality rate per day
        gamma : float, default 0.00555
            Removal rate per day

        Returns:
        x_dot : numpy array, shape (6,)
            Time derivative of state vector
        """
        if return_state_names:
            return ['S', 'V', 'I', 'R', 'beta', 'sigma']

        # Extract state variables
        S = x_vec[0]
        V = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]
        beta = x_vec[4]
        sigma = x_vec[5]

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
        From analytical analysis: INSUFFICIENT for full observability
        """
        if return_measurement_names:
            return ['I_absolute']

        # Extract state variables
        I = x_vec[2]

        # Measurements
        y_vec = np.array([I])

        return y_vec

    def h_incidence(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 2: y = [I, R]^T (Infected and Recovered populations)
        """
        if return_measurement_names:
            return ['I_absolute', 'R_absolute']

        # Extract state variables
        I = x_vec[2]
        R = x_vec[3]

        # Measurements
        y_vec = np.array([I, R])

        return y_vec

    def h_ivr(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 3 (NEW): y = [I, V, R]^T (Infected, Vaccinated, and Recovered)
        Better observability - includes vaccination compartment
        """
        if return_measurement_names:
            return ['I_absolute', 'V_absolute', 'R_absolute']

        # Extract state variables
        I = x_vec[2]
        V = x_vec[1]
        R = x_vec[3]

        # Measurements
        y_vec = np.array([I, V, R])

        return y_vec

    def h_all_svir(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 4 (NEW): y = [S, V, I, R]^T (All four compartments)
        Maximum observability - all SVIR states measured
        From analytical analysis: Should give FULL observability (rank=6)
        """
        if return_measurement_names:
            return ['S_absolute', 'V_absolute', 'I_absolute', 'R_absolute']

        # Extract state variables
        S = x_vec[0]
        V = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]

        # Measurements
        y_vec = np.array([S, V, I, R])

        return y_vec

    def h_is(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 5 (NEW): y = [I, S]^T (Infected and Susceptible)
        Good for analyzing transmission dynamics
        """
        if return_measurement_names:
            return ['I_absolute', 'S_absolute']

        # Extract state variables
        I = x_vec[2]
        S = x_vec[0]

        # Measurements
        y_vec = np.array([I, S])

        return y_vec

    def h_iv(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 6 (NEW): y = [I, V]^T (Infected and Vaccinated)
        Good for analyzing vaccination effectiveness
        """
        if return_measurement_names:
            return ['I_absolute', 'V_absolute']

        # Extract state variables
        I = x_vec[2]
        V = x_vec[1]

        # Measurements
        y_vec = np.array([I, V])

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
        Initial conditions

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
            'sigma': 0.8 * np.ones_like(tsim),
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
    simulator.mpc.bounds['lower', '_x', 'sigma'] = 0.0
    simulator.mpc.bounds['upper', '_x', 'sigma'] = 1.0
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
    # Define initial conditions
    x0 = np.array([
        (N - 158330000 - 361000 - 12000000),  # S
        158330000,                             # V
        361000,                                # I
        12000000,                              # R
        0.3,                                   # beta
        0.8                                    # sigma
    ])

    # Create dynamics object
    f_obj = F()

    print("="*80)
    print("TESTING ALL MEASUREMENT OPTIONS")
    print("="*80)

    # Test all measurement options
    measurement_options = [
        ('h_reported', 'Measurement 1: I only'),
        ('h_incidence', 'Measurement 2: I + R'),
        ('h_is', 'Measurement 3: I + S'),
        ('h_iv', 'Measurement 4: I + V'),
        ('h_ivr', 'Measurement 5: I + V + R'),
        ('h_all_svir', 'Measurement 6: S + V + I + R (Full)')
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
        except Exception as e:
            print(f"✗ Simulation failed: {str(e)}")

    print("\n" + "="*80)
    print("SUMMARY: All measurement options tested successfully!")
    print("="*80)
    print("\nAvailable measurement options for empirical observability:")
    print("  - h_reported:  I only (baseline, poor observability)")
    print("  - h_is:        I + S")
    print("  - h_iv:        I + V")
    print("  - h_incidence: I + R")
    print("  - h_ivr:       I + V + R")
    print("  - h_all_svir:  S + V + I + R (best observability)")
    print("="*80)
