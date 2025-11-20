# -*- coding: utf-8 -*-
"""tb_simulations_corrected.py - TB Simulation using population fractions"""

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
        Continuous time dynamics function for TB SVIR model (fractions)
        """
        if return_state_names:
            return ['S', 'V', 'I', 'R', 'beta', 'sigma']

        # Extract state variables (fractions of population)
        S = x_vec[0]
        V = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]
        beta = x_vec[4]
        sigma = x_vec[5]

        # Extract control inputs
        alpha = u_vec[0]  # vaccination rate
        kappa = u_vec[1]  # social distancing effectiveness

        # Drift dynamics
        f0_contribution = np.array([
            Lambda/N - beta*S*I - mu*S,
            -sigma*beta*V*I - mu*V,
            beta*S*I + sigma*beta*V*I - gamma*I - mu*I,
            gamma*I - mu*R,
            0,
            0
        ])

        # Vaccination control
        f1_contribution = alpha * np.array([-S, S, 0, 0, 0, 0])

        # Social distancing control
        f2_contribution = kappa * np.array([-beta*S*I, -sigma*beta*V*I, beta*S*I + sigma*beta*V*I, 0, 0, 0])

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

    # Measurement: Infected only
    def h_reported(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I_absolute']
        return np.array([x_vec[2]])

    # Measurement: I + R
    def h_incidence(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I_absolute', 'R_absolute']
        return np.array([x_vec[2], x_vec[3]])

    # Measurement: I + V
    def h_iv(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I_absolute', 'V_absolute']
        return np.array([x_vec[2], x_vec[1]])

    # Measurement: I + V + R
    def h_ivr(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I_absolute', 'V_absolute', 'R_absolute']
        return np.array([x_vec[2], x_vec[1], x_vec[3]])

    # Measurement: S + V + I + R
    def h_all_svir(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['S_absolute', 'V_absolute', 'I_absolute', 'R_absolute']
        return np.array([x_vec[0], x_vec[1], x_vec[2], x_vec[3]])


############################################################################################
# TB simulation with MPC
############################################################################################
def simulate_tb(f, h, tsim_length=365, dt=1.0, measurement_names=None,
                setpoint=None, rterm_alpha=1e-4, rterm_kappa=1e-4, x0=None):

    # State and input names
    state_names = f(None, None, return_state_names=True)
    input_names = ['alpha', 'kappa']

    if measurement_names is None:
        measurement_names = h(None, None, return_measurement_names=True)

    # Initialize simulator
    simulator = pybounds.Simulator(f, h, dt=dt, state_names=state_names,
                                   input_names=input_names, measurement_names=measurement_names,
                                   mpc_horizon=int(10/dt))

    # Time horizon
    tsim = np.arange(0, tsim_length, step=dt)
    NA = np.zeros_like(tsim)

    # Default setpoints (fractions)
    if setpoint is None:
        if x0 is not None:
            I_initial = x0[2]
            V_initial = x0[1]
        else:
            I_initial = 361000/N
            V_initial = 158330000/N

        # Vaccination target: ramp to 80% fraction
        V_target = 0.8
        V_setpoint = np.minimum(V_target * (tsim / 180), V_target)

        # Infection target: exponential decay
        I_target = 0.001
        I_setpoint = I_initial * np.exp(-tsim / 200)

        setpoint = {
            'S': NA,
            'V': V_setpoint,
            'I': I_setpoint,
            'R': NA,
            'beta': 0.3 * np.ones_like(tsim),
            'sigma': 0.8 * np.ones_like(tsim)
        }

    simulator.update_dict(setpoint, name='setpoint')

    # MPC cost
    cost_V = (simulator.model.x['V'] - simulator.model.tvp['V_set']) ** 2
    cost_I = (simulator.model.x['I'] - simulator.model.tvp['I_set']) ** 2
    cost = 10 * cost_I + 10 * cost_V
    simulator.mpc.set_objective(mterm=cost, lterm=cost)

    # Input penalties
    simulator.mpc.set_rterm(alpha=rterm_alpha, kappa=rterm_kappa)

    # Bounds (fractions)
    simulator.mpc.bounds['lower', '_x', 'S'] = 0.0
    simulator.mpc.bounds['upper', '_x', 'S'] = 1.0
    simulator.mpc.bounds['lower', '_x', 'V'] = 0.0
    simulator.mpc.bounds['upper', '_x', 'V'] = 1.0
    simulator.mpc.bounds['lower', '_x', 'I'] = 0.0
    simulator.mpc.bounds['upper', '_x', 'I'] = 1.0
    simulator.mpc.bounds['lower', '_x', 'R'] = 0.0
    simulator.mpc.bounds['upper', '_x', 'R'] = 1.0
    simulator.mpc.bounds['lower', '_x', 'beta'] = 0.0
    simulator.mpc.bounds['upper', '_x', 'beta'] = 1.0
    simulator.mpc.bounds['lower', '_x', 'sigma'] = 0.0
    simulator.mpc.bounds['upper', '_x', 'sigma'] = 1.0
    simulator.mpc.bounds['lower', '_u', 'alpha'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'alpha'] = 0.01  # smaller to prevent S collapse
    simulator.mpc.bounds['lower', '_u', 'kappa'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'kappa'] = 1.0

    # Run simulation
    t_sim, x_sim, u_sim, y_sim = simulator.simulate(x0=x0, u=None, mpc=True, return_full_output=True)

    return t_sim, x_sim, u_sim, y_sim, simulator


############################################################################################
# Example usage
############################################################################################
if __name__ == "__main__":

    # Initial conditions (fractions)
    x0 = np.array([
        (N - 158330000 - 361000 - 12000000)/N,  # S
        158330000/N,                             # V
        361000/N,                                # I
        12000000/N,                              # R
        0.3,                                     # beta
        0.8                                      # sigma
    ])

    f_obj = F()

    # Choose measurement option
    h_obj = H(measurement_option='h_all_svir')  # full SVIR measurements

    t_sim, x_sim, u_sim, y_sim, simulator = simulate_tb(
        f_obj.f, h_obj.h, tsim_length=365, dt=1.0, x0=x0
    )

    # Plot S, V, I, R
    plt.figure(figsize=(8,6))
    plt.plot(t_sim, x_sim['S'], label='Susceptible')
    plt.plot(t_sim, x_sim['V'], label='Vaccinated')
    plt.plot(t_sim, x_sim['I'], label='Infected')
    plt.plot(t_sim, x_sim['R'], label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population fraction')
    plt.title('TB Model Simulation (fractions)')
    plt.legend()
    plt.grid(True)
    plt.show()
