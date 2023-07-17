"""
A script demonstrating the Runge-Kutta method for solving ODEs.

author: Fabrizio Musacchio (fabriziomusacchio.com)
date: Oct 03, 2020
"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.integrate import solve_ivp
# %% SIMPLE ODE
# Define the function
def f(t, y):
    return y

# Exact solution
def y_exact(t):
    return np.exp(t)

# Define the Runge-Kutta methods
def rk1(y, t, dt, derivs):
    k1 = dt * derivs(t, y)
    y_next = y + k1
    return y_next

def rk2(y, t, dt, derivs):
    k1 = dt * derivs(t, y)
    k2 = dt * derivs(t + dt / 2, y + k1 / 2)
    y_next = y + k2
    return y_next

def rk3(y, t, dt, derivs):
    k1 = dt * derivs(t, y)
    k2 = dt * derivs(t + dt / 2, y + k1 / 2)
    k3 = dt * derivs(t + dt, y - k1 + 2 * k2)
    y_next = y + (k1 + 4 * k2 + k3) / 6
    return y_next

def rk4(y, t, dt, derivs):
    k1 = dt * derivs(t, y)
    k2 = dt * derivs(t + dt / 2, y + k1 / 2)
    k3 = dt * derivs(t + dt / 2, y + k2 / 2)
    k4 = dt * derivs(t + dt, y + k3)
    y_next = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y_next

# Define parameters
t_start = 0.0
t_end = 2.0
N = 10  # number of steps
h = (t_end - t_start) / N  # step size
t_values = np.linspace(t_start, t_end, N+1)

# Initial condition
y_0 = 1.0

# Solve the ODE using different methods and calculate MSE
y_values_rk = dict()
mse_values = dict()

for method, name in [(rk1, 'RK1'), (rk2, 'RK2'), (rk3, 'RK3'), (rk4, 'RK4')]:
    y_values = np.zeros(N+1)
    y_values[0] = y_0
    for i in range(N):
        y_values[i+1] = method(y_values[i], t_values[i], h, f)
    y_values_rk[name] = y_values
    mse_values[name] = mean_squared_error(y_exact(t_values), y_values)

# Plot the results
plt.figure(figsize=(6, 5))
plt.plot(t_values, y_exact(t_values), label='Exact solution', linestyle='--', color='k', lw=4)
for name in ['RK1', 'RK2', 'RK3', 'RK4']:
    plt.plot(t_values, y_values_rk[name], label=f'{name} (MSE: {mse_values[name]:.2e})')
plt.xlabel('t')
plt.ylabel('y')
plt.title(f"Solving $y' = y$ with step-size h={h}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'runge_kutta_method_exponential_h_{np.round(h, 2)}.png', dpi=200)
plt.show()
# %% PENDULUM
def pendulum(t, state):
    y, v = state
    dydt = v
    dvdt = -y
    return np.array([dydt, dvdt])

# Exact solution
def y_exact(t):
    return state_0[0] * np.cos(t)

# Define parameters
t_start = 0.0
t_end = 10.0
N = 200  # number of steps
h = (t_end - t_start) / N  # step size
t_values = np.linspace(t_start, t_end, N+1)

# Initial condition
state_0 = np.array([0.1, 0])  # small initial angle, zero initial velocity

# Solve the ODE using different methods and calculate MSE
state_values_rk = dict()
mse_values = dict()

for method, name in [(rk1, 'RK1'), (rk2, 'RK2'), (rk3, 'RK3'), (rk4, 'RK4')]:
    state_values = np.zeros((N+1, 2))
    state_values[0] = state_0
    for i in range(N):
        state_values[i+1] = method(state_values[i], t_values[i], h, pendulum)
    state_values_rk[name] = state_values
    # Calculate MSE for the angle
    mse_values[name] = mean_squared_error(np.sin(t_values), state_values[:, 0])


# Calculate MSE and plot the results
plt.figure(figsize=(6, 5))
plt.plot(t_values, y_exact(t_values), label='Exact solution', linestyle='--', color='k', lw=4)
for name in ['RK1', 'RK2', 'RK3', 'RK4']:
    mse_values[name] = mean_squared_error(y_exact(t_values), state_values_rk[name][:, 0])
    plt.plot(t_values, state_values_rk[name][:, 0], label=f'{name} (MSE: {mse_values[name]:.2e})')
plt.xlabel('t')
plt.ylabel('y (angle)')
plt.title(f"Solving $y'' = - \sin(y) \approx -y$ (pendulum) with step-size h={h}")
plt.legend()
plt.grid(True)
plt.ylim(-1.0, 1.0)
plt.tight_layout()
plt.savefig(f'runge_kutta_method_pendulum_h_{np.round(h, 2)}.png', dpi=200)
plt.show()
# %% SOLVE_IVP SOLUTION
def pendulum(t, state):
    y, v = state
    return [v, -y]  # Return the derivatives [dy/dt, dv/dt]

# Define the initial conditions and time span
t0 = 0.0
t_end = 10.0
y0 = 0.1  # Initial angle
v0 = 0.0  # Initial angular velocity

# Define the step size
step_size = 0.01

# Solve the differential equation using solve_ivp
sol = solve_ivp(pendulum, [t0, t_end], [y0, v0], max_step=step_size)

# Extract the solution
t_values = sol.t
y_values = sol.y[0]  # Angle values

# Calculate the exact solution
exact_solution = y0 * np.cos(t_values)

plt.figure(figsize=(6, 5))
plt.plot(t_values, exact_solution, label='Exact solution', c='k', lw=4)
plt.plot(t_values, y_values, label='Numerical solution (RK4/5)', c='cyan', ls='--', lw=2)
plt.xlabel('t')
plt.ylabel('y (angle)')
plt.title('Simple pendulum solution solved with solve_ivp')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('pendulum_solve_ivp.png', dpi=200)
plt.show()
# %% END
