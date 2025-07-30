import control as ct
import numpy as np
import matplotlib.pyplot as plt
import math

T = 0.1
desired_poles_cts = [-2.8 + 2.57j, -2.8 - 2.57j, -100]
desired_poles = [math.e**pole for pole in desired_poles_cts]

# desired_poles_gamma = [pole - 1 for pole in desired_poles]
# coeff_mat = np.array([[-1, 0, 0], [0, -T, -0.5 * T**2], [0, 0, -(T**2)]])
# L = np.linalg.solve(coeff_mat, np.poly(desired_poles_gamma)[1:])

A_d = np.array([[1, T, (T**2) / 2], [0, 1, T], [0, 0, 1]])
C_d = np.array([[1, 0, 0]])

L = -ct.place(A_d.T, C_d.T, desired_poles).T
print(f"L: {L}")
L = np.reshape(L, (3, 1))

A_obs = A_d + L @ C_d
B_obs = -L
C_obs = np.array([1, 0, 0])

eigvals = np.linalg.eigvals(A_obs)
print(f"Eigenvalues of A+LC: {eigvals}")
print(f"eigenalues are close: {np.isclose(desired_poles, eigvals, atol=0.0)}")

sys = ct.ss(A_obs, B_obs, C_obs, 0, dt=T)
print(sys)

time, response = ct.step_response(sys)
info = ct.step_info(sys, SettlingTimeThreshold=0.01)

print(f"1% Settling Time: {info['SettlingTime']} seconds")
print(f"Percent Overshoot: {info['Overshoot']}%")
print(f"Rise Time: {info['RiseTime']}%")
print(f"Steady State: {info['SteadyStateValue']}%")

plt.figure(figsize=(10, 6))
plt.plot(time, response)
plt.title("Step Response")
plt.xlabel("Time [s]")
plt.ylabel("Output")
plt.grid(True)
plt.show()
