import control as ct
import numpy as np
import matplotlib.pyplot as plt

T = 0.1
desired_poles = [0, 0, 0]
desired_poles_gamma = [pole - 1 for pole in desired_poles]

coeff_mat = np.array([[-1, 0, 0], [0, -T, -0.5 * T**2], [0, 0, -(T**2)]])

L = np.linalg.solve(coeff_mat, np.poly(desired_poles_gamma)[1:])
print(f"L: {L}")

A_d = np.array([[1, T, (T**2) / 2], [0, 1, T], [0, 0, 1]])
C_d = np.array([[1, 0, 0]])

A_obs = A_d + (np.reshape(L, (3, 1)) @ C_d)
C_obs = np.array([0, 0, 1])  # change this to check each state one by one

eigvals = np.linalg.eigvals(A_obs)
print(f"Eigenvalues of A+LC: {eigvals}")

sys = ct.ss(A_obs, np.zeros((3, 1)), C_obs, 0, dt=T)
print(sys)

time, response = ct.step_response(sys)

# Plot the step response
plt.figure(figsize=(10, 6))
plt.plot(time, response)
plt.title("Step Response")
plt.xlabel("Time [s]")
plt.ylabel("Output")
plt.grid(True)
plt.show()

info = ct.step_info(sys, SettlingTimeThreshold=0.01)

print(f"2% Settling Time: {info['SettlingTime']} seconds")
print(f"Percent Overshoot: {info['Overshoot']}%")
print(f"Rise Time: {info['RiseTime']}%")
print(f"Steady State: {info['SteadyStateValue']}%")
