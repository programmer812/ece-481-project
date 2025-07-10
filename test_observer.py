import numpy as np
import matplotlib.pyplot as plt
import control as ct

l1 = 0
l2 = 0
l3 = -1

T = 0.1

A_d = np.array([[1, T, (T**2) / 2], [0, 1, T], [0, 0, 1]])
B_d = np.array([[(T**3) / 6], [(T**2) / 2], [T]])
C_d = np.array([[0, 0, 1]])

L = np.array([[0], [0], [-1]])
F = np.array([[-67.17214498 - 51.83430747 - 12.47951264]])

A_obs = A_d + (L @ C_d) + (B_d @ F)
B_obs = -L
C_obs = np.array([0, 0, 1])  # change this to check each state one by one

sys = ct.ss(A_obs, B_obs, C_obs, 0, dt=T)
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
