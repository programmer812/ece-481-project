import control as ct
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import find_peaks

T = 0.1
SIMULATION_DURATION = 3
TIME_STEPS_VEC = np.arange(0, SIMULATION_DURATION, T)

desired_poles_cts = [-2.8 + 2.57j, -2.8 - 2.57j, -100]
desired_poles = [math.e ** (pole * T) for pole in desired_poles_cts]

# desired_poles_gamma = [pole - 1 for pole in desired_poles]
# coeff_mat = np.array([[-1, 0, 0], [0, -T, -0.5 * T**2], [0, 0, -(T**2)]])
# L = np.linalg.solve(coeff_mat, np.poly(desired_poles_gamma)[1:])

A_d = np.array([[1, T, (T**2) / 2], [0, 1, T], [0, 0, 1]])
B_d = np.array([[T**3 / 6], [T**2 / 2], [T]])
C_d = np.array([[1, 0, 0]])

L = -ct.place(A_d.T, C_d.T, desired_poles).T
print(f"L: {L}")
# print(f"L shape: {L.shape}")

u = np.ones_like(TIME_STEPS_VEC)  # unit step input

x = np.array([[0.0], [0.0], [0.0]])
y = C_d @ x
x_hat = -1.0 + 2.0 * np.random.rand(3, 1)

x_plot = []
x_hat_plot = []

for t in TIME_STEPS_VEC:
    x_hat = (A_d + L @ C_d) @ x_hat + B_d @ u[int(t) : int(t + 1)] - L @ y

    x = A_d @ x + B_d @ u[int(t) : int(t + 1)]
    y = C_d @ x

    x_plot.append(x)
    x_hat_plot.append(x_hat)

# B_obs = -L
# C_obs = np.array([1, 0, 0])

eigvals = np.linalg.eigvals(A_d + L @ C_d)
# print(f"Eigenvalues of A + LC: {eigvals}")
# print(f"Eigenvalues are close: {np.isclose(desired_poles, eigvals, atol=0.0)}")

# Compute estimation error (x_hat - x)
error = [x_hat_plot[i] - x_plot[i] for i in range(len(TIME_STEPS_VEC))]

# for i in range(3):
#     for j in range(3):
#         error_ij = np.array([e[i, j] for e in error])

#         # Overshoot
#         peaks, _ = find_peaks(error_ij)
#         if len(peaks) > 0:
#             peak_values = error_ij[peaks]
#             max_peak_relative_index = np.argmax(peak_values) # index in the peak_values array
#             max_peak_index = peaks[max_peak_relative_index] # index in the error_ij array

#             # print(x_plot[max_peak_index][i][j], error_ij[max_peak_index], x_hat_plot[max_peak_index][i][j])

#             max_peak = x_hat_plot[max_peak_index][i][j]
#             ss_value = x_plot[-1][i][j]

#             overshoot_percent = abs((max_peak - ss_value) / ss_value) * 100
#         else:
#             overshoot_percent = 0.0

#         # # Rise time
#         # initial_error = error0[0]
#         # rise_start_threshold = 0.9 * initial_error
#         # rise_end_threshold = 0.1 * initial_error
#         # rise_time = None
#         # for i, e in enumerate(error0):
#         #     if abs(e) < abs(rise_start_threshold) and rise_time is None:
#         #         rise_time = TIME_STEPS_VEC[i]

#         # # Settling time (1%)
#         # settling_time = None
#         # threshold = 0.01 * abs(x_plot[-1][0, 0])
#         # for i in reversed(range(len(error0))):
#         #     if abs(error0[i]) > threshold:
#         #         settling_time = TIME_STEPS_VEC[i + 1] if i + 1 < len(TIME_STEPS_VEC) else TIME_STEPS_VEC[i]
#         #         break

#         # # Steady-state error
#         # steady_state_error = error0[-1]

#         # Print specs
#         print("=== Estimation Performance ===")
#         print(f"Overshoot: {overshoot_percent:.2f}%")
#         # print(f"Rise time (approx): {rise_time:.2f} s")
#         # print(f"Settling time (1%): {settling_time:.2f} s")
#         # print(f"Steady-state error: {steady_state_error:.4f}")

plt.figure()
plt.plot(TIME_STEPS_VEC, [x[0][0] for x in x_plot], marker="o", ms=0.5, label="x1_x")
plt.plot(
    TIME_STEPS_VEC, [x[0][0] for x in x_hat_plot], marker="x", ms=0.5, label="x_hat1_x"
)
plt.grid()
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("State")
plt.show()

plt.figure()
plt.plot(TIME_STEPS_VEC, [x[1][0] for x in x_plot], marker="o", ms=0.5, label="x2_x")
plt.plot(
    TIME_STEPS_VEC, [x[1][0] for x in x_hat_plot], marker="x", ms=0.5, label="x_hat2_x"
)
plt.grid()
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("State")
plt.show()

plt.figure()
plt.plot(TIME_STEPS_VEC, [x[2][0] for x in x_plot], marker="o", ms=0.5, label="x3_x")
plt.plot(
    TIME_STEPS_VEC, [x[2][0] for x in x_hat_plot], marker="x", ms=0.5, label="x_hat3_x"
)
plt.grid()
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("State")
plt.show()
