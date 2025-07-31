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

desired_poles_gamma = [pole - 1 for pole in desired_poles]
coeff_mat = np.array([[-1, 0, 0], [0, -T, -0.5 * T**2], [0, 0, -(T**2)]])

L = np.reshape(np.linalg.solve(coeff_mat, np.poly(desired_poles_gamma)[1:]), (3, 1))
print(f"L: {L}")

A_d = np.array([[1, T, (T**2) / 2], [0, 1, T], [0, 0, 1]])
B_d = np.array([[T**3 / 6], [T**2 / 2], [T]])
C_d = np.array([[1, 0, 0]])

eigvals = np.linalg.eigvals(A_d + L @ C_d)
# print(f"Eigenvalues of A + LC: {eigvals}")
# print(f"Eigenvalues are close: {np.isclose(desired_poles, eigvals, atol=0.0)}")

# u = np.ones_like(TIME_STEPS_VEC) # unit step input

x = np.array([[0.0], [0.0], [0.0]])
y = C_d @ x
x_hat = np.array([[-1.0], [-1.0], [-1.0]])

xvals = []
estimated_xvals = []

for t in TIME_STEPS_VEC:
    x_hat = (A_d + L @ C_d) @ x_hat - L @ y

    x = A_d @ x
    y = C_d @ x

    xvals.append(x)
    estimated_xvals.append(x_hat)

error = [estimated_xvals[i] - xvals[i] for i in range(len(TIME_STEPS_VEC))]

for i in range(3):
    error_i = np.array([e[i, 0] for e in error])

    # Overshoot
    peaks, _ = find_peaks(error_i)
    if len(peaks) > 0:
        peak_values = error_i[peaks]
        max_peak_relative_index = np.argmax(peak_values) # index in the peak_values array
        max_peak_index = peaks[max_peak_relative_index] # index in the error_i array

        max_peak = estimated_xvals[max_peak_index][i]

        overshoot_percent = abs(max_peak[0]) * 100
    else:
        overshoot_percent = 0.0

    # Rise time
    initial_error = error_i[i]
    rise_start_threshold = 0.9 * initial_error
    rise_end_threshold = 0.1 * initial_error
    rise_time = None
    for i, e in enumerate(error_i):
        if abs(e) < abs(rise_start_threshold) and rise_time is None:
            rise_time = TIME_STEPS_VEC[i]

    # Settling time (1%)
    settling_time = None
    threshold = 0.01
    for i in reversed(range(len(error_i))):
        if abs(error_i[i]) > threshold:
            if i + 1 < len(TIME_STEPS_VEC):
                settling_time = TIME_STEPS_VEC[i + 1]
            else:
                settling_time = TIME_STEPS_VEC[i]
            break

    # Steady-state error
    steady_state_error = abs(error_i[-1])

    # Print specs
    print("=== Estimation Performance ===")
    print(f"Overshoot: {overshoot_percent:.2f}%")
    print(f"Rise time (approx): {rise_time:.2f} s")
    print(f"Settling time (1%): {settling_time:.2f} s")
    print(f"Steady-state error: {steady_state_error:.4f}")

for idx, component in enumerate(["x", "y", "z"]):
    plt.figure()
    plt.plot(
        TIME_STEPS_VEC,
        [x[idx][0] for x in xvals],
        label="state",
    )
    plt.plot(
        TIME_STEPS_VEC,
        [x[idx][0] for x in estimated_xvals],
        label="estimated state",
    )
    plt.grid()
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("State")
    plt.title(component)
    plt.show()
