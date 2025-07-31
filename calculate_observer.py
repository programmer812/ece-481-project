import numpy as np
import matplotlib.pyplot as plt
import math

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

for idx, component in enumerate(["position", "velocity", "acceleration"]):
    print(f"Component {component}")

    error = [
        (estimated_xvals[i][idx] - xvals[i][idx])[0].item()
        for i in range(len(TIME_STEPS_VEC))
    ]

    start_val = error[0]
    start_val_sign = math.copysign(1, start_val)
    step_size = abs(start_val)
    print(f"Step size: {step_size}")

    peak_idx = min(
        list(range(len(TIME_STEPS_VEC))), key=lambda i: start_val_sign * error[i]
    )
    overshoot_amount = abs(error[peak_idx])
    overshoot_percent = overshoot_amount / step_size * 100
    print(f"Overshoot: {overshoot_percent}%")

    low_threshold = 0.9 * step_size
    passed_low_threshold = [abs(e) <= low_threshold for e in error]
    low_idx = passed_low_threshold.index(True)

    high_threshold = 0.1 * step_size
    passed_high_threshold = [abs(e) <= high_threshold for e in error]
    high_idx = passed_high_threshold.index(True)

    rise_time = TIME_STEPS_VEC[high_idx] - TIME_STEPS_VEC[low_idx]
    print(f"Rise Time: {rise_time}")

    settling_threshold = 0.01 * step_size
    passed_settling_threshold = [abs(e) <= settling_threshold for e in error]
    settling_idx = len(passed_settling_threshold) - passed_settling_threshold[
        ::-1
    ].index(False)
    settling_time = TIME_STEPS_VEC[settling_idx]
    print(f"Settling Time: {settling_time}")

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

    # plt.axhline(y=error[peak_idx], linestyle=":", label="peak", color="green")
    # plt.axhline(y=low_threshold, linestyle=":", label="10%", color="pink")
    # plt.axhline(y=high_threshold, linestyle=":", label="90%", color="red")
    # plt.axhline(y=settling_threshold, linestyle=":", label="1%", color="purple")
    # plt.axhline(y=-settling_threshold, linestyle=":", label="1%", color="purple")

    plt.grid()
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("State")
    plt.title(component)

    plt.show()
