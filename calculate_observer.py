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
print(f"Eigenvalues of A + LC: {eigvals}")
print(f"Eigenvalues are close: {np.isclose(desired_poles, eigvals, atol=0.0)}")

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
