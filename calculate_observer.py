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

u = np.ones_like(TIME_STEPS_VEC)

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

eigvals = np.linalg.eigvals(A_d + L @ C_d)
print(f"Eigenvalues of A + LC: {eigvals}")
print(f"Eigenvalues are close: {np.isclose(desired_poles, eigvals, atol=0.0)}")

error = [x_hat_plot[i] - x_plot[i] for i in range(len(TIME_STEPS_VEC))]

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
