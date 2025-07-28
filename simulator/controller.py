import numpy as np

T = 0.1

A_d = np.array([[1, T, (T**2) / 2], [0, 1, T], [0, 0, 1]])
B_d = np.array([[(T**3) / 6], [(T**2) / 2], [T]])
C_d = np.array([[0, 0, 1]])

L = np.array([[0], [0], [-1]])
F = np.array([[-67.17214498, -51.83430747, -12.47951264]])


class Controller:
    def __init__(self):
        self.state_estimate = [np.array([[0], [0], [0]]) for i in range(3)]
        self.u = [0, 0, 0]

    def calculate_acceleration(self, y, ref):
        # does reference generation take place here or in test_sim?
        # adjust the reference to compensate for the steady state
        # subtract off the reference in the calculation of u_new

        u_new = [F @ x_hat for x_hat in self.state_estimate]
        self.state_estimate = [
            (A_d + (L @ C_d)) @ x_hat + u * B_d - coord * L
            for x_hat, u, coord in zip(self.state_estimate, self.u, y)
        ]

        self.u = u_new
        return self.u
