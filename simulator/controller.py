import numpy as np
import math

T = 0.1

A_d = np.array([[1, T, (T**2) / 2], [0, 1, T], [0, 0, 1]])
B_d = np.array([[(T**3) / 6], [(T**2) / 2], [T]])
C_d = np.array([[1, 0, 0]])

L = np.array([[-3], [-25], [-100]])
F = np.array([[-67.17214498, -51.83430747, -12.47951264]])

controller_ss_gain = 0.01488712
acceleration = 0.01  # m/s


def generate_trajectory(current, target):
    current = np.reshape(current, (3, 1))
    target = np.reshape(target, (3, 1))

    a_with_dir = acceleration * (target - current) / np.linalg.norm(target - current)
    k = int(math.sqrt(np.linalg.norm(target - current) / acceleration) / T)

    velocities_part1 = [i * T * a_with_dir for i in range(k)]
    positions_part1 = [
        current + i * T * velocity + ((i * T) ** 2) / 2 * a_with_dir
        for i, velocity in enumerate(velocities_part1)
    ]

    half_velocity = k * T * a_with_dir
    half_position = current + 0.5 * (target - current)

    velocities_part2 = [half_velocity + i * T * (-a_with_dir) for i in range(k)]
    positions_part2 = [
        half_position + i * T * velocity + ((i * T) ** 2) / 2 * (-a_with_dir)
        for i, velocity in enumerate(velocities_part2)
    ]

    trajectory = (
        [current] * int(1 / T)
        + positions_part1
        + positions_part2
        + [target] * int(1 / T)
    )
    return [point / controller_ss_gain for point in trajectory]


class Controller:
    def __init__(self):
        self.state_estimate = [np.array([[0], [0], [0]]) for _ in range(3)]
        self.u = [0, 0, 0]

    def calculate_acceleration(self, y, ref):
        y_hats = [C_d @ x_hat for x_hat in self.state_estimate]

        self.state_estimate = [
            A_d @ x_hat + (u * B_d) + L @ (y_hat - output_coord)
            for x_hat, u, y_hat, output_coord in zip(
                self.state_estimate, self.u, y_hats, y
            )
        ]
        self.u = [
            ref_coord + F @ x_hat for x_hat, ref_coord in zip(self.state_estimate, ref)
        ]

        return np.array(self.state_estimate).reshape((3, 3)).T.flatten(), self.u
