import numpy as np

T = 0.1

A_d = np.array([[1, T, (T**2) / 2], [0, 1, T], [0, 0, 1]])
B_d = np.array([[(T**3) / 6], [(T**2) / 2], [T]])
C_d = np.array([[1, 0, 0]])

L = np.array([[-1.53803172], [-5.92698271], [-10.92812269]])
F = np.array([[-67.17214498, -51.83430747, -12.47951264]])

trajectory_time = 20
start_time = 1
end_time = 5


def generate_trajectory(current, target):
    current = np.reshape(current, (3, 1))
    target = np.reshape(target, (3, 1))

    a = 4 * np.linalg.norm(target - current) / (trajectory_time**2)
    a_with_dir = a * (target - current) / np.linalg.norm(target - current)

    k = int(trajectory_time / (2 * T))

    positions_part1 = [current + 0.5 * ((i * T) ** 2) * a_with_dir for i in range(k)]
    velocities_part1 = [i * T * a_with_dir for i in range(k)]

    half_position = current + 0.5 * (target - current)
    half_velocity = k * T * a_with_dir

    velocities_part3 = [half_velocity - i * T * a_with_dir for i in range(k)]
    positions_part3 = [
        half_position + i * T * half_velocity - 0.5 * ((i * T) ** 2) * a_with_dir
        for i in range(k)
    ]

    pos_trajectory = (
        [current] * int(start_time / T)
        + positions_part1
        + positions_part3
        + [target] * int(end_time / T)
    )

    v_trajectory = (
        [np.array([[0], [0], [0]])] * int(start_time / T)
        + velocities_part1
        + velocities_part3
        + [np.array([[0], [0], [0]])] * int(end_time / T)
    )

    accelerations_part1 = [a_with_dir] * len(velocities_part1)
    accelerations_part3 = [-a_with_dir] * len(velocities_part3)
    a_trajectory = (
        [np.array([[0], [0], [0]])] * int(start_time / T)
        + accelerations_part1
        + accelerations_part3
        + [np.array([[0], [0], [0]])] * int(end_time / T)
    )

    return pos_trajectory, v_trajectory, a_trajectory


class Controller:
    def __init__(self):
        self.state_estimate = [np.array([[0], [0], [0]]) for _ in range(3)]
        self.u = [0, 0, 0]

    def calculate_acceleration(self, y, position, velocity, acceleration):
        y_hats = [C_d @ x_hat for x_hat in self.state_estimate]

        self.state_estimate = [
            A_d @ x_hat + (u * B_d) + (y_hat[0] - output_coord)[0] * L
            for x_hat, u, y_hat, output_coord in zip(
                self.state_estimate, self.u, y_hats, y
            )
        ]

        desired_state = np.column_stack((position, velocity, acceleration))

        self.u = [
            -F @ (np.reshape(target, (3, 1)) - x_hat)
            for x_hat, target in zip(self.state_estimate, desired_state)
        ]

        return np.array(self.state_estimate).reshape((3, 3)).T.flatten(), self.u
