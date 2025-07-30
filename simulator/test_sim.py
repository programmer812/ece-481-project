import numpy as np
from flapper_sim import FlapperSim
from controller import Controller, generate_trajectory
import time

T = 0.1

f = FlapperSim(robot_pose=np.array([1.0, 2.0, 0.8, 1.57]))
c = Controller()

y = f.get_output_measurement()
target = np.array([2, -2, 0.4])

p_trajectory, v_trajectory, a_trajectory = generate_trajectory(y, target)

for position, velocity, acceleration in zip(p_trajectory, v_trajectory, a_trajectory):
    time_loop_start = time.time()

    state_estimate, input = c.calculate_acceleration(
        y, position, velocity, acceleration, f.x
    )
    f.step(x=state_estimate, u=input)

    time_loop_end = time.time()
    time.sleep(max(0, T - (time_loop_end - time_loop_start)))

    y = f.get_output_measurement()
