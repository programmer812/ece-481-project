import numpy as np
from flapper_sim import FlapperSim
from controller import Controller, generate_trajectory
import time

T = 0.1

f = FlapperSim(robot_pose=np.array([1.0, 2.0, 0.8, 1.57]))
# c = Controller(f)
c = Controller()

y = f.get_output_measurement()
target = np.array([2, -2, 0.4])

reference = generate_trajectory(y, target)

for ref_point in reference:
    time_loop_start = time.time()

    input, state_estimate = c.calculate_acceleration(y, ref_point)
    f.step(x=state_estimate, u=input)

    time_loop_end = time.time()
    time.sleep(max(0, T - (time_loop_end - time_loop_start)))

    y = f.get_output_measurement()
