import numpy as np
from flapper import Flapper
from controller import Controller, generate_trajectory
import time

T = 0.1

f = Flapper(backend_server_ip='192.168.0.102')
c = Controller()

y = f.get_output_measurement()
target = np.array([0, 0, 0.5])

p_trajectory, v_trajectory, a_trajectory = generate_trajectory(y, target)

for position, velocity, acceleration in zip(p_trajectory, v_trajectory, a_trajectory):
    time_loop_start = time.time()

    state_estimate, input = c.calculate_acceleration(
        y, position, velocity, acceleration, f.x
    )
    f.step(x=state_estimate, u=np.array(input))

    time_loop_end = time.time()
    time.sleep(max(0, T - (time_loop_end - time_loop_start)))

    y = f.get_output_measurement()