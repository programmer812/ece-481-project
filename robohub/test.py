import numpy as np
from flapper import Flapper
from controller import Controller, generate_trajectory
import time
import json

T = 0.1

f = Flapper(backend_server_ip='192.168.0.102')
c = Controller()

y = f.get_output_measurement()
target = np.array([0, 0, 0.5])

p_trajectory, v_trajectory, a_trajectory = generate_trajectory(y, target)

actual_p = []
actual_v = []
actual_a = []

for position, velocity, acceleration in zip(p_trajectory, v_trajectory, a_trajectory):
    time_loop_start = time.time()

    state_estimate, input = c.calculate_acceleration(
        y, position, velocity, acceleration
    )
    f.step(x=state_estimate, u=input)
    
    actual_p.append(np.copy(f.x[0:3]))
    actual_v.append(np.copy(f.x[3:6]))
    actual_a.append(np.copy(f.x[6:9]))

    time_loop_end = time.time()
    time.sleep(max(0, T - (time_loop_end - time_loop_start)))

    y = f.get_output_measurement()
    
p_trajectory = np.reshape(p_trajectory, (len(p_trajectory), 3)).tolist()
v_trajectory = np.reshape(v_trajectory, (len(v_trajectory), 3)).tolist()
a_trajectory = np.reshape(a_trajectory, (len(a_trajectory), 3)).tolist()

actual_p = np.reshape(actual_p, (len(actual_p), 3)).tolist()
actual_v = np.reshape(actual_v, (len(actual_v), 3)).tolist()
actual_a = np.reshape(actual_a, (len(actual_a), 3)).tolist()

with open("log.json", "w") as f:
    json.dump(
        {
            "trajectory": {"p": p_trajectory, "v": v_trajectory, "a": a_trajectory},
            "actual": {"p": actual_p, "v": actual_v, "a": actual_a},
        },
        f,
    )