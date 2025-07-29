import numpy as np
from flapper_sim import FlapperSim
from controller import Controller, generate_trajectory

f = FlapperSim(robot_pose=np.array([1.0, 2.0, 0.8, 1.57]))
c = Controller()

y = f.get_output_measurement()
target = np.array([0.5, 0.5, 0.5])

# plot to check correct - want piecewise constant acceleration
reference = generate_trajectory(y, target)

for ref_point in reference:
    u = c.calculate_acceleration(y, ref_point)
    f.step(u=u)
    y = f.get_output_measurement()
