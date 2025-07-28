import time
import numpy as np
from flapper import Flapper

f = Flapper(backend_server_ip="192.168.0.2")

while True:
    y = f.get_output_measurement()

    print('[student code, main script] output measurement', y)
    
    estimated_state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    control_input = np.array([1.0, 2.0, 3.0])

    f.step(x=estimated_state, u=control_input)