import numpy as np
from controller import generate_trajectory
import matplotlib.pyplot as plt
from scipy.signal import medfilt

T = 0.1

current = np.array([1.0, 2.0, 0.8])
target = np.array([2, -2, 0.4])

reference = generate_trajectory(current, target)
print(f"Final position: {reference[-1]}")
print(f"Should be: {target}")

for component_idx, component in enumerate(["x", "y", "z"]):
    ref = [point[component_idx][0] for point in reference]
    times = [T * i for i in range(len(ref))]

    v = [ref[i + 1] - ref[i] for i in range(len(ref) - 1)]
    v = medfilt(v, kernel_size=3)

    a = [v[i + 1] - v[i] for i in range(len(v) - 1)]
    a = medfilt(a, kernel_size=3)

    plt.plot(times, ref)
    plt.title(f"position - {component}")
    plt.show(block=True)

    plt.plot(times[:-1], v)
    plt.title(f"velocity - {component}")
    plt.show(block=True)

    plt.plot(times[:-2], a)
    plt.title(f"acceleration - {component}")
    plt.show(block=True)
