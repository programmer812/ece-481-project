import numpy as np
from controller import generate_trajectory
import matplotlib.pyplot as plt
from scipy.signal import medfilt

T = 0.1

current = np.array([1.0, 2.0, 0.8])
target = np.array([2, -2, 0.4])

reference, v_trajectory, a_trajectory = generate_trajectory(current, target)
assert len(reference) == len(v_trajectory) == len(a_trajectory)

print(f"Final position: {reference[-1]}")
print(f"Should be: {target}")

reference_array = np.array(reference)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot(
    reference_array[:, 0],
    reference_array[:, 1],
    reference_array[:, 2],
    label="Reference Trajectory",
)

ax.scatter(
    reference_array[0, 0],
    reference_array[0, 1],
    reference_array[0, 2],
    color="green",
    label="Start",
    s=50,
)
ax.scatter(
    reference_array[-1, 0],
    reference_array[-1, 1],
    reference_array[-1, 2],
    color="red",
    label="Target",
    s=50,
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Generated Reference Trajectory")
ax.legend()
plt.tight_layout()
plt.show()

for component_idx, component in enumerate(["x", "y", "z"]):
    ref = [point[component_idx][0] for point in reference]
    v_traj = [point[component_idx][0] for point in v_trajectory]
    a_traj = [point[component_idx][0] for point in a_trajectory]

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

    plt.plot(times, v_traj)
    plt.title(f"velocity reference - {component}")
    plt.show(block=True)

    plt.plot(times[:-2], a)
    plt.title(f"acceleration - {component}")
    plt.show(block=True)

    plt.plot(times, a_traj)
    plt.title(f"acceleration reference - {component}")
    plt.show(block=True)
