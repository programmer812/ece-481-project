import matplotlib.pyplot as plt
import json

T = 0.1

with open("log.json") as f:
    log = json.load(f)

for state_component in ["p", "v", "a"]:
    for space_component in range(3):
        trajectory = [
            point[space_component] for point in log["trajectory"][state_component]
        ]
        actual = [point[space_component] for point in log["actual"][state_component]]

        assert len(actual) == len(trajectory)
        times = [i * T for i in range(len(actual))]

        plt.plot(times, trajectory, label="trajectory")
        plt.plot(times, actual, label="actual")
        plt.legend()
        plt.title(
            f"{state_component} - {'x' if space_component == 0 else 'y' if space_component == 1 else 'z'}"
        )
        plt.show()
