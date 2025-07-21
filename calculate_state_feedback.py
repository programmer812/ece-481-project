import control as ct
import numpy as np
import math
import matplotlib.pyplot as plt

T = 0.1
desired_cts_poles = [-3, -3, -100]

A_d = np.array([[1, T, (T**2) / 2], [0, 1, T], [0, 0, 1]])
B_d = np.array([[(T**3) / 6], [(T**2) / 2], [T]])
C_d = np.array([1, 0, 0])

sys_d = ct.ss(A_d, B_d, C_d, 0)
C = ct.ctrb(A_d, B_d)
C_inv = np.linalg.inv(C)

print(f"system: {sys_d}")
print(f"Controllability matrix: {C}")
print(f"Inverse of controllability matrix: {C_inv}")
print(f"Rank of controllability matrix: {np.linalg.matrix_rank(C)}")
print(f"Rank of observability matrix: {np.linalg.matrix_rank(ct.obsv(A_d, C_d))}")

char_poly_coeffs = np.poly(np.linalg.eigvals(A_d))[1:]
A_bar = np.array([[0, 1, 0], [0, 0, 1], -1 * char_poly_coeffs[::-1]])
B_bar = np.array([[0], [0], [1]])
sys_bar = ct.ss(A_bar, B_bar, C_d, 0)

C_bar = ct.ctrb(A_bar, B_bar)
C_bar_inv = np.linalg.inv(C_bar)

print(f"Canonical form: {sys_bar}")
print(f"Canonical form controllability matrix: {C_bar}")
print(f"Inverse of canonical form controllability matrix: {C_bar_inv}")

V = C @ C_bar_inv
V_inv = np.linalg.inv(V)

print(f"V: {V}")
print(f"V inverse: {V_inv}")
print(f"A_bar = V_inv A V: {np.isclose(A_bar, V_inv @ A_d @ V)}")
print(f"B_bar = V_inv B: {np.isclose(B_bar, V_inv @ B_d)}")

desired_poles = [math.e ** (pole * T) for pole in desired_cts_poles]
desired_coeffs = np.poly(desired_poles)[1:]

F_bar = (char_poly_coeffs - desired_coeffs)[::-1]
F = F_bar @ V_inv

canonical_form_eigvals = np.linalg.eigvals(A_bar + (B_bar @ np.reshape(F_bar, (1, 3))))
final_eigvals = np.linalg.eigvals(A_d + (B_d @ np.reshape(F, (1, 3))))

print(f"Coefficients of desired characteristic polynomial: {desired_coeffs}")
print(f"Coefficients of characteristic polynomial: {char_poly_coeffs}")
print(f"F_bar: {F_bar}")
print(f"F: {F}")
print(f"Eigenvalues of A + BF: {canonical_form_eigvals}")
print(f"Final eigenvalues: {final_eigvals}")

A_final = A_d + (B_d @ np.reshape(F, (1, 3)))
sys_final = ct.ss(A_final, B_d, C_d, 0, dt=T)

print("System with controller:", sys_final)

time, response = ct.step_response(sys_final)

# Plot the step response
plt.figure(figsize=(10, 6))
plt.plot(time, response)
plt.title("Step Response")
plt.xlabel("Time [s]")
plt.ylabel("Output")
plt.grid(True)
plt.show()

info = ct.step_info(sys_final, SettlingTimeThreshold=0.01)

print(f"1% Settling Time: {info['SettlingTime']} seconds")
print(f"Percent Overshoot: {info['Overshoot']}%")
print(f"Rise Time: {info['RiseTime']}%")
print(f"Steady State: {info['SteadyStateValue']}%")
