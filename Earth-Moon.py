import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# --- 1. Constants for Earth-Moon System ---
G = 6.67430e-11
m_E = 5.97219e24
m_M = 7.342e22
L_EM = 3.844e8

# --- 2. Normalized Units and Mass Parameter ---
M_total = m_E + m_M
mu = m_M / M_total

omega = np.sqrt(G * M_total / L_EM**3)
T_unit = 1 / omega
V_unit = L_EM * omega

# --- 3. CRTBP Equations of Motion (Normalized) ---
# This function defines the differential equations for the Circular Restricted Three-Body Problem.
def crtbp_ode(t, state, mu):
    x, y, vx, vy = state

    # Calculate squared distances from the third body to Earth (primary) and Moon (secondary)
    r1_squared = (x + mu)**2 + y**2
    r2_squared = (x - (1 - mu))**2 + y**2

    # Calculate cubed distances (used in the acceleration terms)
    r1_cubed = r1_squared**(3/2)
    r2_cubed = r2_squared**(3/2)

    # Derivatives of the pseudo-potential U with respect to x and y.
    # U = 0.5 * (x^2 + y^2) + (1-mu)/r1 + mu/r2
    dU_dx = x - (1 - mu) * (x + mu) / r1_cubed - mu * (x - (1 - mu)) / r2_cubed
    dU_dy = y - (1 - mu) * y / r1_cubed - mu * y / r2_cubed

    # Equations of motion in the rotating frame
    ax = 2 * vy + dU_dx
    ay = -2 * vx + dU_dy

    return [vx, vy, ax, ay]

# --- 4. Calculate Lagrange Points L1 and L2 ---
# Lagrange points are equilibrium points where the gravitational and centrifugal forces balance.
# For L1, L2, and L3, they lie on the x-axis (y=0, z=0).
# Their positions are found by setting the derivative of the pseudo-potential with respect to x to zero.
def lagrange_equation(x, mu):
    # This equation is derived from dU/dx = 0
    return x - (1 - mu) * (x + mu) / np.abs(x + mu)**3 - mu * (x - (1 - mu)) / np.abs(x - (1 - mu))**3

x_L1_guess = 1 - mu - (mu/3)**(1/3)
x_L2_guess = 1 - mu + (mu/3)**(1/3)

L1_x = fsolve(lagrange_equation, x_L1_guess, args=(mu,))[0]
L2_x = fsolve(lagrange_equation, x_L2_guess, args=(mu,))[0]

L1_x_km = L1_x * L_EM / 1000
L2_x_km = L2_x * L_EM / 1000

# --- 5. Lyapunov Orbits (Approximation) ---
# Lyapunov orbits are periodic orbits around the collinear Lagrange points (L1, L2, L3).

# Desired y-amplitude for the orbits, based on the provided image (approx. 40,000 km).
y_amplitude_km = 40000
# Convert y-amplitude to normalized units.
y0_norm = y_amplitude_km / (L_EM / 1000)

# Estimated initial x-velocities (vx0) for L1 and L2 orbits.
# These values are sensitive and have been tuned to produce a good visual approximation.
vx0_L1_norm = -0.0053
vx0_L2_norm = -0.0052

# Period for integration in normalized time units.
# A typical Lyapunov orbit period in the Earth-Moon system is around 14-15 days.
# Normalized time unit (T_unit) is approximately 4.34 days.
# So, 14 days / 4.34 days/unit ≈ 3.22 normalized units.
T_orbit_norm = 3.22
# Simulate for 2.5 periods to clearly show the orbital path.
T_sim = T_orbit_norm * 2.5

# Initial conditions for the L1 Lyapunov orbit: [x0, y0, vx0, vy0]
# Starting at the Lagrange point's x-coordinate, with max y-displacement and corresponding vx.
initial_state_L1 = [L1_x, y0_norm, vx0_L1_norm, 0.0]

# Initial conditions for the L2 Lyapunov orbit: [x0, y0, vx0, vy0]
initial_state_L2 = [L2_x, y0_norm, vx0_L2_norm, 0.0]

# Time span for the numerical integration (from t=0 to T_sim).
t_span = (0, T_sim)

# Solve the Ordinary Differential Equations (ODEs) for both orbits.
# `solve_ivp` is used for robust numerical integration.
# `dense_output=True` allows for smooth interpolation of the solution.
# `rtol` and `atol` are relative and absolute tolerances for integration accuracy.
sol_L1 = solve_ivp(crtbp_ode, t_span, initial_state_L1, args=(mu,),
                   dense_output=True, rtol=1e-10, atol=1e-10)
sol_L2 = solve_ivp(crtbp_ode, t_span, initial_state_L2, args=(mu,),
                   dense_output=True, rtol=1e-10, atol=1e-10)

# Generate a finely spaced time array for plotting the integrated orbits smoothly.
t_plot = np.linspace(t_span[0], t_span[1], 1000)
# Evaluate the solutions at these time points.
orbit_L1 = sol_L1.sol(t_plot)
orbit_L2 = sol_L2.sol(t_plot)

# Convert normalized coordinates of the orbits to kilometers for plotting.
x_L1_km = orbit_L1[0, :] * L_EM / 1000
y_L1_km = orbit_L1[1, :] * L_EM / 1000

x_L2_km = orbit_L2[0, :] * L_EM / 1000
y_L2_km = orbit_L2[1, :] * L_EM / 1000

# Calculate Moon's position in kilometers (Earth is at -mu * L_EM, Moon at (1-mu) * L_EM).
x_Moon_km = (1 - mu) * L_EM / 1000

# --- 6. Plotting ---
plt.figure(figsize=(10, 7)) # Set figure size for better visualization

# Plot the L1 and L2 Lyapunov orbits.
plt.plot(x_L1_km, y_L1_km, color='red', linewidth=2, label='L1 Lyapunov Orbit')
plt.plot(x_L2_km, y_L2_km, color='blue', linewidth=2, label='L2 Lyapunov Orbit')

# Plot the Moon's position.
plt.scatter(x_Moon_km, 0, color='gray', s=200, marker='o', label='Moon', zorder=5) # Moon as a filled circle

# Plot the Lagrange points EML1 and EML2.
plt.scatter(L1_x_km, 0, color='red', marker='+', s=300, linewidth=3, label='EML1', zorder=5)
plt.scatter(L2_x_km, 0, color='blue', marker='*', s=300, linewidth=3, label='EML2', zorder=5)

# Add text labels for Moon, EML1, EML2 to match the image.
plt.text(x_Moon_km + 0.05 * L_EM / 1000, 0, 'Moon', color='gray', ha='left', va='center', fontsize=14, fontweight='bold')
plt.text(L1_x_km, 0.05 * L_EM / 1000, 'EML1', color='red', ha='center', va='bottom', fontsize=14, fontweight='bold')
plt.text(L2_x_km, 0.05 * L_EM / 1000, 'EML2', color='blue', ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add "To Earth" arrow and text, mimicking the image's style and position.
# Define plot limits to match the image.
x_min, x_max = 2.9e5, 4.7e5
y_min, y_max = -0.5e5, 0.5e5

# Calculate positions for the arrow and text based on plot limits.
arrow_x_text = x_min + 0.15 * (x_max - x_min) # Text position (right of arrow tip)
arrow_x_tip = x_min + 0.05 * (x_max - x_min)  # Arrow tip position (leftmost)
arrow_y_pos = y_max - 0.08 * (y_max - y_min) # Y position for arrow and text

plt.annotate('To Earth', xy=(arrow_x_tip, arrow_y_pos), xytext=(arrow_x_text, arrow_y_pos),
             arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=10, connectionstyle='arc3,rad=0'),
             color='red', fontsize=14, ha='left', va='center', fontweight='bold')


# Set plot limits as per the provided image.
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Set axis labels.
plt.xlabel('x [km]', fontsize=14)
plt.ylabel('y [km]', fontsize=14)
# Set plot title.
plt.title('Sample Planar Lyapunov Orbits in the Earth-Moon System', fontsize=16, pad=20)
# Add a grid for better readability.
plt.grid(True, linestyle='--', alpha=0.6)
# Ensure equal aspect ratio to prevent distortion of orbits.
plt.gca().set_aspect('equal', adjustable='box')

# Format tick labels to use scientific notation (x 10^5) as seen in the image.
plt.ticklabel_format(axis='x', style='sci', scilimits=(5,5))
plt.ticklabel_format(axis='y', style='sci', scilimits=(5,5))

# Display the plot.
plt.show()

print("\nNote: The generated Lyapunov orbits are approximations based on chosen initial conditions.")
print("Achieving perfectly closed Lyapunov orbits typically requires advanced numerical methods like differential correction or continuation methods.")
print("The plot aims to visually resemble the provided image.")
