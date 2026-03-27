import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# --- 1. Physics Configuration ---
G = 9.81      # Gravity (m/s^2)
L1, L2 = 1.0, 1.0  # Lengths (m)
M1, M2 = 1.0, 1.0  # Masses (kg)
FPS = 60      # Animation speed
TRAIL_LEN = 120    # Number of points in the fading trail

def get_derivs(t, state):
    """Calculates derivatives for the double pendulum system."""
    th1, w1, th2, w2 = state
    
    delta = th2 - th1
    den1 = (M1 + M2) * L1 - M2 * L1 * np.cos(delta)**2
    den2 = (L2 / L1) * den1
    
    # Angular accelerations
    d_w1 = (M2*L1*w1**2*np.sin(delta)*np.cos(delta) + 
            M2*G*np.sin(th2)*np.cos(delta) + 
            M2*L2*w2**2*np.sin(delta) - 
            (M1+M2)*G*np.sin(th1)) / den1
            
    d_w2 = (-M2*L2*w2**2*np.sin(delta)*np.cos(delta) + 
            (M1+M2)*G*np.sin(th1)*np.cos(delta) - 
            (M1+M2)*L1*w1**2*np.sin(delta) - 
            (M1+M2)*G*np.sin(th2)) / den2
            
    return [w1, d_w1, w2, d_w2]

# --- 2. Simulation Setup ---
t_eval = np.linspace(0, 30, 30 * FPS)
# Initial angles (rad) and velocities (rad/s)
init_state = [np.pi/2, 0, np.pi/2, 0] 

sol = solve_ivp(get_derivs, (0, 30), init_state, t_eval=t_eval, method='RK45')
th1, th2 = sol.y[0], sol.y[2]

# Convert to Cartesian coordinates
x1 = L1 * np.sin(th1)
y1 = -L1 * np.cos(th1)
x2 = x1 + L2 * np.sin(th2)
y2 = y1 - L2 * np.cos(th2)

# --- 3. Animation & Styling ---
fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
ax.set_facecolor('black')
ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
ax.set_aspect('equal')
ax.axis('off')

# Pendulum arms (Magenta and Cyan neon)
arm1, = ax.plot([], [], color='#FF00FF', lw=4, solid_capstyle='round', zorder=5)
arm2, = ax.plot([], [], color='#00FFFF', lw=4, solid_capstyle='round', zorder=5)

# Trajectory trail using a list of line segments for alpha fading
trail_segs = [ax.plot([], [], color='#00FFCC', lw=2, alpha=0)[0] for _ in range(TRAIL_LEN)]

def update(i):
    # Update Arms
    arm1.set_data([0, x1[i]], [0, y1[i]])
    arm2.set_data([x1[i], x2[i]], [y1[i], y2[i]])
    
    # Update Trail (fading segments)
    start_idx = max(0, i - TRAIL_LEN)
    for idx, frame_pos in enumerate(range(start_idx, i)):
        # Linear alpha fade-out for older segments
        alpha = (idx / TRAIL_LEN) ** 1.5
        trail_segs[idx].set_data(x2[frame_pos:frame_pos+2], y2[frame_pos:frame_pos+2])
        trail_segs[idx].set_alpha(alpha)
        
    return [arm1, arm2] + trail_segs

ani = FuncAnimation(fig, update, frames=len(t_eval), interval=1000/FPS, blit=True)
plt.show()
