import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# --- Matplotlib Font and Global Settings ---
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

font = {'family': 'Times New Roman',
    # 'weight': 'bold',
    'size': 16}


mpl.rc('font', **font)


# ==============================================================================
# Part 1: Define Key Parameters for the Conceptual Illustration
# (These values are used only for visualization and do not affect the labels.)
# ==============================================================================


# Transition point on the x-axis
T_star_val = 50

# Key values on the y-axis
y_start_val = 10
y_saturate_val = 100

# Maximum value on the x-axis
T_max_val = 80



# ==============================================================================
# Part 2: Generate Plotting Data
# ==============================================================================

# --- Our result (blue line: initially increasing, then saturating) ---

# Increasing segment
x_our_rising = np.array([1, T_star_val])
y_our_rising = np.array([y_start_val, y_saturate_val])

# Saturation segment
x_our_flat = np.array([T_star_val, T_max_val])
y_our_flat = np.array([y_saturate_val, y_saturate_val])

# --- Existing result (red line: continuously increasing) ---

# Calculate the slope of the existing result
slope = (y_saturate_val - y_start_val) / (T_star_val - 1)

# Calculate the y-value at T_max
y_red_end = y_start_val + slope * (T_max_val - 1)

x_existing = np.array([1, T_max_val])
y_existing = np.array([y_start_val, y_red_end])



# ==============================================================================
# Part 3: Plotting and Formatting
# ==============================================================================
fig, ax = plt.subplots()

# Plot our result
ax.plot(x_our_rising, y_our_rising, '-', color='darkblue', linewidth=2, label='Ours')
ax.plot(x_our_flat, y_our_flat, '-', color='darkblue', linewidth=2)

# Apply a vertical offset to improve visual separation between the curves
offset = 1.5
y_existing_offset = y_existing + offset

# Plot the existing result
ax.plot(x_existing, y_existing_offset, '--', color='darkred', linewidth=2, label='Liu et al. (2024)')


# --- Configure axis labels and grid ---
ax.set_xlabel('Communication round $T$')
ax.set_ylabel(r"Privacy budget $\epsilon^{\prime}$")
ax.grid(True)

fig.tight_layout()


# --- Configure x-axis ticks and labels ---
x_ticks = [1, T_star_val]
x_labels = ['1', r'$\frac{\Phi}{\phi}$']

ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels)

# --- Configure y-axis ticks and labels ---
y_ticks = [y_start_val, y_saturate_val]
y_labels = [r'$\mathcal{O}(\frac{\alpha r c^2}{\sigma^2}\phi)$', r'$\mathcal{O}(\frac{\alpha r c^2}{\sigma^2}\Phi)$']

ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)


# --- Configure the legend ---
ax.legend(loc='upper left', fontsize = 14, bbox_to_anchor=(0.035, 0.98))

# --- Save and display the figure ---
plt.savefig('figure_2.pdf', format='pdf', dpi=600, bbox_inches='tight')

plt.show()