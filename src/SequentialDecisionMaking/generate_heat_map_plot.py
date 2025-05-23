"""
Created on Thu Nov  7 09:30:54 2024

@author: Jonas Petersen
"""
#%%
import numpy as np
import pickle
import matplotlib.pyplot as plt


with open("./src/SequentialDecisionMaking/costs_baseline.pkl", "rb") as f:
    costs_baseline = pickle.load(f)

with open("./src/SequentialDecisionMaking/costs_by_hand.pkl", "rb") as f:
    costs_by_hand = pickle.load(f)

holding_costs = np.arange(1, 20+1, 1)
unit_values = np.arange(1, 100+1, 1)

# Calculate heatmap values
heatmap_values = costs_by_hand[1:,1:] / costs_baseline[1:,1:]

# Plotting the heatmap
plt.figure(figsize=(10, 8))
plt.imshow(heatmap_values, cmap="Spectral", aspect="auto", origin="lower")
cbar = plt.colorbar(shrink=0.8)

# Set custom ticks for colorbar
custom_ticks = [0,0.1,0.2,0.3,0.4,0.5,0.6, 0.7, 0.8, 0.9, 1.0]
cbar.set_ticks(custom_ticks)  # Set the tick positions
cbar.set_ticklabels([f"{tick:.1f}" for tick in custom_ticks])  # Set custom labels if desired
cbar.set_label(r"$\frac{\mathbb{E}[C^*|D,I]}{\mathbb{E}[C^{(1)}|D,I]}$")
plt.xticks(ticks=np.arange(len(holding_costs)), labels=holding_costs)
plt.yticks(ticks=np.arange(len(unit_values), step=3), labels=unit_values[::3])
plt.xlabel(r"$h_t$ (Holding Cost)")
plt.ylabel(r"$c_t$ (Unit Value)")
plt.tight_layout()
plt.show()
plt.savefig('./docs/SequentialDecisionMaking/figures/numerical_per.pdf')


#%%


import numpy as np
import matplotlib.pyplot as plt


# Define the fraction function
def fraction(holding_cost, unit_value):
    return 4 / (2 +holding_cost / unit_value + unit_value / holding_cost)

# Define holding costs and unit values
step = 0.1
holding_costs = np.arange(1, 20+step, step)
unit_values = np.arange(1, 100+step, step)

# Calculate heatmap values
heatmap_values = np.array([
    [fraction(h, u) for h in holding_costs] for u in unit_values
])

# Plotting the heatmap
plt.figure(figsize=(10, 8))
plt.imshow(heatmap_values, cmap="Spectral", aspect="auto", origin="lower")
cbar = plt.colorbar(shrink=0.8)

# Set custom ticks for colorbar
custom_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
cbar.set_ticks(custom_ticks)
cbar.set_ticklabels([f"{tick:.1f}" for tick in custom_ticks])
cbar.set_label(r"$\frac{\mathbb{E}[C^*|D,I]}{\mathbb{E}[C^{(1)}|D,I]}$")

# Adjust ticks to match holding costs and unit values
plt.xticks(ticks=np.arange(len(holding_costs), step = 10), labels = holding_costs[::10].astype(int))
plt.yticks(ticks=np.arange(len(unit_values), step = 22), labels = unit_values[::22].astype(int))

# Labels
plt.xlabel(r"$h_t$ (Holding Cost)")
plt.ylabel(r"$c_t$ (Unit Value)")
plt.tight_layout()

# Display and save the plot
plt.show()
plt.savefig('./docs/SequentialDecisionMaking/figures/analytical_per.pdf')
