import matplotlib.pyplot as plt
import numpy as np

# Extracted data from the provided text
methods = [
    "OURS(Ret_T_m)", "OURS(Ret_T_m_filtered)",
    "OURS(SD_I2I_m)", "OURS(SD_I2I_m_filtered)",
    "OURS(SD_I2I_h)", "OURS(SD_I2I_h_filtered)",
    "OURS(SD_I2I_lec)", "OURS(SD_I2I_lec_filtered)",
    "OURS(SD_I2I_lem)", "OURS(SD_I2I_lem_filtered)",
    "OURS(SD_T2I_lem)", "OURS(SD_T2I_lem_filtered)",
    "OURS(SD_T2I_m)", "OURS(SD_T2I_m_filtered)",
    "OURS(SD_ControlNet_m)", "OURS(SD_ControlNet_m_filtered)",
    "OURS(SD_T2I_TInv_m)", "OURS(SD_T2I_TInv_m_filtered)"
]
values = [
    80.83, 80.14, 73.96, 74.27, 74.75, 75.05, 71.87, 70.78,
    72.69, 70.83, 79.56, 78.88, 82.26, 81.06, 72.32, 70.91,
    74.7, 74.68
]

# Grouped Bar Plot
bar_width = 0.35
index = np.arange(len(methods) // 2)

fig, ax = plt.subplots(figsize=(14, 8))
bar1 = ax.bar(index, values[::2], bar_width, label='Original', color='b', edgecolor='black')
bar2 = ax.bar(index + bar_width, values[1::2], bar_width, label='Filtered', color='r', edgecolor='black')

# Aesthetics
ax.set_xlabel('Methods', fontsize=16)
ax.set_ylabel('Values', fontsize=16)
ax.set_title('Comparison between Original and Filtered Methods', fontsize=18)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(methods[::2], rotation=45, ha='right', fontsize=12)
ax.legend()

plt.tight_layout()
plt.show()
