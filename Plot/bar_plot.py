import matplotlib.pyplot as plt

# Data
methods = ["ERM", "MixUp", "CutMix", "AugMix", "RandAugment", "CutOut", "RSC", "MEADA", "ACVC", "PixMix", "L2D", "OURS(Ret_T_h)", "OURS(SD_I2I_h)", "OURS(SD_T2I_h)", "OURS(SD_ControlNet_h)", "OURS(InstructPix2Pix_instruct)"]
values = [11.32, 9.46, 12.1, 11.4, 11.34, 11.48, 10.1, 11.84, 10.08, 11.72, 9.0, 11.4, 5.9, 3.6, 4.2, 6.6]

# Plot
plt.figure(figsize=(14, 8))
plt.plot(methods, values, marker='*', linestyle='--', color='dodgerblue', alpha=0.5, markersize=10, linewidth=1.5)
plt.scatter(methods, values, color='dodgerblue', s=100, edgecolors='black', linewidth=0.5)  # Emphasize the star markers

# Aesthetics
plt.title('Comparison of Methods based on Last Column', fontsize=16)
plt.ylabel('Value', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.show()
