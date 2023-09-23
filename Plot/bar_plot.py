import matplotlib.pyplot as plt


def plot_bar(methods, values, title='Comparison of Methods', ylabel='Value', save_path=None):
    """
    Plots a bar graph of methods based on provided values.

    Args:
    - methods (list): List of method names.
    - values (list): Corresponding values for each method.
    - title (str): Title for the plot.
    - ylabel (str): Y-axis label.

    Returns:
    - None
    """
    # Determine colors for bars
    colors = ['#d62728' if 'OURS' in method else '#1f77b4' for method in
              methods]  # Muted red for OURS, soft blue for others

    # Plot
    plt.figure(figsize=(14, 8))
    bars = plt.bar(methods, values, color=colors, edgecolor='black', alpha=0.7)

    # Aesthetics
    # plt.title(title, fontsize=20, fontweight='bold')
    plt.ylabel(ylabel, fontsize=16)
    # plt.xlabel('Methods', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()


# Data
methods = ["ERM", "MixUp", "CutMix", "AugMix", "RandAugment", "CutOut", "RSC", "MEADA", "ACVC", "PixMix", "L2D", "OURS(Ret_T_h)", "OURS(SD_I2I_h)", "OURS(SD_T2I_h)", "OURS(SD_ControlNet_h)", "OURS(InstructPix2Pix_instruct)"]
values = [11.32, 9.46, 12.1, 11.4, 11.34, 11.48, 10.1, 11.84, 10.08, 11.72, 9.0, 11.4, 5.9, 3.6, 4.2, 6.6]
