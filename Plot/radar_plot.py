import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

def plot_radar_base(data_dict):
    # Extract labels from the data dictionary
    labels = list(data_dict[list(data_dict.keys())[0]].keys())

    # Number of variables
    num_vars = len(labels)

    # Split the circle into even parts and save the angles
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    # Normalize the data
    mins = {label: min([algo_data[label] for algo_data in data_dict.values()]) for label in labels}
    maxs = {label: max([algo_data[label] for algo_data in data_dict.values()]) for label in labels}
    normalized_data = {}
    for algo, values in data_dict.items():
        normalized_data[algo] = {label: (value - mins[label]) / (maxs[label] - mins[label]) for label, value in values.items()}

    # Set figure and axis
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Generate distinct colors using colormap
    colors = plt.cm.jet(np.linspace(0, 1, len(data_dict)))

    # Plot each algorithm's normalized data
    for algo, values, color in zip(normalized_data.keys(), normalized_data.values(), colors):
        values_list = list(values.values())
        values_list += values_list[:1]  # Complete the loop

        # Check for dimension mismatch
        if len(angles) != len(values_list):
            raise ValueError(
                f'Dimension mismatch for {algo}. Expected {len(angles)} values but got {len(values_list)}.')

        ax.plot(angles, values_list, marker='o', label=algo, color=color)

    # Fix axis to go in the right order and start at 12 o'clock.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label.
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Set the y-axis labels to original accuracy
    y_ticks = np.linspace(0, 1, 6)  # Example: 6 ticks
    y_labels = [f"{mins[label] + tick * (maxs[label] - mins[label]):.2f}" for tick in y_ticks for label in labels[:1]]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Add legend
    ax.legend(loc='best')

    plt.title('Normalized Algorithms Accuracy with Original Scale')
    plt.savefig("sample.png")
    plt.show()


def plot_radar_sbs(data_dict1, data_dict2, save_path="sample.png"):
    # Extract labels from the data dictionaries and exclude 'Average'
    labels = [key for key in data_dict1[list(data_dict1.keys())[0]].keys() if key != 'Average']

    # Determine the maximum legend entry length for padding
    max_entry_length1 = max([len(algo) + len(f" - {data_dict1[algo]['Average']:.2f}%") for algo in data_dict1.keys()])
    max_entry_length2 = max([len(algo) + len(f" - {data_dict2[algo]['Average']:.2f}%") for algo in data_dict2.keys()])
    max_entry_length = max(max_entry_length1, max_entry_length2)

    # Number of variables
    num_vars = len(labels)

    # Split the circle into even parts and save the angles
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    # Normalize the data excluding 'Average'
    mins = {label: min([algo_data[label] for algo_data in data_dict1.values() if label in algo_data]) for label in
            labels}
    maxs = {label: max([algo_data[label] for algo_data in data_dict1.values() if label in algo_data]) for label in
            labels}

    normalized_data1 = {}
    normalized_data2 = {}
    for algo, values in data_dict1.items():
        normalized_data1[algo] = {label: (values[label] - mins[label]) / (maxs[label] - mins[label]) for label in labels
                                  if label in values}
    for algo, values in data_dict2.items():
        normalized_data2[algo] = {label: (values[label] - mins[label]) / (maxs[label] - mins[label]) for label in labels
                                  if label in values}

    # Create a GridSpec object
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.6)  # Adjust width_ratios and wspace as needed

    # Create the figure and axes using the GridSpec object
    fig = plt.figure(figsize=(18, 8),constrained_layout=True)
    ax1 = plt.subplot(gs[0], polar=True)
    ax2 = plt.subplot(gs[1], polar=True)
    # Generate distinct colors using colormap
    colors = plt.cm.jet(np.linspace(0, 1, max(len(data_dict1), len(data_dict2))))

    # Plot each algorithm's normalized data with padding for legend labels
    for ax, data, data_dict in zip([ax1, ax2], [normalized_data1, normalized_data2], [data_dict1, data_dict2]):
        for algo, values, color in zip(data.keys(), data.values(), colors):
            values_list = [values[label] for label in labels]
            values_list += values_list[:1]  # Complete the loop

            # Calculate padding spaces for the legend entry
            entry_length = len(algo) + len(f" - {data_dict[algo]['Average']:.2f}%")
            padding_spaces = ' ' * (max_entry_length - entry_length)

            # Highlight OURS methods with solid lines and others with dashed lines
            if "OURS" in algo:
                ax.plot(angles, values_list, marker='^', markersize=10,
                        label=f"{algo}{padding_spaces} - {data_dict[algo]['Average']:.2f}%", color=color, linewidth=2,
                        alpha=0.4)
            else:
                ax.plot(angles, values_list, marker='o',
                        label=f"{algo}{padding_spaces} - {data_dict[algo]['Average']:.2f}%", color=color,
                        linestyle='--', alpha=0.4)

            # Fix axis to go in the right order and start at 12 o'clock.
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)

            # Draw axis lines for each angle and label.
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)

            # Set the y-axis labels
            y_ticks = np.linspace(0, 1, 6)  # Example: 6 ticks
            y_labels = [f"{mins[label] + tick * (maxs[label] - mins[label]):.2f}" for tick in y_ticks for label in
                        labels[:1]]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels)

    # Create custom legends in the middle with titles and adjusted positioning
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Place the legends in the center between the two radar plots with titles
    legend1 = fig.legend(handles1, labels1, loc='upper center', bbox_to_anchor=(0.5, 0.95), title="ResNet18")
    legend2 = fig.legend(handles2, labels2, loc='lower center', bbox_to_anchor=(0.5, 0.05), title="ResNet50")
    for text in legend1.get_texts():
        text.set_fontfamily('monospace')
    for text in legend2.get_texts():
        text.set_fontfamily('monospace')

    # Set title
    # plt.suptitle('Algorithms Performance', fontsize=16)
    # plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_radar(data_dict, save_path="sample.png"):
    # Extract labels from the data dictionary and exclude 'Average'
    labels = [key for key in data_dict[list(data_dict.keys())[0]].keys() if key != 'Average']

    # Determine the maximum legend entry length for padding
    max_entry_length = max([len(algo) + len(f" - {data_dict[algo]['Average']:.2f}%") for algo in data_dict.keys()])

    # Number of variables
    num_vars = len(labels)

    # Split the circle into even parts and save the angles
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    # Normalize the data excluding 'Average'
    mins = {label: min([algo_data[label] for algo_data in data_dict.values() if label in algo_data]) for label in labels}
    maxs = {label: max([algo_data[label] for algo_data in data_dict.values() if label in algo_data]) for label in labels}
    normalized_data = {}
    for algo, values in data_dict.items():
        normalized_data[algo] = {label: (values[label] - mins[label]) / (maxs[label] - mins[label]) for label in labels if label in values}

    # Set figure and axis
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Generate distinct colors using colormap
    colors = plt.cm.jet(np.linspace(0, 1, len(data_dict)))

    # Plot each algorithm's normalized data
    for algo, values, color in zip(normalized_data.keys(), normalized_data.values(), colors):
        values_list = [values[label] for label in labels]
        values_list += values_list[:1]  # Complete the loop

        # Calculate padding spaces for the legend entry
        entry_length = len(algo) + len(f" - {data_dict[algo]['Average']:.2f}%")
        padding_spaces = ' ' * (max_entry_length - entry_length)

        # Highlight OURS methods with solid lines and others with dashed lines
        if "OURS" in algo:
            ax.plot(angles, values_list, marker='o',
                    label=f"{algo}{padding_spaces} - {data_dict[algo]['Average']:.2f}%", color=color, linewidth=2, alpha=0.8)
        else:
            ax.plot(angles, values_list, marker='o',
                    label=f"{algo}{padding_spaces} - {data_dict[algo]['Average']:.2f}%", color=color, linestyle='--', alpha=0.6)

    # Add legend outside the plot with monospace font
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    for text in legend.get_texts():
        text.set_fontfamily('monospace')

    # Fix axis to go in the right order and start at 12 o'clock.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label.
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Set the y-axis labels
    y_ticks = np.linspace(0, 1, 6)  # Example: 6 ticks
    y_labels = [f"{mins[label] + tick * (maxs[label] - mins[label]):.2f}" for tick in y_ticks for label in labels[:1]]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Add legend outside the plot
    # ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.title('Algorithms Performance')
    plt.tight_layout()  # Adjust layout to ensure everything fits
    plt.savefig(save_path)
    plt.show()


if __name__=='__main__':
    # Sample data for testing
    data_sample = {
        'ERM': {'autumn': 57.07, 'dim': 60.95, 'grass': 62.4, 'outdoor': 61.82, 'rock': 58.52, 'water': 65.04},
        'AugMix': {'autumn': 56.19, 'dim': 59.18, 'grass': 61.29, 'outdoor': 60.72, 'rock': 58.1, 'water': 63.16}
    }




    plot_radar(data_sample)
