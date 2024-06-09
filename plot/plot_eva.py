import matplotlib.pyplot as plt

def plot_eva(results, x_labels, random_seed, matrix_type_names):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 6))

    # Adjust the spacing between subplots and the margins of the figure
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.10, wspace=0.3)

    for matrix_index, matrix_name in enumerate(matrix_type_names):
        ax = axes[matrix_index]
        res = results[matrix_index]

        # Define colors, line styles, and marker styles
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        line_styles = ['-', '--', '-.', ':']
        marker_styles = ['o', 's', '^', 'D']

        # Plot the lines using the axes object
        for i, line_data in enumerate(res):
            if i == 0:
                ax.plot(x_labels, line_data, label="RGA", linewidth=3, linestyle=line_styles[i],
                        marker=marker_styles[i], markersize=8, markeredgecolor=colors[i], markeredgewidth=1.5,
                        markerfacecolor='none', color=colors[i], zorder=len(res))
            else:
                ax.plot(x_labels, line_data, label=["JGA", "DA", "RA"][i - 1], linewidth=3, linestyle=line_styles[i],
                        marker=marker_styles[i], markersize=8, markeredgecolor=colors[i], markeredgewidth=1.5,
                        markerfacecolor='none', color=colors[i], zorder=i)

        # Set the font size for axis labels
        ax.set_xlabel("Potential locations for low-precision UAVs", fontsize=14)
        ax.set_ylabel("Normalized MSE(dB)", fontsize=14)

        ax.set_xlim(x_labels[0], x_labels[-1])

        # Set the font size and position for the legend
        ax.legend(fontsize=12)

        # Set the font size for axis tick labels
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        # Set the thickness of the axes
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set the grid lines
        ax.grid(True, linestyle='--', linewidth=0.5)

        # Set the title
        ax.set_title(matrix_name, fontsize=16)

    filename = f'fig/output_eva_random_seed={random_seed}.png'
    plt.savefig(filename, dpi=400)
