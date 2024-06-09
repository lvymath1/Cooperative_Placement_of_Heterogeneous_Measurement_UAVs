from matplotlib import pyplot as plt


def plot_time(time_ls, x_labels, random_seed):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    fig.set_size_inches(10, 6)
    plt.rcParams['font.family'] = 'serif'
    # Define grayscale colors and traditional line styles
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    line_styles = ['-', '--', '-.', ':']
    marker_styles = ['o', 's', '^', 'D']

    # Plot the lines using the axes object
    for i, line_data in enumerate(time_ls):
        if i == 0:
            axes.plot(x_labels, line_data, label="RGA", linewidth=4, linestyle=line_styles[i], marker=marker_styles[i],
                      markersize=8, markerfacecolor='none', markeredgewidth=1.5, color=colors[i], zorder=len(time_ls))
        else:
            axes.plot(x_labels, line_data, label=["JGA", "DA", "RA"][i - 1], linewidth=4, linestyle=line_styles[i],
                      marker=marker_styles[i], markersize=8, markerfacecolor='none', markeredgewidth=1.5, color=colors[i],
                      zorder=i)

    # Set the font size for axis labels
    axes.set_xlabel("Potential locations for low-precision UAVs", fontsize=24)
    axes.set_ylabel("Time(s)", fontsize=24)

    axes.set_xlim(x_labels[0], x_labels[-1])

    # Set the font size and position for the legend
    axes.legend(fontsize=20)

    # Set the font size for axis tick labels
    axes.tick_params(axis='x', labelsize=20)
    axes.tick_params(axis='y', labelsize=20)

    # Set the thickness of the axes
    ax = axes
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set the grid lines
    axes.grid(True, linestyle='--', linewidth=0.5)

    # Set y-axis to logarithmic scale
    axes.set_yscale('log')

    # Show the plot
    plt.tight_layout()

    filename = f'fig/output_time_random_seed={random_seed}.png'
    # Save the plot
    plt.savefig(filename, dpi=400)