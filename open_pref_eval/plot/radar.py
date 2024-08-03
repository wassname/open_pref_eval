
from matplotlib import pyplot as plt
from math import pi
import pandas as pd

def radar_plot(x: pd.Series, ax=None, color='b'):
    categories = x.index

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values = x.values.tolist()
    values += values[:1]

    if ax is None:
        ax = plt.subplot(1,1,1, polar=True)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)

    # Plot data
    ax.plot(angles, values, color=color, linewidth=1, linestyle='solid')
    
    # Fill area
    ax.fill(angles, values, color=color, alpha=0.1)
