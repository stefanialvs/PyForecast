import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import seaborn as sns


def plot_distributions(distributions_dict, fig_title=None, xlabel=None):
  n_distributions = len(distributions_dict.keys())
  fig, ax = plt.subplots(1, figsize=(7, 5.5))
  plt.subplots_adjust(wspace=0.35)
  
  n_colors = len(distributions_dict.keys())
  colors = sns.color_palette("hls", n_colors)
  
  for idx, dist_name in enumerate(distributions_dict.keys()):
      train_dist_plot = sns.kdeplot(distributions_dict[dist_name],
                                    bw='silverman',
                                    label=dist_name,
                                    color=colors[idx])
      if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=14)
      ax.set_ylabel('Density', fontsize=14)
      ax.set_title(fig_title, fontsize=15.5)
      ax.grid(True)
      ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
  
  fig.tight_layout()
  if fig_title is not None:
    fig_title = fig_title.replace(' ', '_')
    plot_file = "./results/{}_distributions.png".format(fig_title)
    plt.savefig(plot_file, bbox_inches = "tight", dpi=300)
  plt.show()
