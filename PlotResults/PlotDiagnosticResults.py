import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Sequence
import numpy as np


def CohensD(cohens_d, df: None) -> list:
    """
    Plots the output of Cohens D as a bar plot with a histogram of the values on different axes at the same scale.

    Args:
        result (pd.DataFrame): DataFrame containing the results of Cohens D.
        doall_df (pd.DataFrame, optional): DataFrame containing additional data for plotting. Defaults to None.

    Returns:
        None: Displays the plot.
    """
    import matplotlib.pyplot as plt
    from collections.abc import Sequence

    if df is not None:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("doall_df must be a pandas DataFrame")
        if 'CohensD' not in df.columns:
            raise ValueError("doall_df must contain a 'CohensD' column")
        
        import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    # Dummy data (replace with your actual data)
    np.random.seed(0)
    cohens_d = np.random.normal(0, 0.5, 100)

    # Set up figure and gridspec
    fig = plt.figure(figsize=(12, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 5], wspace=0.05)

    # Histogram on the left
    ax1 = fig.add_subplot(gs[0])
    ax1.hist(cohens_d, bins=20, orientation='horizontal', color=[0.8, 0.2, 0.2])
    ax1.set_xlabel("Proportion")
    ax1.invert_xaxis()
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")

    # Bar plot on the right
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    indices = np.arange(len(cohens_d))
    bars = ax2.bar(indices, cohens_d, color=[0.2, 0.4, 0.6])
    ax2.plot(indices, cohens_d, 'r.')

    # Significance lines
    effect_sizes = [
        (0.2, 'Small effect size', 'g'),
        (0.5, 'Medium effect size', 'b'),
        (0.8, 'Large effect size', 'r'),
        (2.0, 'Huge effect size', 'm')
    ]

    for val, label, color in effect_sizes:
        ax2.axhline(y=val, linestyle='--', color=color, label=label)
        ax2.axhline(y=-val, linestyle='--', color=color)

    # Labels and grid
    ax2.set_xlabel("IDP index")
    ax2.set_ylabel("Cohen's d: $(\\mu_1 - \\mu_2)/\\sigma_{pooled}$")
    ax2.set_title("Effect Size (Cohen's d) for T2 Batch Effect Across Structural IDPs")
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

def plot_result(result, doall_df=None):
   """
   
   Takes an argument of string which matches a column in the result DataFrame and returns a plot of the results.

    Args:
         result (str): The name of the column to plot.
         Dataframe (pd.DataFrame, optional): DataFrame containing additional data for plotting. Defaults to None.
    Returns:
            None: Displays the plot.
    
            
    string options:
        'CohensD': Plots Cohen's D values.
        'BatchPCcorr': Plots batch-PC correlation.
        'ExplainedVariance': Plots explained variance by batch
        'mahanalobis': Plots Mahalanobis distance.
        
    """
   