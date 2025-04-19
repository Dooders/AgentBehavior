import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_zscore_comparison(df, isolation_metrics):
    """
    Create a bar chart showing the maximum z-scores for top divergent agents
    across different isolation metrics.
    
    Args:
        df: DataFrame with agent data
        isolation_metrics: Dictionary of calculated isolation metrics
    """
    print("Generating z-score comparison visualization...")
    
    # Define metrics to include
    metrics = [
        'cosine_distance',
        'euclidean_distance',
        'mahalanobis_distance',
        'mean_pairwise_distance',
        'nearest_neighbor_distance',
        'local_density'
    ]
    
    # Prettier metric names for display
    metric_names = {
        'cosine_distance': 'Cosine',
        'euclidean_distance': 'Euclidean',
        'mahalanobis_distance': 'Mahalanobis',
        'mean_pairwise_distance': 'Mean Pairwise',
        'nearest_neighbor_distance': 'Nearest Neighbor',
        'local_density': 'Local Density'
    }
    
    # Get key agents
    key_agents = ['XWWDLtVr', 'nq7AEggt', 'SCSMTVA2', '6xScYvpu', 'mAN8Vx78']
    
    # Create a dataframe to hold z-scores
    z_data = []
    
    for agent_id in key_agents:
        agent_idx = df[df['agent_id'] == agent_id].index[0]
        
        for metric in metrics:
            z_score = isolation_metrics[metric][agent_idx]
            z_data.append({
                'Agent': agent_id,
                'Metric': metric_names[metric],
                'Z-Score': z_score,
                'Above Threshold': z_score > 2.0
            })
    
    z_df = pd.DataFrame(z_data)
    
    # Count metrics where each agent is above threshold
    metric_counts = z_df[z_df['Above Threshold']].groupby('Agent').size()
    
    # Sort agents by the number of metrics they're divergent in
    agent_order = metric_counts.sort_values(ascending=False).index.tolist()
    
    # Set up the figure
    plt.figure(figsize=(16, 12))
    
    # Create the grouped bar chart with more spacing between groups
    ax = sns.barplot(
        data=z_df,
        x='Agent', y='Z-Score', hue='Metric',
        order=agent_order,
        palette='viridis',
        width=0.7
    )
    
    # Add a horizontal line at the threshold
    ax.axhline(y=2.0, color='r', linestyle='--', linewidth=2, label='Threshold (z=2.0)')
    
    # Customize the plot
    plt.title('Z-Scores by Agent and Isolation Metric', fontsize=18, pad=20)
    plt.xlabel('Agent ID', fontsize=14)
    plt.ylabel('Z-Score', fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0, fontsize=12)
    
    # Add counts to agent labels
    new_labels = [f"{agent}\n({metric_counts[agent]} metrics)" for agent in agent_order]
    ax.set_xticklabels(new_labels, fontsize=12)
    
    # Move legend outside the plot and make it more readable
    legend = plt.legend(
        bbox_to_anchor=(1.05, 1), 
        loc='upper left', 
        title='Isolation Metric',
        fontsize=12,
        title_fontsize=14,
        framealpha=0.9
    )
    
    # Add grid
    plt.grid(axis='y', alpha=0.3)
    
    # Add text annotations on top of the bars for highest values
    # with improved positioning to avoid overlap
    for agent in agent_order:
        # Find highest z-score for this agent
        agent_data = z_df[z_df['Agent'] == agent]
        max_metric = agent_data.loc[agent_data['Z-Score'].idxmax()]
        
        # Only annotate if above threshold
        if max_metric['Z-Score'] > 2.0:
            # Find the position of this bar
            agent_pos = agent_order.index(agent)
            
            # Add annotation with background for better visibility
            plt.annotate(
                f"{max_metric['Z-Score']:.2f}",
                xy=(agent_pos, max_metric['Z-Score']),
                xytext=(0, 10), textcoords='offset points',
                ha='center', va='bottom',
                fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.1')
            )
    
    # Add explanatory text
    plt.figtext(
        0.5, 0.01, 
        "Maximum z-scores for top divergent agents across different isolation metrics.\n"
        "Cosine distance consistently produces the strongest statistical signal.",
        ha='center', fontsize=12, style='italic',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
    )
    
    # Tight layout with more space for the legend
    plt.tight_layout(rect=[0, 0.03, 0.85, 0.97])
    
    # Save figure
    plt.savefig('figures/zscore_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Z-score comparison visualization saved to figures/zscore_comparison.png") 