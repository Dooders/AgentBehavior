import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

def create_isolation_metrics_visualization(df, isolation_metrics):
    """
    Create a visualization showing the distribution of each isolation metric
    with thresholds and anomalous agents highlighted.
    
    Args:
        df: DataFrame with agent data
        isolation_metrics: Dictionary of calculated isolation metrics
    """
    print("Generating isolation metrics comparison visualization...")
    
    # Set up the figure with a grid of subplots
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    # Define metrics to plot
    metrics = [
        'mahalanobis_distance', 
        'euclidean_distance', 
        'cosine_distance',
        'mean_pairwise_distance', 
        'nearest_neighbor_distance', 
        'local_density'
    ]
    
    # Prettier titles
    metric_titles = {
        'mahalanobis_distance': 'Mahalanobis Distance',
        'euclidean_distance': 'Euclidean Distance',
        'cosine_distance': 'Cosine Distance',
        'mean_pairwise_distance': 'Mean Pairwise Distance',
        'nearest_neighbor_distance': 'Nearest Neighbor Distance',
        'local_density': 'Local Density'
    }
    
    # Define key agents to highlight
    key_agents = ['XWWDLtVr', 'nq7AEggt', 'SCSMTVA2', '6xScYvpu']
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        # Create subplot
        ax = fig.add_subplot(gs[i//3, i%3])
        
        # Get z-scores for this metric
        z_scores = isolation_metrics[metric]
        
        # Plot distribution
        sns.histplot(z_scores, kde=True, bins=20, color='gray', alpha=0.7, ax=ax)
        
        # Add vertical line at threshold
        ax.axvline(x=2.0, color='r', linestyle='--', linewidth=2, label='Threshold (z=2.0)')
        
        # Spacing between labels to prevent overlap
        label_positions = {}
        
        # First pass: collect agents above threshold and their z-scores
        agents_above = []
        for agent_id in key_agents:
            agent_idx = df[df['agent_id'] == agent_id].index[0]
            agent_z = z_scores[agent_idx]
            if agent_z > 2.0:
                agents_above.append((agent_id, agent_z))
        
        # Sort by z-score to position labels better
        agents_above.sort(key=lambda x: x[1])
        
        # Calculate positions for labels to prevent overlap
        vertical_positions = []
        step = 0.2  # Vertical spacing between labels
        max_height = ax.get_ylim()[1]
        
        # Start from 10% of max height and increment
        current_height = max_height * 0.1
        for _ in agents_above:
            vertical_positions.append(current_height)
            current_height += max_height * step
        
        # Highlight key agents
        for idx, (agent_id, agent_z) in enumerate(agents_above):
            agent_idx = df[df['agent_id'] == agent_id].index[0]
            
            # Different color for each agent
            if agent_id == 'XWWDLtVr':
                color = 'red'
            elif agent_id == 'nq7AEggt':
                color = 'blue'
            elif agent_id == 'SCSMTVA2':
                color = 'green'
            else:
                color = 'orange'
            
            # Add vertical line at agent's z-score
            ax.axvline(x=agent_z, color=color, linewidth=1.5, alpha=0.7)
            
            # Add annotation with neat background
            ax.annotate(
                f"{agent_id}: z={agent_z:.2f}",
                xy=(agent_z, 0), 
                xytext=(agent_z, vertical_positions[idx]),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color=color, alpha=0.7),
                color=color, ha='center', fontsize=8, va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2', edgecolor=color)
            )
        
        # Customize plot
        ax.set_title(metric_titles[metric], fontsize=14)
        ax.set_xlabel('Z-Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Adjust x-axis to show divergent region clearly
        ax.set_xlim([-3, max(6, np.max(z_scores) + 0.5)])
    
    # Add overall title
    fig.suptitle('Isolation Metric Distributions with Anomalous Agents', fontsize=18, y=0.98)
    
    # Add explanatory text
    plt.figtext(
        0.5, 0.01, 
        "Distribution of agent z-scores across six isolation metrics. Vertical red lines indicate the statistical divergence threshold (z > 2.0).\n"
        "Colored lines show the positions of key anomalous agents in each distribution.",
        ha='center', fontsize=12, style='italic',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
    )
    
    # Tight layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    plt.savefig('figures/isolation_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Isolation metrics visualization saved to figures/isolation_metrics.png") 