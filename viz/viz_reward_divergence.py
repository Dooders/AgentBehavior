import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_reward_divergence_plot(df, isolation_metrics):
    """
    Create a scatter plot showing agent rewards plotted against their average
    divergence scores.
    
    Args:
        df: DataFrame with agent data
        isolation_metrics: Dictionary of calculated isolation metrics
    """
    print("Generating reward vs divergence visualization...")
    
    # Calculate average divergence score across metrics
    # First, stack all metrics
    metrics = ['cosine_distance', 'euclidean_distance', 'mahalanobis_distance', 
               'mean_pairwise_distance', 'nearest_neighbor_distance']
    
    metrics_array = np.column_stack([isolation_metrics[m] for m in metrics])
    avg_divergence = np.mean(metrics_array, axis=1)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot for all agents
    plt.scatter(avg_divergence, df['reward'], 
                alpha=0.5, c='gray', s=50, edgecolor='black')
    
    # Define key agents to highlight
    key_agents = ['XWWDLtVr', 'nq7AEggt', 'SCSMTVA2', '6xScYvpu', 'mAN8Vx78']
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Highlight key agents
    for i, agent_id in enumerate(key_agents):
        agent_idx = df[df['agent_id'] == agent_id].index[0]
        plt.scatter(avg_divergence[agent_idx], df.loc[agent_idx, 'reward'],
                    color=colors[i], s=200, edgecolor='black', zorder=10, 
                    label=agent_id)
        
        # Add text labels
        plt.annotate(
            agent_id,
            (avg_divergence[agent_idx], df.loc[agent_idx, 'reward']),
            fontsize=12, ha='center', va='bottom',
            xytext=(0, 10), textcoords='offset points',
            weight='bold'
        )
    
    # Add best fit line to show overall trend
    sns.regplot(x=avg_divergence, y=df['reward'], 
                scatter=False, ci=None, color='black', 
                line_kws={'linestyle': '--', 'linewidth': 1.5})
    
    # Customize plot
    plt.title('Relationship Between Behavioral Divergence and Reward', 
              fontsize=18, pad=20)
    plt.xlabel('Average Divergence Score (z-score)', fontsize=14)
    plt.ylabel('Agent Reward', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(fontsize=12, title='Notable Agents', title_fontsize=14)
    
    # Add explanatory zones
    plt.axhspan(df['reward'].mean() + df['reward'].std(), df['reward'].max() + 10, 
                alpha=0.1, color='green',
                label='High Reward Zone')
    plt.axhspan(df['reward'].min() - 10, df['reward'].mean() - df['reward'].std(), 
                alpha=0.1, color='red',
                label='Low Reward Zone')
    
    plt.axvspan(2, avg_divergence.max() + 0.5, 
                alpha=0.1, color='blue',
                label='High Divergence Zone')
    
    # Add text annotations for zones
    plt.text(
        avg_divergence.max() - 0.5, 
        df['reward'].max() - 10,
        'Strategic Innovators',
        fontsize=12, ha='right', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round')
    )
    
    plt.text(
        avg_divergence.max() - 0.5, 
        df['reward'].min() + 10,
        'Maladaptive Divergents',
        fontsize=12, ha='right', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round')
    )
    
    plt.text(
        0, 
        df['reward'].mean(),
        'Typical Agents',
        fontsize=12, ha='left', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round')
    )
    
    # Add explanatory text
    plt.figtext(
        0.5, 0.01, 
        "Relationship between agent divergence (averaged across metrics) and reward performance.\n"
        "Note the wide distribution suggesting behavioral divergence can both enhance and diminish performance.",
        ha='center', fontsize=12, style='italic'
    )
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig('figures/reward_vs_divergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Reward vs divergence visualization saved to figures/reward_vs_divergence.png") 