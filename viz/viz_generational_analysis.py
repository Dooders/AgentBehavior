import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_generational_analysis(df, isolation_metrics):
    """
    Create a timeline visualization showing when divergent agents appeared
    across generations, with annotations for key evolutionary events.
    
    Args:
        df: DataFrame with agent data
        isolation_metrics: Dictionary of calculated isolation metrics
    """
    print("Generating generational analysis visualization...")
    
    # Calculate average divergence z-score across metrics
    metrics = ['cosine_distance', 'euclidean_distance', 'mahalanobis_distance', 
               'mean_pairwise_distance', 'nearest_neighbor_distance']
    
    # Stack all metrics
    metrics_array = np.column_stack([isolation_metrics[m] for m in metrics])
    avg_divergence = np.mean(metrics_array, axis=1)
    
    # Add to dataframe
    df_plot = df.copy()
    df_plot['avg_divergence'] = avg_divergence
    
    # Define divergent agents (z-score > 2.0)
    df_plot['is_divergent'] = df_plot['avg_divergence'] > 2.0
    
    # Get only divergent agents
    divergent_df = df_plot[df_plot['is_divergent']]
    
    # Create figure
    plt.figure(figsize=(16, 12))
    
    # Create a scatter plot of generations vs reward, colored by divergence
    scatter = plt.scatter(
        df_plot['generation'], 
        df_plot['reward'],
        c=df_plot['avg_divergence'], 
        cmap='viridis',
        s=100, alpha=0.7, edgecolor='black'
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Average Divergence (z-score)', fontsize=12)
    
    # Custom offsets for agent labels to prevent overlap
    offsets = {
        'XWWDLtVr': (0.2, -15),
        'nq7AEggt': (-0.2, 15),
        'SCSMTVA2': (0.2, 15),
        '6xScYvpu': (-0.2, -15),
        'mAN8Vx78': (0.2, 10)
    }
    
    # Highlight key divergent agents
    key_agents = ['XWWDLtVr', 'nq7AEggt', 'SCSMTVA2', '6xScYvpu', 'mAN8Vx78']
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, agent_id in enumerate(key_agents):
        if agent_id in df_plot['agent_id'].values:
            agent_data = df_plot[df_plot['agent_id'] == agent_id]
            
            plt.scatter(
                agent_data['generation'], 
                agent_data['reward'],
                s=300, color=colors[i], edgecolor='black', linewidth=2,
                zorder=10, label=agent_id
            )
            
            # Add text label with background and custom offset
            offset_x, offset_y = offsets.get(agent_id, (0, 10))
            gen = agent_data['generation'].values[0]
            reward = agent_data['reward'].values[0]
            
            plt.annotate(
                agent_id,
                (gen + offset_x, reward + offset_y),
                fontsize=12, ha='center', va='center',
                weight='bold',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor=colors[i], boxstyle='round,pad=0.2')
            )
    
    # Add generation distribution (how many agents per generation)
    gen_counts = df_plot.groupby('generation').size()
    ax2 = plt.twinx()
    ax2.bar(gen_counts.index, gen_counts.values, alpha=0.15, color='gray', width=0.8)
    ax2.set_ylabel('Number of Agents', fontsize=12, color='gray')
    ax2.tick_params(axis='y', colors='gray')
    
    # Add annotations for hypothetical evolutionary events with better spacing
    events = [
        {'gen': 0, 'text': 'Initial Population', 'y': 140, 'ha': 'center'},
        {'gen': 2, 'text': 'Resource Scarcity\nIntroduced', 'y': 140, 'ha': 'center'},
        {'gen': 5, 'text': 'XWWDLtVr Divergence\nEmerges', 'y': 35, 'ha': 'center'},
        {'gen': 6, 'text': 'SCSMTVA2 High-Reward\nStrategy Appears', 'y': 140, 'ha': 'center'},
        {'gen': 9, 'text': 'Final Generation', 'y': 90, 'ha': 'right'}
    ]
    
    for event in events:
        # Add vertical line
        plt.axvline(x=event['gen'], color='gray', linestyle='--', alpha=0.5)
        
        # Add text annotation with background
        plt.annotate(
            event['text'],
            (event['gen'], event['y']),
            fontsize=10, ha=event['ha'],
            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.3'),
            xytext=(0, 0), textcoords='offset points'
        )
    
    # Add divergent agent count per generation with better positioning
    div_counts = divergent_df.groupby('generation').size()
    
    # Add text showing divergent count for generations with divergent agents
    # Position at the bottom of the plot with space between labels
    for gen, count in div_counts.items():
        plt.text(
            gen, -10, 
            f"{count} divergent\nagents",
            ha='center', fontsize=10, 
            bbox=dict(facecolor='lightyellow', alpha=0.9, edgecolor='orange', 
                     boxstyle='round,pad=0.5')
        )
    
    # Customize plot
    plt.title('Emergence of Divergent Agents Across Generations', fontsize=20, pad=20)
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Set axis limits to provide more space
    plt.ylim(-20, 150)
    
    # Set x-axis to show all generations
    plt.xticks(np.arange(0, 10), fontsize=12)
    
    # Add legend with better formatting
    legend = plt.legend(fontsize=12, title='Key Divergent Agents', 
                       title_fontsize=14, loc='upper left',
                       framealpha=0.9)
    
    # Add explanatory text
    plt.figtext(
        0.5, 0.01, 
        "Timeline showing when divergent agents appeared across evolutionary generations.\n"
        "Note the clustering of anomalies after specific environmental changes or selection pressure adjustments.",
        ha='center', fontsize=12, style='italic',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
    )
    
    # Tight layout with extra space for labels
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig('figures/generational_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generational analysis visualization saved to figures/generational_analysis.png") 