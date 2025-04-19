import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

def create_divergence_types_viz(df, features):
    """
    Create a 2D projection of agent behaviors using t-SNE,
    with colored regions indicating different types of behavioral divergence.
    
    Args:
        df: DataFrame with agent data
        features: Feature matrix for dimensional reduction
    """
    print("Generating divergence types visualization...")
    
    # Apply t-SNE to reduce to 2D for visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=20, n_iter=2000)
    features_2d = tsne.fit_transform(features)
    
    # Define the three divergence types
    strategic_innovators = ['SCSMTVA2'] 
    fundamental_divergents = ['XWWDLtVr']
    systemic_outliers = ['nq7AEggt']
    
    # Additional agents with high divergence
    other_divergents = ['6xScYvpu', 'mAN8Vx78', '7mkdNKSM', 'wDRrgAYS', 'X3DvCEoN', 'NzTqmDqU']
    
    # Create figure
    plt.figure(figsize=(14, 12))
    
    # Create a scatter plot for all agents
    plt.scatter(
        features_2d[:, 0], features_2d[:, 1],
        c=df['reward'], cmap='viridis', 
        alpha=0.6, s=100, edgecolor='black'
    )
    
    # Add colorbar for reward
    cbar = plt.colorbar()
    cbar.set_label('Reward', fontsize=12)
    
    # Custom positioning for agent labels to prevent overlap
    # Format: agent_id: (x_offset, y_offset)
    label_offsets = {
        'SCSMTVA2': (0, 25),
        'XWWDLtVr': (30, 0),
        'nq7AEggt': (-30, -25),
        '6xScYvpu': (25, 15),
        'mAN8Vx78': (-25, 15),
        '7mkdNKSM': (25, -15),
        'wDRrgAYS': (-25, -15),
        'X3DvCEoN': (0, 20),
        'NzTqmDqU': (0, -20)
    }
    
    # Highlight the three divergence types with different markers
    # Strategic Innovators
    for agent_id in strategic_innovators:
        agent_idx = df[df['agent_id'] == agent_id].index[0]
        plt.scatter(
            features_2d[agent_idx, 0], features_2d[agent_idx, 1],
            marker='*', s=500, color='lime', edgecolor='black', linewidth=2,
            label=f'Strategic Innovator: {agent_id}'
        )
        
        # Add text label with custom offset
        offset = label_offsets.get(agent_id, (0, 20))
        plt.annotate(
            agent_id,
            (features_2d[agent_idx, 0], features_2d[agent_idx, 1]),
            fontsize=12, ha='center', va='center',
            xytext=offset, textcoords='offset points',
            weight='bold',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='lime', boxstyle='round,pad=0.2')
        )
        
        # Add circle to highlight the agent's region
        circle = plt.Circle(
            (features_2d[agent_idx, 0], features_2d[agent_idx, 1]), 
            radius=3, color='lime', alpha=0.2
        )
        plt.gca().add_patch(circle)
    
    # Fundamental Divergents
    for agent_id in fundamental_divergents:
        agent_idx = df[df['agent_id'] == agent_id].index[0]
        plt.scatter(
            features_2d[agent_idx, 0], features_2d[agent_idx, 1],
            marker='D', s=400, color='red', edgecolor='black', linewidth=2,
            label=f'Fundamental Divergent: {agent_id}'
        )
        
        # Add text label with custom offset
        offset = label_offsets.get(agent_id, (0, 20))
        plt.annotate(
            agent_id,
            (features_2d[agent_idx, 0], features_2d[agent_idx, 1]),
            fontsize=12, ha='center', va='center',
            xytext=offset, textcoords='offset points',
            weight='bold',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', boxstyle='round,pad=0.2')
        )
        
        # Add circle to highlight the agent's region
        circle = plt.Circle(
            (features_2d[agent_idx, 0], features_2d[agent_idx, 1]), 
            radius=3, color='red', alpha=0.2
        )
        plt.gca().add_patch(circle)
    
    # Systemic Outliers
    for agent_id in systemic_outliers:
        agent_idx = df[df['agent_id'] == agent_id].index[0]
        plt.scatter(
            features_2d[agent_idx, 0], features_2d[agent_idx, 1],
            marker='^', s=400, color='blue', edgecolor='black', linewidth=2,
            label=f'Systemic Outlier: {agent_id}'
        )
        
        # Add text label with custom offset
        offset = label_offsets.get(agent_id, (0, 20))
        plt.annotate(
            agent_id,
            (features_2d[agent_idx, 0], features_2d[agent_idx, 1]),
            fontsize=12, ha='center', va='center',
            xytext=offset, textcoords='offset points',
            weight='bold',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='blue', boxstyle='round,pad=0.2')
        )
        
        # Add circle to highlight the agent's region
        circle = plt.Circle(
            (features_2d[agent_idx, 0], features_2d[agent_idx, 1]), 
            radius=3, color='blue', alpha=0.2
        )
        plt.gca().add_patch(circle)
    
    # Also highlight other divergent agents with smaller markers
    for agent_id in other_divergents:
        agent_idx = df[df['agent_id'] == agent_id].index[0]
        plt.scatter(
            features_2d[agent_idx, 0], features_2d[agent_idx, 1],
            marker='o', s=200, color='orange', edgecolor='black', linewidth=1
        )
        
        # Add text label with custom offset and background
        offset = label_offsets.get(agent_id, (0, 15))
        plt.annotate(
            agent_id,
            (features_2d[agent_idx, 0], features_2d[agent_idx, 1]),
            fontsize=10, ha='center', va='center',
            xytext=offset, textcoords='offset points',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='orange', boxstyle='round,pad=0.1')
        )
    
    # Find a good position for the 'Other Divergent Agents' label
    # that doesn't overlap with other elements
    first_other_agent_idx = df[df['agent_id'] == other_divergents[0]].index[0]
    label_x = features_2d[first_other_agent_idx, 0] + 5
    label_y = features_2d[first_other_agent_idx, 1] - 5
    
    # Add a text label for the group of other divergents with clear background
    plt.text(
        label_x, label_y,
        'Other Divergent Agents',
        fontsize=12, ha='center', color='black',
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='orange', 
                 boxstyle='round,pad=0.3')
    )
    
    # Add legend for the three main types with better positioning
    legend_elements = [
        mpatches.Patch(facecolor='lime', alpha=0.5, edgecolor='black',
                      label='Strategic Innovators'),
        mpatches.Patch(facecolor='red', alpha=0.5, edgecolor='black',
                      label='Fundamental Divergents'),
        mpatches.Patch(facecolor='blue', alpha=0.5, edgecolor='black',
                      label='Systemic Outliers'),
        mpatches.Patch(facecolor='orange', alpha=0.5, edgecolor='black',
                      label='Other Divergent Agents')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12, 
               framealpha=0.9, title='Divergence Types', title_fontsize=14)
    
    # Customize plot
    plt.title('Agent Behavioral Space: Types of Divergence', fontsize=18, pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add explanatory text
    plt.figtext(
        0.5, 0.01, 
        "Dimensional reduction visualization of agent behavioral space showing clustering of different divergence types.\n"
        "Note the clear separation between different categories of behaviorally anomalous agents.",
        ha='center', fontsize=12, style='italic',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
    )
    
    # Save figure with extra padding
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig('figures/divergence_types.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Divergence types visualization saved to figures/divergence_types.png") 