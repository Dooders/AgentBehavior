import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

def create_population_visualization(df, features):
    """
    Create a scatter plot showing the full agent population with divergent agents highlighted.
    
    Args:
        df: DataFrame with agent data
        features: Feature matrix used for dimensionality reduction
    """
    print("Generating agent population visualization...")
    
    # Apply t-SNE to reduce to 2D for visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    # Identify key agents
    key_agents = ['XWWDLtVr', 'nq7AEggt', 'SCSMTVA2', '6xScYvpu', 'mAN8Vx78']
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Background agents (gray)
    background_mask = ~df['agent_id'].isin(key_agents)
    plt.scatter(
        features_2d[background_mask, 0], 
        features_2d[background_mask, 1],
        color='gray', alpha=0.3, s=50, label='Population'
    )
    
    # Highlight divergent agents with distinct colors
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Custom offsets for specific agents to prevent overlap
    offsets = {
        'XWWDLtVr': (0, 20),
        'nq7AEggt': (0, -25),
        'SCSMTVA2': (15, 15),
        '6xScYvpu': (-20, 0),
        'mAN8Vx78': (20, -10)
    }
    
    for i, agent_id in enumerate(key_agents):
        agent_idx = df[df['agent_id'] == agent_id].index
        if len(agent_idx) > 0:
            plt.scatter(
                features_2d[agent_idx, 0], 
                features_2d[agent_idx, 1],
                color=colors[i], s=150, label=agent_id, edgecolor='black'
            )
            
            # Add text label with background box for better readability
            plt.annotate(
                agent_id,
                (features_2d[agent_idx, 0][0], features_2d[agent_idx, 1][0]),
                fontsize=12, ha='center', va='center',
                xytext=offsets[agent_id], textcoords='offset points',
                weight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=colors[i], boxstyle='round,pad=0.2')
            )
    
    # Customize plot
    plt.title('Agent Population Visualization', fontsize=18, pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add legend with larger markers
    plt.legend(fontsize=12, markerscale=1.5, loc='upper right')
    
    # Add explanatory text
    plt.figtext(
        0.5, 0.01, 
        "t-SNE projection of agent behavioral features. Divergent agents are highlighted and labeled.",
        ha='center', fontsize=12, style='italic',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
    )
    
    # Add ellipse around XWWDLtVr to emphasize its isolation
    agent_idx = df[df['agent_id'] == 'XWWDLtVr'].index[0]
    x, y = features_2d[agent_idx, 0], features_2d[agent_idx, 1]
    from matplotlib.patches import Ellipse
    ellipse = Ellipse((x, y), width=5, height=5, fill=False, edgecolor='red', 
                      linestyle='--', linewidth=2, alpha=0.7)
    plt.gca().add_patch(ellipse)
    
    # Tight layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('figures/agent_population.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Agent population visualization saved to figures/agent_population.png") 