import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with the specified number of variables.
    
    Based on matplotlib examples.
    """
    # Calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    
    # Rotate theta to start at the top
    theta += np.pi/2
    
    # Create figure and polar subplot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='polar')
    
    # Set plot direction to clockwise
    ax.set_theta_direction(-1)
    
    # Set first axis to be at the top
    ax.set_theta_offset(np.pi/2)
    
    # Draw axis lines for each angle and label
    ax.set_thetagrids(np.degrees(theta), [])
    
    # Return the figure and subplot
    return fig, ax, theta

def create_agent_profile(df):
    """
    Create a radar chart showing XWWDLtVr's behavioral features
    compared to population averages.
    
    Args:
        df: DataFrame with agent data including features
    """
    print("Generating agent profile radar chart...")
    
    # Select feature columns (excluding metadata columns)
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    
    # Define key agent and get its index
    agent_id = 'XWWDLtVr'
    agent_idx = df[df['agent_id'] == agent_id].index[0]
    
    # Calculate population average (excluding the agent of interest)
    population_mask = df['agent_id'] != agent_id
    population_avg = df.loc[population_mask, feature_cols].mean()
    
    # Get agent features
    agent_features = df.loc[agent_idx, feature_cols]
    
    # Normalize both to 0-1 scale for visualization
    # Find min and max for all features
    all_features = df[feature_cols].values
    min_vals = np.min(all_features, axis=0)
    max_vals = np.max(all_features, axis=0)
    
    # Normalize
    norm_agent = (agent_features - min_vals) / (max_vals - min_vals)
    norm_pop = (population_avg - min_vals) / (max_vals - min_vals)
    
    # Create readable feature names
    feature_names = [
        'Resource Acquisition',
        'Movement Pattern',
        'Social Interaction',
        'Goal Orientation',
        'Exploration Tendency',
        'Risk Tolerance',
        'Competition Strategy',
        'Cooperation Likelihood',
        'Resource Management',
        'Activity Rhythm',
        'Environment Awareness',
        'Decision Consistency'
    ]
    
    # Use only a subset of features for clearer visualization
    n_features = min(len(feature_names), 12)
    
    # Create radar chart
    fig, ax, theta = radar_factory(n_features)
    
    # Plot data
    ax.plot(theta, norm_agent.values[:n_features], 'o-', linewidth=2.5, 
            color='red', label=f'Agent {agent_id}')
    ax.fill(theta, norm_agent.values[:n_features], alpha=0.25, color='red')
    
    ax.plot(theta, norm_pop.values[:n_features], 'o-', linewidth=2.5,
            color='blue', label='Population Average')
    ax.fill(theta, norm_pop.values[:n_features], alpha=0.25, color='blue')
    
    # Set labels with better positioning and readability
    for i, label in enumerate(feature_names[:n_features]):
        angle = theta[i]
        # Calculate position with larger radius to prevent overlap with plot
        x = np.cos(angle) * 1.4
        y = np.sin(angle) * 1.4
        
        # Adjust horizontal alignment based on angle
        ha = 'left'
        if angle > np.pi/2 and angle < 3*np.pi/2:
            ha = 'right'
        
        # Add text with background for better readability
        plt.text(angle, 1.3, label, 
                ha=ha, va='center', 
                rotation=np.degrees(angle - np.pi/2),
                fontsize=11, weight='bold',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
    
    # Set radial ticks and grid
    ax.set_rlabel_position(0)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_rmax(1.2)  # Extend the radius a bit to make room for labels
    
    # Add title with more space
    plt.title(f'Behavioral Profile: Agent {agent_id} vs. Population Average',
              fontsize=18, pad=50, weight='bold')
    
    # Add legend with better positioning
    plt.legend(loc='lower right', fontsize=12, framealpha=0.9)
    
    # Add explanatory text
    plt.figtext(
        0.5, 0.02, 
        f"Radar chart comparing Agent {agent_id}'s behavioral features with population averages.\n"
        "The distinctive shape indicates fundamental differences in operational strategy.",
        ha='center', fontsize=12, style='italic',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
    )
    
    # Save figure with additional padding
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('figures/xwwdltvr_profile.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Agent profile radar chart saved to figures/xwwdltvr_profile.png") 