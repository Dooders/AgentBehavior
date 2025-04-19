import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path

def create_future_work_diagram():
    """
    Create a conceptual diagram showing the planned detailed behavioral analysis
    of Agent XWWDLtVr, with different aspects of behavior to be investigated.
    """
    print("Generating future work diagram...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Central agent node
    agent_color = '#FF5733'  # Red-orange
    central_circle = plt.Circle((0.5, 0.5), 0.15, color=agent_color, alpha=0.7, 
                               edgecolor='black', linewidth=2)
    ax.add_patch(central_circle)
    
    # Add agent label
    plt.text(0.5, 0.5, 'Agent\nXWWDLtVr', 
             ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Define investigation areas
    areas = [
        {'name': 'Action Patterns', 'color': '#3498DB', 'angle': 0,
         'subnodes': ['Movement Trajectories', 'Decision Timing', 'Pattern Consistency']},
        {'name': 'Resource Management', 'color': '#2ECC71', 'angle': 72,
         'subnodes': ['Acquisition Strategy', 'Storage Behavior', 'Consumption Rate']},
        {'name': 'Social Interactions', 'color': '#9B59B6', 'angle': 144,
         'subnodes': ['Cooperation Patterns', 'Competition Strategy', 'Agent Avoidance']},
        {'name': 'Environmental Response', 'color': '#F1C40F', 'angle': 216,
         'subnodes': ['Adaptation to Changes', 'Territory Preference', 'Risk Response']},
        {'name': 'Learning Dynamics', 'color': '#1ABC9C', 'angle': 288,
         'subnodes': ['Strategy Evolution', 'Behavior Consistency', 'Exploration/Exploitation']}
    ]
    
    # Add each area as a node
    for area in areas:
        # Calculate position (in polar coordinates from center)
        angle_rad = np.radians(area['angle'])
        r_primary = 0.3  # Distance from center
        x = 0.5 + r_primary * np.cos(angle_rad)
        y = 0.5 + r_primary * np.sin(angle_rad)
        
        # Primary node
        circle = plt.Circle((x, y), 0.1, color=area['color'], alpha=0.7,
                           edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        
        # Add connection line to center
        plt.plot([0.5, x], [0.5, y], 'k-', linewidth=2, alpha=0.6)
        
        # Add label
        plt.text(x, y, area['name'], ha='center', va='center', 
                fontsize=11, fontweight='bold', wrap=True)
        
        # Add subnodes
        for i, subnode in enumerate(area['subnodes']):
            # Calculate subnode position
            sub_angle = angle_rad + np.radians(-30 + i*30)  # Spread subnodes
            r_sub = 0.2  # Distance from primary node
            sub_x = x + r_sub * np.cos(sub_angle)
            sub_y = y + r_sub * np.sin(sub_angle)
            
            # Subnode shape
            subcircle = plt.Circle((sub_x, sub_y), 0.06, color=area['color'], alpha=0.5,
                                  edgecolor='black', linewidth=1)
            ax.add_patch(subcircle)
            
            # Add connection line
            plt.plot([x, sub_x], [y, sub_y], 'k-', linewidth=1, alpha=0.4)
            
            # Add label
            plt.text(sub_x, sub_y, subnode, ha='center', va='center', 
                    fontsize=8, wrap=True)
    
    # Add hypothetical investigation tools/methods around the diagram
    methods = [
        {'name': 'Trajectory\nAnalysis', 'x': 0.15, 'y': 0.85},
        {'name': 'Statistical\nSignificance\nTesting', 'x': 0.85, 'y': 0.85},
        {'name': 'Network\nAnalysis', 'x': 0.15, 'y': 0.15},
        {'name': 'Time Series\nDecomposition', 'x': 0.85, 'y': 0.15}
    ]
    
    for method in methods:
        # Add method box
        rect = mpatches.Rectangle((method['x']-0.08, method['y']-0.06), 0.16, 0.12,
                                 color='lightgray', alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        
        # Add label
        plt.text(method['x'], method['y'], method['name'], ha='center', va='center', 
                fontsize=9, style='italic')
    
    # Add title
    plt.text(0.5, 0.95, 'Planned Investigative Approach for Agent XWWDLtVr', 
             ha='center', fontsize=18, fontweight='bold')
    
    # Add explanatory text
    plt.text(0.5, 0.05, 
             'Our future research will focus on detailed behavioral analysis of Agent XWWDLtVr,\n'
             'examining action patterns, resource management, social interactions,\n'
             'environmental responses, and learning dynamics to understand its divergence.',
             ha='center', fontsize=12, style='italic')
    
    # Set bounds and remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('figures/future_work.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Future work diagram saved to figures/future_work.png") 