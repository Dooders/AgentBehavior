import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patheffects as PathEffects

def create_metric_agreement_viz(df, isolation_metrics):
    """
    Create a visualization showing the overlap of agents identified as
    anomalous across different metrics.
    
    Args:
        df: DataFrame with agent data
        isolation_metrics: Dictionary of calculated isolation metrics
    """
    print("Generating metric agreement visualization...")
    
    # Define the threshold for anomaly detection
    threshold = 2.0
    
    # Select key metrics (excluding local density which wasn't effective)
    metrics = [
        'mahalanobis_distance',
        'euclidean_distance', 
        'cosine_distance',
        'mean_pairwise_distance',
        'nearest_neighbor_distance'
    ]
    
    # Find anomalous agents for each metric
    anomalous_agents = {}
    for metric in metrics:
        z_scores = isolation_metrics[metric]
        # Get indices of agents above threshold
        anomalous_idx = np.where(z_scores > threshold)[0]
        # Get agent IDs
        anomalous_agents[metric] = set(df.iloc[anomalous_idx]['agent_id'])
    
    # Create a graph to visualize the relationships
    G = nx.Graph()
    
    # Add agent nodes
    all_anomalous = set()
    for agents in anomalous_agents.values():
        all_anomalous.update(agents)
    
    # Focus on key agents
    key_agents = ['XWWDLtVr', 'nq7AEggt', 'SCSMTVA2', '6xScYvpu', 'mAN8Vx78', 
                 '7mkdNKSM', 'wDRrgAYS', 'X3DvCEoN']
    
    # Count metrics for each agent
    agent_counts = {agent: 0 for agent in all_anomalous}
    for metric, agents in anomalous_agents.items():
        for agent in agents:
            agent_counts[agent] += 1
    
    # Compute node sizes based on count
    sizes = {agent: 300 * count for agent, count in agent_counts.items()}
    
    # Add agent nodes
    for agent in all_anomalous:
        if agent in key_agents:
            # Highlight key agents
            if agent == 'XWWDLtVr':
                color = 'red'
            elif agent == 'nq7AEggt':
                color = 'blue'
            elif agent == 'SCSMTVA2':
                color = 'green'
            elif agent == '6xScYvpu':
                color = 'orange'
            else:
                color = 'purple'
            
            G.add_node(agent, type='agent', size=sizes[agent], 
                       color=color, count=agent_counts[agent])
        elif agent_counts[agent] >= 2:
            # Only show other agents if they appear in at least 2 metrics
            G.add_node(agent, type='agent', size=sizes[agent], 
                       color='gray', count=agent_counts[agent])
    
    # Add metric nodes
    for metric in metrics:
        G.add_node(metric, type='metric', size=1200, color='lightblue')
    
    # Add edges between metrics and agents
    for metric, agents in anomalous_agents.items():
        for agent in agents:
            if agent in G.nodes():
                G.add_edge(metric, agent)
    
    # Create figure
    plt.figure(figsize=(16, 12))
    
    # Create layout with more space between nodes
    # Position metrics in a circular layout with larger radius
    metric_pos = nx.circular_layout(metrics, scale=1.0)
    
    # Position agents using spring layout but with more spacing
    agent_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'agent']
    agent_subgraph = G.subgraph(agent_nodes)
    agent_pos = nx.spring_layout(agent_subgraph, k=0.5, iterations=100)
    
    # Combine positions
    pos = {**metric_pos, **agent_pos}
    
    # Draw metric nodes
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=[n for n, attr in G.nodes(data=True) if attr['type'] == 'metric'],
        node_size=[G.nodes[n]['size'] for n in G.nodes() if G.nodes[n]['type'] == 'metric'],
        node_color=[G.nodes[n]['color'] for n in G.nodes() if G.nodes[n]['type'] == 'metric'],
        alpha=0.8, edgecolors='black', linewidths=2
    )
    
    # Draw agent nodes
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=[n for n, attr in G.nodes(data=True) if attr['type'] == 'agent'],
        node_size=[G.nodes[n]['size'] for n in G.nodes() if G.nodes[n]['type'] == 'agent'],
        node_color=[G.nodes[n]['color'] for n in G.nodes() if G.nodes[n]['type'] == 'agent'],
        alpha=0.9, edgecolors='black', linewidths=1
    )
    
    # Draw edges with curved style for clarity
    nx.draw_networkx_edges(
        G, pos, width=1.2, alpha=0.6,
        edge_color='gray',
        connectionstyle='arc3,rad=0.1'
    )
    
    # Draw metric labels with white outlines for better visibility
    metric_labels = {n: n.replace('_distance', '') for n in metrics}
    for node, label in metric_labels.items():
        x, y = pos[node]
        text = plt.text(
            x, y, label, 
            fontsize=14, fontweight='bold', ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3')
        )
        text.set_path_effects([
            PathEffects.withStroke(linewidth=3, foreground='white')
        ])
    
    # Draw agent labels with backgrounds for better visibility
    for node, attr in G.nodes(data=True):
        if attr['type'] == 'agent':
            x, y = pos[node]
            count = attr['count']
            # Add background to text for better readability
            plt.text(
                x, y, f"{node}\n({count} metrics)", 
                fontsize=10, fontweight='bold', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2')
            )
    
    # Add title
    plt.title('Metric Agreement: Agents Identified as Anomalous Across Metrics',
              fontsize=20, pad=20)
    
    # Add explanatory text box with better positioning
    textstr = '\n'.join([
        'Node Size: Number of metrics identifying the agent',
        'Edge: Agent identified by connected metric',
        'XWWDLtVr (red): Appears in 5 metrics',
        'nq7AEggt (blue): Appears in 4 metrics',
        'SCSMTVA2 (green): Appears in 3 metrics',
        '6xScYvpu (orange): Appears in 2 metrics'
    ])
    
    plt.figtext(0.02, 0.02, textstr, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
    
    # Remove axis
    plt.axis('off')
    
    # Add explanatory text
    plt.figtext(
        0.5, 0.01, 
        "Overlap analysis of anomalous agents across different isolation metrics.\n"
        "Agents appearing in multiple metrics represent more reliable indicators of true behavioral divergence.",
        ha='center', fontsize=12, style='italic',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
    )
    
    # Save figure with extra padding
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig('figures/metric_agreement.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Metric agreement visualization saved to figures/metric_agreement.png")

    # Alternative implementation if matplotlib_venn is not available
    try:
        # Check if matplotlib_venn is imported successfully
        import matplotlib_venn
    except ImportError:
        print("Note: matplotlib_venn not available. Using network visualization instead.") 