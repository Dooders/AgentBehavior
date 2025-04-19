import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sqlite3
import argparse
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Configuration
DATA_PATH = 'data'
DEFAULT_EMBED_FILE = 'agent_embeddings_3d.npy'
DB_PATH = os.path.join(DATA_PATH, 'simulation.db')

def load_reduced_embeddings(embed_path):
    """Load reduced embeddings from numpy file."""
    return np.load(embed_path)

def load_agent_metadata(db_path):
    """Load agent metadata from the simulation database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query to get agent data
    query = """
    SELECT a.agent_id, a.agent_type, a.generation, a.birth_time, a.death_time,
           MAX(s.resource_level) as max_resources, AVG(s.current_health) as avg_health,
           COUNT(act.action_id) as num_actions
    FROM agents a
    LEFT JOIN agent_states s ON a.agent_id = s.agent_id
    LEFT JOIN agent_actions act ON a.agent_id = act.agent_id
    GROUP BY a.agent_id
    """
    
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    agents_data = cursor.fetchall()
    
    # Convert to a list of dictionaries
    agents = []
    for agent in agents_data:
        agent_dict = dict(zip(columns, agent))
        agents.append(agent_dict)
    
    conn.close()
    return agents

def load_faiss_metadata(json_path):
    """Load metadata from FAISS JSON file."""
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def create_static_plots(embeddings, agent_metadata, output_dir=DATA_PATH):
    """Create static 2D and 3D plots based on various agent attributes."""
    # Extract relevant attributes for coloring
    agent_types = [agent.get('agent_type', 'unknown') for agent in agent_metadata]
    generations = [agent.get('generation', 0) for agent in agent_metadata]
    
    # Create directory for plots if it doesn't exist
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Define a colormap for agent types
    type_cmap = ListedColormap(['red', 'blue', 'green', 'orange', 'purple'])
    
    # 2D plots (using first two dimensions)
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                         c=agent_types, cmap=type_cmap, alpha=0.7)
    plt.title('Agent Embeddings by Type (2D)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.colorbar(scatter, label='Agent Type')
    plt.savefig(os.path.join(plots_dir, 'embeddings_by_type_2d.png'))
    plt.close()
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                         c=generations, cmap='viridis', alpha=0.7)
    plt.title('Agent Embeddings by Generation (2D)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.colorbar(scatter, label='Generation')
    plt.savefig(os.path.join(plots_dir, 'embeddings_by_generation_2d.png'))
    plt.close()
    
    # 3D plots if we have enough dimensions
    if embeddings.shape[1] >= 3:
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                            c=agent_types, cmap=type_cmap, alpha=0.7)
        ax.set_title('Agent Embeddings by Type (3D)')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        plt.colorbar(scatter, label='Agent Type')
        plt.savefig(os.path.join(plots_dir, 'embeddings_by_type_3d.png'))
        plt.close()
        
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                            c=generations, cmap='viridis', alpha=0.7)
        ax.set_title('Agent Embeddings by Generation (3D)')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        plt.colorbar(scatter, label='Generation')
        plt.savefig(os.path.join(plots_dir, 'embeddings_by_generation_3d.png'))
        plt.close()
    
    print(f"Static plots saved to {plots_dir}")

def create_interactive_plots(embeddings, agent_metadata, output_dir=DATA_PATH):
    """Create interactive Plotly visualizations of the embeddings."""
    # Create a DataFrame with embeddings and metadata
    df = pd.DataFrame()
    
    # Add embeddings
    for i in range(embeddings.shape[1]):
        df[f'dim_{i+1}'] = embeddings[:, i]
    
    # Add metadata
    for key in agent_metadata[0].keys():
        df[key] = [agent.get(key, None) for agent in agent_metadata]
    
    # Create directory for interactive plots
    interactive_dir = os.path.join(output_dir, 'interactive_plots')
    os.makedirs(interactive_dir, exist_ok=True)
    
    # 2D interactive plot by agent type
    fig = px.scatter(
        df, x='dim_1', y='dim_2',
        color='agent_type',
        hover_data=['agent_id', 'generation', 'birth_time', 'max_resources', 'avg_health'],
        title='Interactive Agent Embeddings by Type (2D)',
        labels={'dim_1': 'Dimension 1', 'dim_2': 'Dimension 2'},
        opacity=0.7
    )
    fig.write_html(os.path.join(interactive_dir, 'embeddings_by_type_2d.html'))
    
    # 2D interactive plot by generation
    fig = px.scatter(
        df, x='dim_1', y='dim_2',
        color='generation',
        hover_data=['agent_id', 'agent_type', 'birth_time', 'max_resources', 'avg_health'],
        title='Interactive Agent Embeddings by Generation (2D)',
        labels={'dim_1': 'Dimension 1', 'dim_2': 'Dimension 2'},
        opacity=0.7,
        color_continuous_scale='viridis'
    )
    fig.write_html(os.path.join(interactive_dir, 'embeddings_by_generation_2d.html'))
    
    # 3D interactive plot if we have enough dimensions
    if embeddings.shape[1] >= 3:
        fig = px.scatter_3d(
            df, x='dim_1', y='dim_2', z='dim_3',
            color='agent_type',
            hover_data=['agent_id', 'generation', 'birth_time', 'max_resources', 'avg_health'],
            title='Interactive Agent Embeddings by Type (3D)',
            labels={'dim_1': 'Dimension 1', 'dim_2': 'Dimension 2', 'dim_3': 'Dimension 3'},
            opacity=0.7
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='Dimension 1'),
                yaxis=dict(title='Dimension 2'),
                zaxis=dict(title='Dimension 3'),
            )
        )
        fig.write_html(os.path.join(interactive_dir, 'embeddings_by_type_3d.html'))
        
        fig = px.scatter_3d(
            df, x='dim_1', y='dim_2', z='dim_3',
            color='generation',
            hover_data=['agent_id', 'agent_type', 'birth_time', 'max_resources', 'avg_health'],
            title='Interactive Agent Embeddings by Generation (3D)',
            labels={'dim_1': 'Dimension 1', 'dim_2': 'Dimension 2', 'dim_3': 'Dimension 3'},
            opacity=0.7,
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='Dimension 1'),
                yaxis=dict(title='Dimension 2'),
                zaxis=dict(title='Dimension 3'),
            )
        )
        fig.write_html(os.path.join(interactive_dir, 'embeddings_by_generation_3d.html'))
    
    print(f"Interactive plots saved to {interactive_dir}")

def main():
    """Main function to visualize reduced embeddings."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize reduced agent embeddings')
    parser.add_argument('--embed_file', type=str, default=os.path.join(DATA_PATH, DEFAULT_EMBED_FILE),
                        help='Path to the numpy file containing reduced embeddings')
    parser.add_argument('--db_path', type=str, default=DB_PATH,
                        help='Path to the simulation database')
    parser.add_argument('--json_path', type=str, default=os.path.join(DATA_PATH, 'faiss_index.json'),
                        help='Path to the FAISS index JSON file')
    parser.add_argument('--output_dir', type=str, default=DATA_PATH,
                        help='Directory to save outputs')
    args = parser.parse_args()
    
    # Load embeddings
    print(f"Loading reduced embeddings from {args.embed_file}...")
    reduced_embeddings = load_reduced_embeddings(args.embed_file)
    print(f"Loaded embeddings with shape: {reduced_embeddings.shape}")
    
    # Load agent metadata from database
    print(f"Loading agent metadata from {args.db_path}...")
    agent_metadata = load_agent_metadata(args.db_path)
    print(f"Loaded metadata for {len(agent_metadata)} agents")
    
    # Check if the lengths match
    if len(agent_metadata) != reduced_embeddings.shape[0]:
        print(f"Warning: Number of agents in metadata ({len(agent_metadata)}) "
              f"doesn't match number of embeddings ({reduced_embeddings.shape[0]})")
        # Try to handle this by loading metadata from JSON file
        print(f"Attempting to load metadata from FAISS JSON file...")
        faiss_metadata = load_faiss_metadata(args.json_path)
        if 'metadata' in faiss_metadata and len(faiss_metadata['metadata']) == reduced_embeddings.shape[0]:
            agent_metadata = faiss_metadata['metadata']
            print(f"Using metadata from FAISS JSON file instead")
        else:
            print("Warning: Proceeding with mismatched data. Results may be incorrect.")
            # Truncate the data to match
            min_length = min(len(agent_metadata), reduced_embeddings.shape[0])
            agent_metadata = agent_metadata[:min_length]
            reduced_embeddings = reduced_embeddings[:min_length, :]
    
    # Create static plots
    print("Creating static plots...")
    create_static_plots(reduced_embeddings, agent_metadata, args.output_dir)
    
    # Create interactive plots
    print("Creating interactive plots...")
    create_interactive_plots(reduced_embeddings, agent_metadata, args.output_dir)
    
    print("Visualization complete!")

if __name__ == "__main__":
    main() 