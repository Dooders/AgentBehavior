#!/usr/bin/env python
"""
Script to visualize agent embeddings in 2D.
This script loads agent vector embeddings from the FAISS index,
reduces dimensionality to 2D using t-SNE or PCA, and creates a scatter plot
visualization for a sample of agents.
"""

import argparse
import json
import sqlite3
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import faiss

# Set data directory
DATA_DIR = "data"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")
DB_PATH = os.path.join(DATA_DIR, "simulation.db")


class FaissVectorIndex:
    """FAISS-based vector index for efficient similarity search."""

    @classmethod
    def load(cls, filepath: str):
        """Load a FAISS index from a file."""
        try:
            # Load metadata
            with open(f"{filepath}.json", "r") as f:
                data = json.load(f)

            # Create instance
            instance = cls()
            instance.dimension = data["dimension"]
            instance.metric = data.get("metric", "cosine")
            instance.index_type = data.get("index_type", "Flat")

            # Load FAISS index
            instance.index = faiss.read_index(f"{filepath}.faiss")

            # Load IDs and metadata
            instance.ids = data["ids"]
            instance.metadata = data["metadata"]

            return instance
        except Exception as e:
            print(f"Failed to load FAISS index: {str(e)}")
            raise


def get_agent_ids(db_path, limit=20):
    """Get a random sample of agent IDs from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get total count of agents
    cursor.execute("SELECT COUNT(*) FROM agents")
    total_agents = cursor.fetchone()[0]
    
    # Get all agent IDs
    cursor.execute("SELECT agent_id FROM agents")
    agent_ids = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    
    # Randomly sample agent IDs
    if len(agent_ids) > limit:
        agent_ids = random.sample(agent_ids, limit)
    
    return agent_ids


def get_agent_metadata(db_path, agent_ids):
    """Get metadata for the specified agents."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    agent_metadata = {}
    
    for agent_id in agent_ids:
        cursor.execute(
            """
            SELECT agent_id, agent_type, position_x, position_y, generation
            FROM agents
            WHERE agent_id = ?
            """,
            (agent_id,)
        )
        row = cursor.fetchone()
        if row:
            agent_metadata[agent_id] = dict(row)
    
    conn.close()
    return agent_metadata


def extract_agent_state_vectors(faiss_index, agent_ids):
    """Extract all state vectors for specified agent IDs from the FAISS index."""
    agent_state_vectors = {}
    
    # Print the first few IDs from the index for debugging
    print(f"First 5 index IDs: {faiss_index.ids[:5] if len(faiss_index.ids) >= 5 else faiss_index.ids}")
    print(f"Looking for states from agent IDs: {agent_ids[:5] if len(agent_ids) >= 5 else agent_ids}")
    
    # Try to find state IDs in the format "agent_id-step_number"
    found_agents = set()
    for i, id_str in enumerate(faiss_index.ids):
        parts = id_str.split('-')
        if len(parts) > 1:
            agent_id = parts[0]
            step_number = parts[1] if len(parts) > 1 else "unknown"
            
            if agent_id in agent_ids:
                if agent_id not in found_agents:
                    found_agents.add(agent_id)
                    print(f"Found vectors for agent {agent_id}")
                
                # Get the vector from the FAISS index
                vector = faiss_index.index.reconstruct(i)
                
                # Store with both agent_id and step_number
                if agent_id not in agent_state_vectors:
                    agent_state_vectors[agent_id] = {}
                
                agent_state_vectors[agent_id][step_number] = vector
    
    # Print stats about what we found
    print(f"Found vectors for {len(found_agents)} agents out of {len(agent_ids)} requested")
    for agent_id, states in agent_state_vectors.items():
        print(f"  Agent {agent_id}: {len(states)} states")
    
    return agent_state_vectors


def reduce_state_dimensions(agent_state_vectors, method="tsne", n_components=2):
    """Reduce dimensionality of all state vectors using t-SNE or PCA."""
    # Flatten all vectors while tracking agent_id and step_number
    flat_vectors = []
    vector_metadata = []
    
    for agent_id, states in agent_state_vectors.items():
        for step_number, vector in states.items():
            flat_vectors.append(vector)
            vector_metadata.append({"agent_id": agent_id, "step_number": step_number})
    
    # Check if we have vectors to reduce
    if not flat_vectors:
        print("No vectors to reduce dimensions for.")
        return {}
    
    # Convert to numpy array
    vector_matrix = np.array(flat_vectors)
    
    # Apply dimensionality reduction based on method
    if method.lower() == "tsne":
        # Apply t-SNE - perplexity must be less than n_samples
        perplexity = min(30, max(5, len(flat_vectors)//5))  # Adjust perplexity based on sample size
        print(f"Using t-SNE with perplexity value: {perplexity} for {len(flat_vectors)} vectors")
        
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
        reduced_vectors = tsne.fit_transform(vector_matrix)
    elif method.lower() == "pca":
        # Apply PCA
        print(f"Using PCA to reduce {len(flat_vectors)} vectors to {n_components} dimensions")
        pca = PCA(n_components=n_components, random_state=42)
        reduced_vectors = pca.fit_transform(vector_matrix)
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    # Return the reduced vectors with their metadata
    result = []
    for i, (coords, meta) in enumerate(zip(reduced_vectors, vector_metadata)):
        result.append({
            "agent_id": meta["agent_id"],
            "step_number": meta["step_number"],
            "coords": coords
        })
    
    return result


def visualize_agent_states(reduced_vectors, agent_metadata, output_file=None, method="tsne"):
    """Create a scatter plot visualization of agent states in 2D space."""
    if not reduced_vectors:
        print("No vectors to visualize.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Generate a color map for agents - one color per agent
    agent_ids = list(set(v["agent_id"] for v in reduced_vectors))
    agent_id_to_color = {}
    colormap = plt.cm.get_cmap('tab20', len(agent_ids))
    
    for i, agent_id in enumerate(agent_ids):
        agent_id_to_color[agent_id] = colormap(i)
    
    # Group data by agent ID for more efficient plotting
    agent_data = {}
    for vector_data in reduced_vectors:
        agent_id = vector_data["agent_id"]
        if agent_id not in agent_data:
            agent_data[agent_id] = {
                "x": [],
                "y": [],
                "color": agent_id_to_color[agent_id]
            }
        agent_data[agent_id]["x"].append(vector_data["coords"][0])
        agent_data[agent_id]["y"].append(vector_data["coords"][1])
    
    # Plot each agent's states as a single scatter plot
    for agent_id, data in agent_data.items():
        # Get shortened agent ID for the legend
        short_id = f"Agent {agent_id[-8:]}"
        ax.scatter(data["x"], data["y"], c=[data["color"]], s=100, alpha=0.7, label=short_id)
    
    # Add title and labels
    method_name = "t-SNE" if method.lower() == "tsne" else "PCA"
    ax.set_title(f'Agent State Embeddings Visualization ({method_name})')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    
    # Create legend with unique agent IDs
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Agents", 
              loc='center left', bbox_to_anchor=(1, 0.5), 
              ncol=1 + len(agent_ids)//20)
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    fig.subplots_adjust(right=0.75)
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    else:
        plt.show()


def visualize_side_by_side(pca_vectors, tsne_vectors, agent_metadata, output_file=None):
    """Create a side-by-side visualization of PCA and t-SNE in a single image."""
    if not pca_vectors or not tsne_vectors:
        print("No vectors to visualize.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Generate a color map for agents - one color per agent
    agent_ids = list(set(v["agent_id"] for v in pca_vectors))
    agent_id_to_color = {}
    colormap = plt.cm.get_cmap('tab20', len(agent_ids))
    
    for i, agent_id in enumerate(agent_ids):
        agent_id_to_color[agent_id] = colormap(i)
    
    # Helper function to plot on a specific axis
    def plot_on_axis(ax, vectors, method_name):
        # Group data by agent ID for more efficient plotting
        agent_data = {}
        for vector_data in vectors:
            agent_id = vector_data["agent_id"]
            if agent_id not in agent_data:
                agent_data[agent_id] = {
                    "x": [],
                    "y": [],
                    "color": agent_id_to_color[agent_id]
                }
            agent_data[agent_id]["x"].append(vector_data["coords"][0])
            agent_data[agent_id]["y"].append(vector_data["coords"][1])
        
        # Plot each agent's states
        for agent_id, data in agent_data.items():
            # Get shortened agent ID for the legend
            short_id = f"Agent {agent_id[-8:]}"
            ax.scatter(data["x"], data["y"], c=[data["color"]], s=100, alpha=0.7, label=short_id)
        
        # Set titles and labels
        ax.set_title(f'Agent States ({method_name})')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
    
    # Plot PCA and t-SNE
    plot_on_axis(ax1, pca_vectors, "PCA")
    plot_on_axis(ax2, tsne_vectors, "t-SNE")
    
    # Create a single legend for the entire figure
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), title="Agents", 
               loc='lower center', bbox_to_anchor=(0.5, 0.01),
               ncol=min(5, len(agent_ids)))
    
    fig.suptitle('Agent State Embeddings Comparison: PCA vs t-SNE', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Side-by-side visualization saved to {output_file}")
    else:
        plt.show()


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Visualize agent state embeddings in 2D"
    )
    
    parser.add_argument(
        "--num_agents",
        type=int,
        default=20,
        help="Number of agents to sample for visualization"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default="agent_states_visualization.png",
        help="Path to save the visualization"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        choices=["tsne", "pca", "both", "side_by_side"],
        default="tsne",
        help="Dimensionality reduction method to use (tsne, pca, both, or side_by_side for comparison)"
    )
    
    args = parser.parse_args()
    
    # Load FAISS index
    print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
    faiss_index = FaissVectorIndex.load(FAISS_INDEX_PATH)
    print(f"Loaded index with {len(faiss_index.ids)} vectors")
    
    # Get random sample of agent IDs
    print(f"Sampling {args.num_agents} agents from database...")
    agent_ids = get_agent_ids(DB_PATH, limit=args.num_agents)
    print(f"Selected {len(agent_ids)} agents")
    
    # Get agent metadata
    agent_metadata = get_agent_metadata(DB_PATH, agent_ids)
    
    # Extract all state vectors for agents
    print("Extracting state vectors from FAISS index...")
    agent_state_vectors = extract_agent_state_vectors(faiss_index, agent_ids)
    
    if args.method == "both":
        # Generate both PCA and t-SNE visualizations with the same agents
        print("Generating both PCA and t-SNE visualizations...")
        
        # Generate PCA visualization
        print("Reducing vector dimensions using PCA...")
        pca_reduced_vectors = reduce_state_dimensions(agent_state_vectors, method="pca")
        pca_output_file = args.output_file.replace(".png", "_pca.png") if args.output_file.endswith(".png") else f"{args.output_file}_pca.png"
        print("Creating PCA visualization...")
        visualize_agent_states(pca_reduced_vectors, agent_metadata, pca_output_file, method="pca")
        
        # Generate t-SNE visualization
        print("Reducing vector dimensions using t-SNE...")
        tsne_reduced_vectors = reduce_state_dimensions(agent_state_vectors, method="tsne")
        tsne_output_file = args.output_file.replace(".png", "_tsne.png") if args.output_file.endswith(".png") else f"{args.output_file}_tsne.png"
        print("Creating t-SNE visualization...")
        visualize_agent_states(tsne_reduced_vectors, agent_metadata, tsne_output_file, method="tsne")
        
        print("Done!")
    elif args.method == "side_by_side":
        # Generate side-by-side comparison of PCA and t-SNE
        print("Generating side-by-side comparison of PCA and t-SNE...")
        
        # Generate PCA reduced vectors
        print("Reducing vector dimensions using PCA...")
        pca_reduced_vectors = reduce_state_dimensions(agent_state_vectors, method="pca")
        
        # Generate t-SNE reduced vectors
        print("Reducing vector dimensions using t-SNE...")
        tsne_reduced_vectors = reduce_state_dimensions(agent_state_vectors, method="tsne")
        
        # Create side-by-side visualization
        print("Creating side-by-side visualization...")
        side_by_side_output = args.output_file.replace(".png", "_comparison.png") if args.output_file.endswith(".png") else f"{args.output_file}_comparison.png"
        visualize_side_by_side(pca_reduced_vectors, tsne_reduced_vectors, agent_metadata, side_by_side_output)
        
        print("Done!")
    else:
        # Generate just one visualization based on the specified method
        print(f"Reducing vector dimensions using {args.method.upper()}...")
        reduced_vectors = reduce_state_dimensions(agent_state_vectors, method=args.method)
        
        # Visualize
        print("Creating visualization...")
        visualize_agent_states(reduced_vectors, agent_metadata, args.output_file, method=args.method)
        print("Done!")


if __name__ == "__main__":
    main() 