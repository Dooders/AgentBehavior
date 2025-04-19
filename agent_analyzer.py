#!/usr/bin/env python
"""
Script to analyze why agent beAEmVqo is different from other agents.
This script loads agent vector embeddings from the FAISS index,
compares the target agent to others, and provides visualizations
and statistical analysis of the differences.
"""

import os
import json
import sqlite3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from scipy.stats import ttest_ind
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from collections import Counter, defaultdict

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


def get_agent_metadata(db_path, agent_id=None):
    """Get metadata for specified agent or all agents if no ID provided."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if agent_id:
        cursor.execute(
            """
            SELECT agent_id, agent_type, birth_time, death_time, position_x, position_y, 
                   initial_resources, starting_health, starvation_threshold, 
                   genome_id, generation, action_weights
            FROM agents 
            WHERE agent_id = ?
            """, 
            (agent_id,)
        )
        row = cursor.fetchone()
        result = dict(row) if row else None
    else:
        cursor.execute(
            """
            SELECT agent_id, agent_type, birth_time, death_time, position_x, position_y,
                   initial_resources, starting_health, starvation_threshold,
                   genome_id, generation, action_weights
            FROM agents
            """
        )
        rows = cursor.fetchall()
        result = {row['agent_id']: dict(row) for row in rows}
    
    conn.close()
    return result


def get_agent_actions(db_path, agent_id):
    """Get actions performed by the specified agent."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT action_type, COUNT(*) as count
        FROM agent_actions
        WHERE agent_id = ?
        GROUP BY action_type
        ORDER BY COUNT(*) DESC
        """,
        (agent_id,)
    )
    
    actions = {row['action_type']: row['count'] for row in cursor.fetchall()}
    conn.close()
    return actions


def get_agent_states(db_path, agent_id):
    """Get state history for the specified agent."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT step_number, position_x, position_y, resource_level, 
               current_health, is_defending, total_reward, age
        FROM agent_states
        WHERE agent_id = ?
        ORDER BY step_number ASC
        """,
        (agent_id,)
    )
    
    states = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return states


def extract_agent_state_vectors(faiss_index, target_agent_id=None):
    """Extract state vectors for specified agent or all agents."""
    agent_state_vectors = defaultdict(dict)
    
    # Process all IDs in the FAISS index
    for i, id_str in enumerate(faiss_index.ids):
        parts = id_str.split('-')
        if len(parts) > 1:
            agent_id = parts[0]
            step_number = parts[1] if len(parts) > 1 else "unknown"
            
            # If target_agent_id is specified, only collect vectors for that agent
            if target_agent_id and agent_id != target_agent_id:
                continue
                
            # Get the vector from the FAISS index
            vector = faiss_index.index.reconstruct(i)
            
            # Store with both agent_id and step_number
            agent_state_vectors[agent_id][step_number] = vector
    
    return agent_state_vectors


def analyze_vector_statistics(agent_vectors):
    """Calculate statistical properties of agent vectors."""
    stats = {}
    
    for agent_id, states in agent_vectors.items():
        # Convert state vectors to a numpy array
        vectors = np.array(list(states.values()))
        
        # Calculate basic statistics
        stats[agent_id] = {
            "mean": np.mean(vectors, axis=0),
            "std": np.std(vectors, axis=0),
            "min": np.min(vectors, axis=0),
            "max": np.max(vectors, axis=0),
            "num_states": len(vectors),
            # Calculate self-similarity (cosine similarity between consecutive states)
            "self_similarity": np.mean([
                cosine_similarity(vectors[i:i+1], vectors[i+1:i+2])[0][0]
                for i in range(len(vectors)-1)
            ]) if len(vectors) > 1 else 0
        }
    
    return stats


def plot_agent_comparison(target_id, all_vectors, agent_metadata, stats, output_file=None):
    """Create visualizations comparing the target agent to others."""
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(f"Analysis of Agent {target_id}", fontsize=16)
    
    # 1. Vector Mean Comparison
    ax1 = fig.add_subplot(2, 2, 1)
    target_mean = stats[target_id]["mean"]
    other_means = np.array([stats[a]["mean"] for a in stats if a != target_id])
    
    # Calculate t-test for each dimension
    t_values = []
    p_values = []
    
    for i in range(len(target_mean)):
        t, p = ttest_ind(
            np.array([target_mean[i]]), 
            other_means[:, i],
            equal_var=False
        )
        t_values.append(t)
        p_values.append(p)
    
    # Plot dimensions with significant differences
    sig_dims = np.where(np.array(p_values) < 0.01)[0]
    
    if len(sig_dims) > 0:
        sig_dims = sig_dims[:10]  # Take top 10 most significant dimensions
        x = np.arange(len(sig_dims))
        width = 0.35
        
        # Plot target agent values
        target_values = [target_mean[i] for i in sig_dims]
        ax1.bar(x - width/2, target_values, width, label=f'Agent {target_id}')
        
        # Plot average of other agents
        other_values = [np.mean(other_means[:, i]) for i in sig_dims]
        ax1.bar(x + width/2, other_values, width, label='Other Agents (avg)')
        
        ax1.set_xlabel('Vector Dimensions with Significant Differences')
        ax1.set_ylabel('Value')
        ax1.set_title('Most Different Vector Dimensions')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Dim {i}' for i in sig_dims], rotation=45)
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, "No significantly different dimensions found", 
                 horizontalalignment='center', verticalalignment='center')
    
    # 2. Self-Similarity vs. Other Agents
    ax2 = fig.add_subplot(2, 2, 2)
    target_states = list(all_vectors[target_id].values())
    
    # Calculate average similarity to other agents
    similarity_to_others = {}
    for other_id in [a for a in all_vectors if a != target_id]:
        other_states = list(all_vectors[other_id].values())
        similarities = []
        
        # Sample if there are too many comparisons
        max_comparisons = 100
        if len(target_states) * len(other_states) > max_comparisons:
            # Sample pairs of states
            pairs = np.random.choice(
                len(target_states) * len(other_states), 
                max_comparisons, 
                replace=False
            )
            for idx in pairs:
                i = idx // len(other_states)
                j = idx % len(other_states)
                sim = cosine_similarity([target_states[i]], [other_states[j]])[0][0]
                similarities.append(sim)
        else:
            # Calculate all pairwise similarities
            for target_state in target_states:
                for other_state in other_states:
                    sim = cosine_similarity([target_state], [other_state])[0][0]
                    similarities.append(sim)
                
        similarity_to_others[other_id] = np.mean(similarities)
    
    # Plot self-similarity vs similarity to others
    self_sim = stats[target_id]["self_similarity"]
    other_sims = list(similarity_to_others.values())
    
    labels = ['Self-Similarity'] + [f'Sim to {a[-6:]}' for a in similarity_to_others]
    values = [self_sim] + other_sims
    
    ax2.bar(range(len(values)), values)
    ax2.set_xlabel('Comparison')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Self-Similarity vs. Similarity to Other Agents')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=90)
    
    # 3. PCA visualization of target agent alongside others
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Collect vectors from target and sample from others
    vectors = []
    labels = []
    
    # Add all target agent vectors
    for vec in all_vectors[target_id].values():
        vectors.append(vec)
        labels.append(target_id)
    
    # Sample from other agents (max 10 agents, 50 states each)
    other_agents = list(all_vectors.keys())
    if target_id in other_agents:
        other_agents.remove(target_id)
    
    sample_agents = np.random.choice(other_agents, min(10, len(other_agents)), replace=False)
    for agent_id in sample_agents:
        states = list(all_vectors[agent_id].values())
        if len(states) > 50:
            # Fix: Sample indices instead of the states themselves
            indices = np.random.choice(len(states), 50, replace=False)
            sampled_states = [states[i] for i in indices]
        else:
            sampled_states = states
        
        for vec in sampled_states:
            vectors.append(vec)
            labels.append(agent_id)
    
    # Apply PCA
    vectors_array = np.array(vectors)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors_array)
    
    # Plot with different colors for each agent
    unique_agents = list(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_agents)))
    
    for i, agent in enumerate(unique_agents):
        mask = [label == agent for label in labels]
        
        # Make target agent more prominent
        if agent == target_id:
            ax3.scatter(
                reduced[mask, 0], reduced[mask, 1], 
                c=[colors[i]], s=100, marker='o', label=f'Agent {agent[-6:]}'
            )
        else:
            ax3.scatter(
                reduced[mask, 0], reduced[mask, 1], 
                c=[colors[i]], s=30, alpha=0.7, label=f'Agent {agent[-6:]}'
            )
    
    ax3.set_title('PCA of Agent State Vectors')
    ax3.set_xlabel('Component 1')
    ax3.set_ylabel('Component 2')
    ax3.legend()
    
    # 4. Agent Metadata Comparison
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Get target agent metadata
    target_meta = agent_metadata[target_id]
    
    # Select important scalar attributes
    attrs = ['generation', 'initial_resources', 'starting_health']
    attr_values = {}
    
    for attr in attrs:
        if attr in target_meta:
            # Get target value
            target_val = target_meta[attr]
            
            # Get values from other agents
            other_vals = [
                agent_metadata[a][attr] 
                for a in agent_metadata 
                if a != target_id and attr in agent_metadata[a]
            ]
            
            if other_vals:
                attr_values[attr] = {
                    'target': target_val,
                    'others_mean': np.mean(other_vals),
                    'others_std': np.std(other_vals)
                }
    
    # Plot comparison
    if attr_values:
        x = np.arange(len(attr_values))
        width = 0.35
        
        target_bars = [attr_values[attr]['target'] for attr in attr_values]
        others_bars = [attr_values[attr]['others_mean'] for attr in attr_values]
        
        ax4.bar(x - width/2, target_bars, width, label=f'Agent {target_id}')
        ax4.bar(x + width/2, others_bars, width, label='Other Agents (avg)')
        
        ax4.set_xlabel('Attribute')
        ax4.set_ylabel('Value')
        ax4.set_title('Agent Metadata Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(list(attr_values.keys()))
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, "No comparable metadata found", 
                 horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save if output file specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    else:
        plt.show()


def analyze_agent_behavior(db_path, target_id, all_vectors, output_file=None):
    """Analyze and visualize the behavior patterns of the target agent."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get target agent actions
    target_actions = get_agent_actions(db_path, target_id)
    
    # Get actions for other agents for comparison
    cursor.execute(
        """
        SELECT DISTINCT agent_id FROM agent_actions
        WHERE agent_id != ? 
        LIMIT 10
        """,
        (target_id,)
    )
    
    comparison_agents = [row['agent_id'] for row in cursor.fetchall()]
    
    # Get action distributions for comparison agents
    comparison_actions = {}
    for agent_id in comparison_agents:
        comparison_actions[agent_id] = get_agent_actions(db_path, agent_id)
    
    # Standardize action types across all agents
    all_action_types = set()
    for actions in [target_actions] + list(comparison_actions.values()):
        all_action_types.update(actions.keys())
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle(f"Behavioral Analysis of Agent {target_id}", fontsize=16)
    
    # 1. Action Distribution Comparison
    # Prepare data
    agents = [target_id] + comparison_agents[:5]  # Limit to 5 comparison agents for readability
    data = []
    
    for agent_id in agents:
        agent_data = {}
        
        # Get actions for this agent
        if agent_id == target_id:
            actions = target_actions
        else:
            actions = comparison_actions[agent_id]
        
        # Fill in all action types
        for action_type in all_action_types:
            agent_data[action_type] = actions.get(action_type, 0)
        
        # Calculate total actions to get percentages
        total_actions = sum(agent_data.values())
        if total_actions > 0:
            for action_type in agent_data:
                agent_data[action_type] = (agent_data[action_type] / total_actions) * 100
        
        data.append(agent_data)
    
    # Plot as a grouped bar chart
    width = 0.15
    x = np.arange(len(all_action_types))
    
    for i, agent_id in enumerate(agents):
        values = [data[i].get(action, 0) for action in all_action_types]
        offset = width * (i - len(agents)/2 + 0.5)
        
        # Make target agent stand out
        if agent_id == target_id:
            ax1.bar(x + offset, values, width, label=f'Agent {agent_id[-6:]}', 
                    color='red', alpha=0.8)
        else:
            ax1.bar(x + offset, values, width, label=f'Agent {agent_id[-6:]}', 
                    alpha=0.6)
    
    ax1.set_xlabel('Action Type')
    ax1.set_ylabel('Percentage of Total Actions (%)')
    ax1.set_title('Action Distribution Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(list(all_action_types), rotation=45)
    ax1.legend()
    
    # 2. Target Agent State Trajectories
    states = get_agent_states(db_path, target_id)
    
    if states:
        # Extract time series data
        steps = [s['step_number'] for s in states]
        resources = [s['resource_level'] for s in states]
        health = [s['current_health'] for s in states]
        rewards = [s['total_reward'] for s in states]
        
        # Plot multiple metrics on the same axis with different scales
        color1, color2, color3 = 'blue', 'green', 'purple'
        
        # First axis for resources and health
        ax2.set_xlabel('Step Number')
        ax2.set_ylabel('Resources / Health', color=color1)
        ax2.plot(steps, resources, color=color1, label='Resources')
        ax2.plot(steps, health, color=color2, label='Health')
        ax2.tick_params(axis='y', labelcolor=color1)
        
        # Second axis for rewards
        ax3 = ax2.twinx()
        ax3.set_ylabel('Total Reward', color=color3)
        ax3.plot(steps, rewards, color=color3, label='Reward', linestyle='--')
        ax3.tick_params(axis='y', labelcolor=color3)
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax2.set_title(f'State Trajectories for Agent {target_id}')
    else:
        ax2.text(0.5, 0.5, "No state data found for this agent", 
                horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save if output file specified
    if output_file:
        behavior_file = output_file.replace('.png', '_behavior.png')
        plt.savefig(behavior_file, dpi=300, bbox_inches='tight')
        print(f"Behavior analysis saved to {behavior_file}")
    else:
        plt.show()
    
    conn.close()


def find_agent_by_prefix(db_path, id_prefix):
    """Find the full agent ID that starts with the given prefix."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT agent_id
        FROM agents
        WHERE agent_id LIKE ?
        LIMIT 10
        """,
        (f"{id_prefix}%",)
    )
    
    matching_ids = [row['agent_id'] for row in cursor.fetchall()]
    conn.close()
    
    return matching_ids


def main():
    """Main function to analyze the target agent."""
    parser = argparse.ArgumentParser(
        description="Analyze why agent 56q2nhmuN2SqH9beAEmVqo is different from others"
    )
    
    parser.add_argument(
        "--target_agent", 
        type=str, 
        default="56q2nhmuN2SqH9beAEmVqo",
        help="ID or ID prefix of the agent to analyze"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default="agent_analysis.png",
        help="Path to save the analysis visualization"
    )
    
    args = parser.parse_args()
    
    # Find the full agent ID from the prefix
    print(f"Finding agent(s) with ID prefix '{args.target_agent}'...")
    matching_agent_ids = find_agent_by_prefix(DB_PATH, args.target_agent)
    
    if not matching_agent_ids:
        print(f"Error: No agents found with ID prefix '{args.target_agent}'")
        return
    
    if len(matching_agent_ids) > 1:
        print(f"Found {len(matching_agent_ids)} agents matching the prefix '{args.target_agent}':")
        for i, agent_id in enumerate(matching_agent_ids):
            print(f"  {i+1}. {agent_id}")
        
        # Use the first matching agent by default
        target_agent_id = matching_agent_ids[0]
        print(f"\nUsing the first matching agent: {target_agent_id}")
    else:
        target_agent_id = matching_agent_ids[0]
        print(f"Found agent with ID: {target_agent_id}")
    
    # 1. Load FAISS index
    print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
    faiss_index = FaissVectorIndex.load(FAISS_INDEX_PATH)
    print(f"Loaded index with {len(faiss_index.ids)} vectors")
    
    # 2. Get agent metadata
    print("Loading agent metadata from database...")
    agent_metadata = get_agent_metadata(DB_PATH)
    
    if target_agent_id not in agent_metadata:
        print(f"Error: Agent {target_agent_id} not found in database")
        return
    
    print(f"Analyzing agent {target_agent_id}...")
    
    # 3. Extract state vectors for all agents
    print("Extracting state vectors from FAISS index...")
    all_vectors = extract_agent_state_vectors(faiss_index)
    
    if target_agent_id not in all_vectors:
        print(f"Error: No state vectors found for agent {target_agent_id} in FAISS index")
        return
    
    # 4. Calculate vector statistics
    print("Calculating vector statistics...")
    vector_stats = analyze_vector_statistics(all_vectors)
    
    # 5. Generate visualizations
    print("Generating analysis visualizations...")
    plot_agent_comparison(
        target_agent_id, 
        all_vectors, 
        agent_metadata, 
        vector_stats, 
        args.output_file
    )
    
    # 6. Analyze agent behavior patterns
    print("Analyzing agent behavior patterns...")
    analyze_agent_behavior(
        DB_PATH, 
        target_agent_id, 
        all_vectors, 
        args.output_file
    )
    
    print("Analysis complete!")


if __name__ == "__main__":
    main() 