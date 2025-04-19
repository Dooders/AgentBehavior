#!/usr/bin/env python
"""
Script to analyze what makes our target agent's embeddings different from other agents
by examining vector space distances and state transitions.
"""

import os
import json
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import seaborn as sns

# Constants
DATA_DIR = "data"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")
DB_PATH = os.path.join(DATA_DIR, "simulation.db")
TARGET_AGENT = "56q2nhmuN2SqH9beAEmVqo"


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


def extract_agent_state_vectors():
    """Extract state vectors for all agents from the FAISS index."""
    print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
    faiss_index = FaissVectorIndex.load(FAISS_INDEX_PATH)
    print(f"Loaded index with {len(faiss_index.ids)} vectors")
    
    agent_state_vectors = defaultdict(dict)
    agent_state_order = defaultdict(list)
    
    # Process all IDs in the FAISS index
    for i, id_str in enumerate(faiss_index.ids):
        parts = id_str.split('-')
        if len(parts) > 1:
            agent_id = parts[0]
            step_number = parts[1] if len(parts) > 1 else "unknown"
            
            # Get the vector from the FAISS index
            vector = faiss_index.index.reconstruct(i)
            
            # Store with both agent_id and step_number
            agent_state_vectors[agent_id][step_number] = vector
            agent_state_order[agent_id].append(step_number)
    
    return agent_state_vectors, agent_state_order, faiss_index


def get_agent_states_from_db():
    """Get detailed state information for all agents from the database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Query all agent states
    cursor.execute("""
        SELECT 
            agent_id,
            step_number,
            position_x,
            position_y,
            resource_level,
            current_health,
            is_defending,
            total_reward,
            age
        FROM agent_states
        ORDER BY agent_id, step_number
    """)
    
    agent_states = defaultdict(list)
    for row in cursor.fetchall():
        agent_states[row['agent_id']].append(dict(row))
    
    conn.close()
    return agent_states


def get_agent_actions_from_db():
    """Get actions performed by all agents from the database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Query all agent actions
    cursor.execute("""
        SELECT 
            agent_id,
            step_number,
            action_type,
            action_target_id,
            resources_before,
            resources_after,
            reward
        FROM agent_actions
        ORDER BY agent_id, step_number
    """)
    
    agent_actions = defaultdict(list)
    for row in cursor.fetchall():
        agent_actions[row['agent_id']].append(dict(row))
    
    conn.close()
    return agent_actions


def analyze_agent_transitions():
    """Analyze state transitions for our target agent compared to others."""
    agent_actions = get_agent_actions_from_db()
    
    # Calculate action transition matrices
    target_transitions = defaultdict(lambda: defaultdict(int))
    population_transitions = defaultdict(lambda: defaultdict(int))
    
    # Process target agent actions
    prev_action = None
    for action in agent_actions[TARGET_AGENT]:
        action_type = action['action_type']
        if prev_action:
            target_transitions[prev_action][action_type] += 1
        prev_action = action_type
    
    # Process other agents' actions
    for agent_id, actions in agent_actions.items():
        if agent_id == TARGET_AGENT:
            continue
            
        prev_action = None
        for action in actions:
            action_type = action['action_type']
            if prev_action:
                population_transitions[prev_action][action_type] += 1
            prev_action = action_type
    
    # Normalize transition matrices
    target_norm = {}
    for from_action, to_actions in target_transitions.items():
        total = sum(to_actions.values())
        target_norm[from_action] = {to: count/total for to, count in to_actions.items()}
    
    pop_norm = {}
    for from_action, to_actions in population_transitions.items():
        total = sum(to_actions.values())
        pop_norm[from_action] = {to: count/total for to, count in to_actions.items()}
    
    # Calculate transition differences
    transition_diff = {}
    for from_action in set(target_norm.keys()) | set(pop_norm.keys()):
        if from_action in target_norm and from_action in pop_norm:
            transition_diff[from_action] = {}
            for to_action in set(target_norm[from_action].keys()) | set(pop_norm[from_action].keys()):
                target_val = target_norm[from_action].get(to_action, 0)
                pop_val = pop_norm[from_action].get(to_action, 0)
                transition_diff[from_action][to_action] = target_val - pop_val
    
    return target_transitions, population_transitions, transition_diff


def analyze_vector_distances():
    """Analyze distances between agent state vectors in embedding space."""
    agent_vectors, _, _ = extract_agent_state_vectors()
    
    # Calculate average vector for each agent
    agent_avg_vectors = {}
    for agent_id, states in agent_vectors.items():
        if states:
            agent_avg_vectors[agent_id] = np.mean(list(states.values()), axis=0)
    
    # Calculate distance from target agent to all others
    target_avg = agent_avg_vectors.get(TARGET_AGENT)
    distances = {}
    
    if target_avg is not None:
        for agent_id, avg_vector in agent_avg_vectors.items():
            if agent_id != TARGET_AGENT:
                # Calculate cosine similarity (higher means more similar)
                similarity = cosine_similarity([target_avg], [avg_vector])[0][0]
                distances[agent_id] = 1 - similarity  # Convert to distance (0-2 range)
    
    # Sort agents by distance from target
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])
    
    return sorted_distances, agent_avg_vectors


def analyze_embedding_differences():
    """Analyze the dimensions with greatest difference between target and others."""
    agent_vectors, _, _ = extract_agent_state_vectors()
    
    # Calculate average vector for each agent
    agent_avg_vectors = {}
    for agent_id, states in agent_vectors.items():
        if states:
            agent_avg_vectors[agent_id] = np.mean(list(states.values()), axis=0)
    
    # Calculate population average excluding target
    pop_vectors = [v for k, v in agent_avg_vectors.items() if k != TARGET_AGENT]
    pop_avg = np.mean(pop_vectors, axis=0) if pop_vectors else None
    
    # Calculate differences along each dimension
    target_avg = agent_avg_vectors.get(TARGET_AGENT)
    if target_avg is not None and pop_avg is not None:
        dimension_diffs = target_avg - pop_avg
        
        # Get indices of dimensions with largest absolute differences
        top_diff_indices = np.argsort(np.abs(dimension_diffs))[-20:]  # Top 20 dimensions
        
        # Get the actual difference values for these dimensions
        top_diffs = [(i, dimension_diffs[i]) for i in top_diff_indices]
        
        return top_diffs, dimension_diffs, target_avg, pop_avg
    
    return None, None, None, None


def analyze_state_patterns():
    """Analyze differences in state variable patterns."""
    agent_states = get_agent_states_from_db()
    
    # Calculate statistics for target agent
    target_states = agent_states.get(TARGET_AGENT, [])
    if not target_states:
        return None
    
    target_stats = {
        'resource_level': {
            'values': [s['resource_level'] for s in target_states],
            'mean': np.mean([s['resource_level'] for s in target_states]),
            'std': np.std([s['resource_level'] for s in target_states]),
            'min': np.min([s['resource_level'] for s in target_states]),
            'max': np.max([s['resource_level'] for s in target_states]),
        },
        'health': {
            'values': [s['current_health'] for s in target_states],
            'mean': np.mean([s['current_health'] for s in target_states]),
            'std': np.std([s['current_health'] for s in target_states]),
            'min': np.min([s['current_health'] for s in target_states]),
            'max': np.max([s['current_health'] for s in target_states]),
        },
        'reward': {
            'values': [s['total_reward'] for s in target_states],
            'mean': np.mean([s['total_reward'] for s in target_states]),
            'std': np.std([s['total_reward'] for s in target_states]),
            'min': np.min([s['total_reward'] for s in target_states]),
            'max': np.max([s['total_reward'] for s in target_states]),
        },
        'position': {
            'x': [s['position_x'] for s in target_states],
            'y': [s['position_y'] for s in target_states],
        }
    }
    
    # Calculate population statistics (excluding target)
    population_stats = {}
    
    # First collect all values
    all_resources = []
    all_health = []
    all_rewards = []
    
    for agent_id, states in agent_states.items():
        if agent_id != TARGET_AGENT and states:
            all_resources.extend([s['resource_level'] for s in states])
            all_health.extend([s['current_health'] for s in states])
            all_rewards.extend([s['total_reward'] for s in states])
    
    # Calculate statistics
    if all_resources and all_health and all_rewards:
        population_stats = {
            'resource_level': {
                'mean': np.mean(all_resources),
                'std': np.std(all_resources),
                'min': np.min(all_resources),
                'max': np.max(all_resources),
            },
            'health': {
                'mean': np.mean(all_health),
                'std': np.std(all_health),
                'min': np.min(all_health),
                'max': np.max(all_health),
            },
            'reward': {
                'mean': np.mean(all_rewards),
                'std': np.std(all_rewards),
                'min': np.min(all_rewards),
                'max': np.max(all_rewards),
            }
        }
    
    return target_stats, population_stats


def visualize_differences():
    """Create visualizations of the key differences."""
    # Get state pattern analysis
    target_stats, pop_stats = analyze_state_patterns()
    
    if not target_stats or not pop_stats:
        print("Insufficient data for visualization.")
        return
    
    # Create figure with multiple plots
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f"Analysis of Agent {TARGET_AGENT} Differences", fontsize=16)
    
    # 1. Resource level distribution
    sns.histplot(target_stats['resource_level']['values'], color='blue', label='Target Agent', ax=axs[0, 0], alpha=0.5)
    axs[0, 0].axvline(x=pop_stats['resource_level']['mean'], color='red', linestyle='--', label='Population Mean')
    axs[0, 0].axvline(x=target_stats['resource_level']['mean'], color='blue', linestyle='--', label='Target Mean')
    axs[0, 0].set_title('Resource Level Distribution')
    axs[0, 0].legend()
    
    # 2. Health distribution
    sns.histplot(target_stats['health']['values'], color='blue', label='Target Agent', ax=axs[0, 1], alpha=0.5)
    axs[0, 1].axvline(x=pop_stats['health']['mean'], color='red', linestyle='--', label='Population Mean')
    axs[0, 1].axvline(x=target_stats['health']['mean'], color='blue', linestyle='--', label='Target Mean')
    axs[0, 1].set_title('Health Distribution')
    axs[0, 1].legend()
    
    # 3. Reward trajectory
    steps = range(len(target_stats['reward']['values']))
    axs[1, 0].plot(steps, target_stats['reward']['values'], color='blue', label='Target Agent')
    axs[1, 0].axhline(y=pop_stats['reward']['mean'], color='red', linestyle='--', label='Population Mean')
    axs[1, 0].set_title('Reward Trajectory')
    axs[1, 0].set_xlabel('Step')
    axs[1, 0].set_ylabel('Reward')
    axs[1, 0].legend()
    
    # 4. Resource trajectory
    axs[1, 1].plot(steps, target_stats['resource_level']['values'], color='blue', label='Target Agent')
    axs[1, 1].axhline(y=pop_stats['resource_level']['mean'], color='red', linestyle='--', label='Population Mean')
    axs[1, 1].set_title('Resource Trajectory')
    axs[1, 1].set_xlabel('Step')
    axs[1, 1].set_ylabel('Resources')
    axs[1, 1].legend()
    
    # 5. Movement patterns (position scatter)
    axs[2, 0].scatter(target_stats['position']['x'], target_stats['position']['y'], alpha=0.5, s=5)
    axs[2, 0].set_title('Movement Pattern')
    axs[2, 0].set_xlabel('X Position')
    axs[2, 0].set_ylabel('Y Position')
    
    # 6. Top embedding dimension differences
    top_diffs, _, _, _ = analyze_embedding_differences()
    if top_diffs:
        # Sort by actual difference value for visualization
        top_diffs.sort(key=lambda x: x[1])
        dim_indices = [str(dim) for dim, _ in top_diffs]
        diff_values = [diff for _, diff in top_diffs]
        
        axs[2, 1].barh(dim_indices, diff_values)
        axs[2, 1].set_title('Top Embedding Dimension Differences')
        axs[2, 1].set_xlabel('Difference (Target - Population)')
        axs[2, 1].set_ylabel('Dimension')
    else:
        axs[2, 1].text(0.5, 0.5, "No embedding difference data available",
                    horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('agent_embedding_differences.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to agent_embedding_differences.png")


def main():
    """Main analysis function."""
    print(f"Analyzing what makes agent {TARGET_AGENT} different in embedding space...")
    
    # Run all analyses
    visualize_differences()
    
    # Get distance analysis
    distances, _ = analyze_vector_distances()
    if distances:
        print("\nDistance from target agent to closest 5 agents:")
        for i, (agent_id, distance) in enumerate(distances[:5]):
            print(f"{i+1}. Agent {agent_id[-6:]}: {distance:.4f}")
        
        print("\nDistance from target agent to furthest 5 agents:")
        for i, (agent_id, distance) in enumerate(distances[-5:]):
            print(f"{i+1}. Agent {agent_id[-6:]}: {distance:.4f}")
    
    # Get top dimension differences
    top_diffs, _, _, _ = analyze_embedding_differences()
    if top_diffs:
        print("\nTop 10 embedding dimensions with greatest differences:")
        for i, (dim, diff) in enumerate(sorted(top_diffs, key=lambda x: abs(x[1]), reverse=True)[:10]):
            print(f"{i+1}. Dimension {dim}: {diff:.4f}")
    
    # Get state pattern differences
    target_stats, pop_stats = analyze_state_patterns()
    if target_stats and pop_stats:
        print("\nKey state variable differences (Target vs Population):")
        print(f"Resource Level: {target_stats['resource_level']['mean']:.2f} vs {pop_stats['resource_level']['mean']:.2f}")
        print(f"Health: {target_stats['health']['mean']:.2f} vs {pop_stats['health']['mean']:.2f}")
        print(f"Reward: {target_stats['reward']['mean']:.2f} vs {pop_stats['reward']['mean']:.2f}")
        
        # Calculate standard deviation ratios
        print("\nVariability ratios (Target std / Population std):")
        print(f"Resource Variability: {target_stats['resource_level']['std'] / pop_stats['resource_level']['std']:.2f}")
        print(f"Health Variability: {target_stats['health']['std'] / pop_stats['health']['std']:.2f}")
        print(f"Reward Variability: {target_stats['reward']['std'] / pop_stats['reward']['std']:.2f}")
    
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main() 