#!/usr/bin/env python
"""
Outlier Agent Detector

This script identifies agents that are "isolated" or significantly different from the
rest of the agent population by analyzing their embedding vectors and calculating
various isolation metrics.

Usage:
    python outlier_agent_detector.py [--top N] [--metric distance_type] [--output output_file]

Example:
    python outlier_agent_detector.py --top 10 --metric mahalanobis --output outliers.json
"""

import os
import json
import sqlite3
import argparse
import numpy as np
from collections import defaultdict
import pandas as pd
from scipy.spatial.distance import mahalanobis, euclidean, cosine
from scipy.stats import zscore
import faiss
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
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


def extract_agent_state_vectors():
    """Extract state vectors for all agents from the FAISS index."""
    print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
    faiss_index = FaissVectorIndex.load(FAISS_INDEX_PATH)
    print(f"Loaded index with {len(faiss_index.ids)} vectors")
    
    agent_state_vectors = defaultdict(dict)
    
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
    
    return agent_state_vectors, faiss_index


def calculate_agent_average_vectors(agent_state_vectors):
    """Calculate the average embedding vector for each agent."""
    agent_avg_vectors = {}
    for agent_id, states in agent_state_vectors.items():
        if states:
            # Convert values to numpy array and compute mean
            state_vectors = list(states.values())
            agent_avg_vectors[agent_id] = np.mean(state_vectors, axis=0)
    
    return agent_avg_vectors


def calculate_population_statistics(agent_avg_vectors):
    """Calculate population-wide statistics for the embedding vectors."""
    # Stack all average vectors into a matrix
    avg_vectors_matrix = np.stack(list(agent_avg_vectors.values()))
    
    # Calculate population mean and covariance matrix
    pop_mean = np.mean(avg_vectors_matrix, axis=0)
    pop_cov = np.cov(avg_vectors_matrix, rowvar=False)
    
    # Handle potential issues with covariance matrix
    # If matrix is singular, add small regularization
    if np.linalg.det(pop_cov) < 1e-10:
        print("Warning: Covariance matrix is nearly singular, adding regularization")
        pop_cov += np.eye(pop_cov.shape[0]) * 1e-6
    
    return pop_mean, pop_cov, avg_vectors_matrix


def calculate_isolation_metrics(agent_avg_vectors, pop_mean, pop_cov, avg_vectors_matrix):
    """Calculate various isolation metrics for each agent."""
    isolation_metrics = {}
    
    # Calculate inverse of covariance matrix for Mahalanobis distance
    try:
        inv_cov = np.linalg.inv(pop_cov)
    except np.linalg.LinAlgError:
        print("Warning: Covariance matrix inversion failed, using pseudoinverse")
        inv_cov = np.linalg.pinv(pop_cov)
    
    # Calculate population centroid (mean of all agent vectors)
    pop_centroid = pop_mean
    
    for agent_id, avg_vector in agent_avg_vectors.items():
        # Calculate various distance metrics
        
        # 1. Mahalanobis distance (accounts for covariance structure)
        try:
            mahal_dist = mahalanobis(avg_vector, pop_centroid, inv_cov)
        except:
            mahal_dist = float('inf')  # In case of numerical issues
        
        # 2. Euclidean distance to population centroid
        eucl_dist = euclidean(avg_vector, pop_centroid)
        
        # 3. Cosine distance (1 - cosine similarity)
        cos_dist = cosine(avg_vector, pop_centroid)
        
        # 4. Average distance to other agents (mean pairwise distance)
        pairwise_distances = []
        for other_id, other_vec in agent_avg_vectors.items():
            if other_id != agent_id:
                pairwise_distances.append(euclidean(avg_vector, other_vec))
        
        mean_pairwise_dist = np.mean(pairwise_distances) if pairwise_distances else float('inf')
        
        # 5. Nearest neighbor distance (distance to closest agent)
        min_pairwise_dist = min(pairwise_distances) if pairwise_distances else float('inf')
        
        # 6. Local density (number of agents within threshold distance)
        threshold = np.percentile(pairwise_distances, 25) if pairwise_distances else 0
        local_density = sum(1 for d in pairwise_distances if d <= threshold)
        
        # Store all metrics
        isolation_metrics[agent_id] = {
            "mahalanobis_distance": mahal_dist,
            "euclidean_distance": eucl_dist,
            "cosine_distance": cos_dist,
            "mean_pairwise_distance": mean_pairwise_dist,
            "nearest_neighbor_distance": min_pairwise_dist,
            "local_density": local_density
        }
    
    return isolation_metrics


def rank_agents_by_isolation(isolation_metrics, metric="mahalanobis_distance"):
    """Rank agents by their isolation metric values."""
    # For local_density, lower is more isolated
    reverse = metric != "local_density"
    
    # Sort agents by the specified metric
    ranked_agents = sorted(
        isolation_metrics.items(),
        key=lambda x: x[1][metric],
        reverse=reverse
    )
    
    return ranked_agents


def find_outlier_agents(isolation_metrics, metric="mahalanobis_distance", threshold=2.0):
    """Identify outlier agents based on z-scores of isolation metrics."""
    # Extract the specified metric for all agents
    metric_values = [data[metric] for data in isolation_metrics.values()]
    
    # For local_density, we need to invert the values (since lower density = more isolated)
    if metric == "local_density":
        metric_values = [1/max(1, val) for val in metric_values]
    
    # Calculate z-scores
    z_scores = zscore(metric_values)
    
    # Create a mapping from agent_id to z-score
    agent_ids = list(isolation_metrics.keys())
    agent_z_scores = dict(zip(agent_ids, z_scores))
    
    # Find agents with z-scores above threshold (these are the outliers)
    outliers = {agent_id: score for agent_id, score in agent_z_scores.items() if score > threshold}
    
    return outliers


def get_agent_metadata(agent_ids):
    """Retrieve metadata for specified agents from the database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    agent_metadata = {}
    for agent_id in agent_ids:
        cursor.execute(
            """
            SELECT 
                a.agent_id, a.agent_type, a.birth_time, a.death_time, 
                a.generation, a.genome_id,
                COUNT(DISTINCT act.action_type) as action_diversity,
                COUNT(act.action_id) as total_actions,
                AVG(s.resource_level) as avg_resources,
                AVG(s.current_health) as avg_health,
                MAX(s.total_reward) as total_reward
            FROM 
                agents a
            LEFT JOIN
                agent_actions act ON a.agent_id = act.agent_id
            LEFT JOIN
                agent_states s ON a.agent_id = s.agent_id
            WHERE 
                a.agent_id = ?
            GROUP BY
                a.agent_id
            """,
            (agent_id,)
        )
        
        row = cursor.fetchone()
        if row:
            agent_metadata[agent_id] = dict(row)
    
    conn.close()
    return agent_metadata


def visualize_isolation_distribution(isolation_metrics, metric="mahalanobis_distance", top_n=10):
    """Create a visualization of the isolation metric distribution with top outliers highlighted."""
    # Extract metric values
    agent_ids = list(isolation_metrics.keys())
    metric_values = [data[metric] for data in isolation_metrics.values()]
    
    # For density, invert values for consistency (higher = more isolated)
    if metric == "local_density":
        metric_name = "Inverse Local Density"
        metric_values = [1/max(1, val) for val in metric_values]
    else:
        metric_name = metric.replace('_', ' ').title()
    
    # Create a dataframe
    df = pd.DataFrame({
        'agent_id': agent_ids,
        'metric_value': metric_values
    })
    
    # Rank agents
    ranked_agents = rank_agents_by_isolation(isolation_metrics, metric)
    top_agents = [agent_id for agent_id, _ in ranked_agents[:top_n]]
    
    # Mark top agents
    df['is_top'] = df['agent_id'].isin(top_agents)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Distribution plot
    sns.histplot(df['metric_value'], kde=True)
    
    # Mark top agents
    for agent_id in top_agents:
        agent_value = df[df['agent_id'] == agent_id]['metric_value'].values[0]
        plt.axvline(x=agent_value, color='red', linestyle='--', alpha=0.7)
    
    plt.title(f'Distribution of {metric_name} with Top {top_n} Outliers')
    plt.xlabel(metric_name)
    plt.ylabel('Frequency')
    
    # Annotate top agents
    top_df = df[df['is_top']].sort_values('metric_value', ascending=False)
    for i, (_, row) in enumerate(top_df.iterrows()):
        plt.annotate(
            row['agent_id'][-8:],  # Last 8 chars of ID for brevity
            xy=(row['metric_value'], 0),
            xytext=(5, 10 + i*20),  # Offset text to avoid overlap
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2')
        )
    
    plt.tight_layout()
    return plt


def main():
    """Main function to find and report isolated agents."""
    parser = argparse.ArgumentParser(description='Identify isolated/outlier agents.')
    parser.add_argument('--top', type=int, default=10, help='Number of top outliers to report')
    parser.add_argument('--metric', type=str, default='mahalanobis_distance', 
                      choices=['mahalanobis_distance', 'euclidean_distance', 'cosine_distance', 
                              'mean_pairwise_distance', 'nearest_neighbor_distance', 'local_density'],
                      help='Isolation metric to use')
    parser.add_argument('--threshold', type=float, default=2.0, 
                      help='Z-score threshold for outlier detection')
    parser.add_argument('--output', type=str, help='Output file to save results (JSON)')
    parser.add_argument('--visualize', action='store_true', help='Create visualization of isolation distribution')
    parser.add_argument('--viz_output', type=str, help='Output file for visualization')
    
    args = parser.parse_args()
    
    # Extract agent vectors
    print("Extracting agent state vectors...")
    agent_state_vectors, _ = extract_agent_state_vectors()
    
    # Calculate average vectors
    print("Calculating agent average vectors...")
    agent_avg_vectors = calculate_agent_average_vectors(agent_state_vectors)
    
    # Calculate population statistics
    print("Calculating population statistics...")
    pop_mean, pop_cov, avg_vectors_matrix = calculate_population_statistics(agent_avg_vectors)
    
    # Calculate isolation metrics
    print(f"Calculating isolation metrics using {args.metric}...")
    isolation_metrics = calculate_isolation_metrics(
        agent_avg_vectors, pop_mean, pop_cov, avg_vectors_matrix
    )
    
    # Rank agents by isolation
    print(f"Ranking agents by {args.metric}...")
    ranked_agents = rank_agents_by_isolation(isolation_metrics, args.metric)
    
    # Find outlier agents
    print(f"Finding outlier agents (threshold={args.threshold})...")
    outliers = find_outlier_agents(isolation_metrics, args.metric, args.threshold)
    
    # Get metadata for top agents
    top_agent_ids = [agent_id for agent_id, _ in ranked_agents[:args.top]]
    agent_metadata = get_agent_metadata(top_agent_ids)
    
    # Generate report
    print("\n--- Top Isolated Agents ---")
    results = []
    for i, (agent_id, metrics) in enumerate(ranked_agents[:args.top]):
        short_id = agent_id[-8:]  # Use last 8 chars for display
        print(f"{i+1}. Agent {short_id}: {metrics[args.metric]:.4f}")
        
        # Get metadata if available
        metadata = agent_metadata.get(agent_id, {})
        agent_type = metadata.get('agent_type', 'unknown')
        generation = metadata.get('generation', 'unknown')
        reward = metadata.get('total_reward', 'unknown')
        
        # Print additional info
        print(f"   Type: {agent_type}, Generation: {generation}, Reward: {reward}")
        
        # Add to results
        results.append({
            "rank": i+1,
            "agent_id": agent_id,
            "short_id": short_id,
            "isolation_score": metrics[args.metric],
            "isolation_metric": args.metric,
            "metadata": metadata,
            "all_metrics": metrics
        })
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # Generate visualization if requested
    if args.visualize:
        print("\nGenerating isolation distribution visualization...")
        plt = visualize_isolation_distribution(isolation_metrics, args.metric, args.top)
        
        if args.viz_output:
            plt.savefig(args.viz_output, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {args.viz_output}")
        else:
            plt.show()
    
    # Summary of outlier detection
    print(f"\nFound {len(outliers)} statistical outliers (z-score > {args.threshold})")
    if outliers:
        print("Outlier agents:")
        for agent_id, z_score in sorted(outliers.items(), key=lambda x: x[1], reverse=True):
            print(f"  {agent_id[-8:]}: z-score = {z_score:.2f}")


if __name__ == "__main__":
    main() 