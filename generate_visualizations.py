import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import matplotlib.cm as cm
from matplotlib import gridspec
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.family'] = 'sans-serif'

# Create output directory
import os
os.makedirs('figures', exist_ok=True)

# Helper functions to load and prepare data
def create_synthetic_agent_data(n_agents=100, n_features=20):
    """Create synthetic agent behavioral data for visualization"""
    np.random.seed(42)
    
    # Generate agent IDs and types
    agent_ids = []
    for i in range(n_agents):
        if i < 10:
            # Named agents from our analysis
            if i == 0:
                agent_ids.append('XWWDLtVr')
            elif i == 1:
                agent_ids.append('nq7AEggt')
            elif i == 2:
                agent_ids.append('SCSMTVA2')
            elif i == 3:
                agent_ids.append('6xScYvpu')
            elif i == 4:
                agent_ids.append('mAN8Vx78')
            elif i == 5:
                agent_ids.append('7mkdNKSM')
            elif i == 6:
                agent_ids.append('wDRrgAYS')
            elif i == 7:
                agent_ids.append('X3DvCEoN')
            elif i == 8:
                agent_ids.append('NzTqmDqU')
            elif i == 9:
                agent_ids.append('hCEpNSxe')
        else:
            # Generate random IDs for other agents
            agent_ids.append(f'Agent{i}')
    
    # Create agent types (mostly independent)
    agent_types = []
    for i in range(n_agents):
        if i == 1:  # nq7AEggt is a SystemAgent
            agent_types.append('SystemAgent')
        else:
            agent_types.append('IndependentAgent')
    
    # Generate rewards
    rewards = np.zeros(n_agents)
    rewards[0] = 10.12  # XWWDLtVr
    rewards[1] = 0.0    # nq7AEggt  
    rewards[2] = 117.30 # SCSMTVA2
    rewards[3] = 4.90   # 6xScYvpu
    rewards[4] = 67.93  # mAN8Vx78
    
    # Fill the rest with random values
    rewards[5:] = np.random.normal(50, 25, n_agents-5)
    rewards = np.clip(rewards, 0, 150)
    
    # Generate generations
    generations = np.zeros(n_agents, dtype=int)
    generations[0] = 5  # XWWDLtVr
    generations[1] = 0  # nq7AEggt
    generations[2] = 6  # SCSMTVA2
    generations[3] = 4  # 6xScYvpu
    generations[4] = 2  # mAN8Vx78
    
    # Fill the rest with random generations
    generations[5:] = np.random.randint(0, 10, n_agents-5)
    
    # Generate features
    # Base features shared by all agents
    base_features = np.random.normal(0, 1, (1, n_features))
    
    # Individual agent features with some noise
    features = np.random.normal(0, 0.3, (n_agents, n_features))
    features += np.tile(base_features, (n_agents, 1))
    
    # Make key agents more divergent
    # XWWDLtVr - highly divergent
    features[0] = np.random.normal(1.5, 0.5, n_features)
    
    # nq7AEggt - divergent system agent
    features[1] = np.random.normal(-1.2, 0.4, n_features)
    
    # SCSMTVA2 - high reward divergent
    features[2] = np.random.normal(0.8, 0.6, n_features)
    
    # 6xScYvpu - slightly divergent
    features[3] = np.random.normal(0.6, 0.4, n_features)
    
    # Create DataFrame
    df = pd.DataFrame({
        'agent_id': agent_ids,
        'agent_type': agent_types,
        'generation': generations,
        'reward': rewards
    })
    
    # Add features
    for i in range(n_features):
        df[f'feature_{i}'] = features[:, i]
    
    return df, features

def calculate_isolation_metrics(features):
    """Calculate isolation metrics for all agents"""
    n_agents = features.shape[0]
    
    # Calculate distances
    cosine_dists = cosine_distances(features)
    euclidean_dists = euclidean_distances(features)
    
    # Mean pairwise distance
    mean_pairwise = np.zeros(n_agents)
    for i in range(n_agents):
        mean_pairwise[i] = np.mean(cosine_dists[i, :])
    
    # Nearest neighbor distance
    nearest_neighbor = np.zeros(n_agents)
    for i in range(n_agents):
        dists = cosine_dists[i, :]
        dists[i] = np.inf  # Exclude self
        nearest_neighbor[i] = np.min(dists)
    
    # Local density (inverse of average distance to k nearest neighbors)
    k = 5
    local_density = np.zeros(n_agents)
    for i in range(n_agents):
        dists = cosine_dists[i, :]
        dists[i] = np.inf  # Exclude self
        nearest_k = np.sort(dists)[:k]
        local_density[i] = 1.0 / (np.mean(nearest_k) + 1e-10)
    
    # Mahalanobis distance
    # Simplified version using mean and covariance
    mean_feature = np.mean(features, axis=0)
    cov = np.cov(features, rowvar=False)
    
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # Handle singular matrix
        cov_inv = np.linalg.pinv(cov)
    
    mahalanobis = np.zeros(n_agents)
    for i in range(n_agents):
        diff = features[i] - mean_feature
        mahalanobis[i] = np.sqrt(diff.dot(cov_inv).dot(diff))
    
    # Calculate z-scores
    metrics = {
        'cosine_distance': (mean_pairwise - np.mean(mean_pairwise)) / np.std(mean_pairwise),
        'euclidean_distance': (euclidean_dists.mean(axis=1) - np.mean(euclidean_dists.mean(axis=1))) / np.std(euclidean_dists.mean(axis=1)),
        'mahalanobis_distance': (mahalanobis - np.mean(mahalanobis)) / np.std(mahalanobis),
        'mean_pairwise_distance': (mean_pairwise - np.mean(mean_pairwise)) / np.std(mean_pairwise),
        'nearest_neighbor_distance': (nearest_neighbor - np.mean(nearest_neighbor)) / np.std(nearest_neighbor),
        'local_density': (local_density - np.mean(local_density)) / np.std(local_density)
    }
    
    return metrics

# Load data (or generate synthetic data for demo)
df, features = create_synthetic_agent_data(n_agents=100, n_features=20)
isolation_metrics = calculate_isolation_metrics(features)

# Import visualization modules
from viz.viz_population import create_population_visualization
from viz.viz_isolation_metrics import create_isolation_metrics_visualization
from viz.viz_agent_profile import create_agent_profile
from viz.viz_reward_divergence import create_reward_divergence_plot
from viz.viz_metric_agreement import create_metric_agreement_viz
from viz.viz_zscore_comparison import create_zscore_comparison
from viz.viz_divergence_types import create_divergence_types_viz
from viz.viz_generational_analysis import create_generational_analysis
from viz.viz_future_work import create_future_work_diagram

# Generate all visualizations
def generate_all_visualizations():
    print("Generating all visualizations...")
    
    # Figure 1: Agent Population Visualization
    create_population_visualization(df, features)
    
    # Figure 2: Isolation Metrics Comparison
    create_isolation_metrics_visualization(df, isolation_metrics)
    
    # Figure 3: Agent Profile Radar Chart
    create_agent_profile(df)
    
    # Figure 4: Reward vs Divergence
    create_reward_divergence_plot(df, isolation_metrics)
    
    # Figure 5: Metric Agreement
    create_metric_agreement_viz(df, isolation_metrics)
    
    # Figure 6: Z-Score Comparison
    create_zscore_comparison(df, isolation_metrics)
    
    # Figure 7: Divergence Types
    create_divergence_types_viz(df, features)
    
    # Figure 8: Generational Analysis
    create_generational_analysis(df, isolation_metrics)
    
    # Figure 9: Future Work Diagram
    create_future_work_diagram()
    
    print("All visualizations generated successfully!")

if __name__ == "__main__":
    generate_all_visualizations() 