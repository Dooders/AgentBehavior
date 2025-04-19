#!/usr/bin/env python
# Agent Divergence Analysis Server
# Detects agents that diverge from population norms

import argparse
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Any, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('divergence_detector')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
VERSION = "1.0.0"
agent_data = None
metrics = {
    "mahalanobis_distance": "Measures distance from the centroid, accounting for correlation between features",
    "euclidean_distance": "Straight-line distance from the centroid in feature space",
    "isolation_forest": "Uses random forests to isolate points based on feature values",
    "local_outlier_factor": "Compares local density of a point to its neighbors",
    "pca_reconstruction_error": "Error when reconstructing data point after PCA dimensionality reduction"
}

class DivergenceDetector:
    """Base class for different divergence detection techniques."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize with the dataset."""
        self.data = data
        self.feature_cols = [col for col in data.columns if col != 'agent_id']
    
    def preprocess(self) -> np.ndarray:
        """Preprocess the data for analysis."""
        return self.data[self.feature_cols].values
    
    def detect_divergent(self, threshold: float = 2.0) -> Dict[str, Any]:
        """Detect divergent agents and return results."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_z_scores(self, distances: np.ndarray) -> np.ndarray:
        """Convert distances to z-scores."""
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        if std_dist == 0:
            return np.zeros_like(distances)
        return (distances - mean_dist) / std_dist

class MahalanobisDistance(DivergenceDetector):
    """Detect divergent agents using Mahalanobis distance."""
    
    def detect_divergent(self, threshold: float = 2.0) -> Dict[str, Any]:
        """Detect divergent agents using Mahalanobis distance."""
        X = self.preprocess()
        
        # Calculate mean vector and covariance matrix
        mean_vec = np.mean(X, axis=0)
        cov_matrix = np.cov(X, rowvar=False)
        
        # Handle singular covariance matrix
        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            # Use pseudoinverse if matrix is singular
            inv_cov = np.linalg.pinv(cov_matrix)
        
        # Calculate Mahalanobis distances
        distances = []
        for i, x in enumerate(X):
            diff = x - mean_vec
            dist = np.sqrt(diff.dot(inv_cov).dot(diff.T))
            distances.append(dist)
        
        distances = np.array(distances)
        z_scores = self.get_z_scores(distances)
        
        # Prepare results
        results = []
        for i, (dist, z) in enumerate(zip(distances, z_scores)):
            results.append({
                "agent_id": self.data.iloc[i]['agent_id'],
                "distance": float(dist),
                "z_score": float(z),
                "is_divergent": bool(z > threshold)
            })
        
        return {
            "metric": "mahalanobis_distance",
            "results": results,
            "threshold": threshold
        }

class EuclideanDistance(DivergenceDetector):
    """Detect divergent agents using Euclidean distance."""
    
    def detect_divergent(self, threshold: float = 2.0) -> Dict[str, Any]:
        """Detect divergent agents using Euclidean distance."""
        X = self.preprocess()
        
        # Calculate mean vector (centroid)
        mean_vec = np.mean(X, axis=0)
        
        # Calculate Euclidean distances
        distances = np.sqrt(np.sum(np.square(X - mean_vec), axis=1))
        z_scores = self.get_z_scores(distances)
        
        # Prepare results
        results = []
        for i, (dist, z) in enumerate(zip(distances, z_scores)):
            results.append({
                "agent_id": self.data.iloc[i]['agent_id'],
                "distance": float(dist),
                "z_score": float(z),
                "is_divergent": bool(z > threshold)
            })
        
        return {
            "metric": "euclidean_distance",
            "results": results,
            "threshold": threshold
        }

class IsolationForestDetector(DivergenceDetector):
    """Detect divergent agents using Isolation Forest algorithm."""
    
    def detect_divergent(self, threshold: float = 2.0) -> Dict[str, Any]:
        """Detect divergent agents using Isolation Forest."""
        X = self.preprocess()
        
        # Train Isolation Forest model
        iso_forest = IsolationForest(contamination='auto', random_state=42)
        iso_forest.fit(X)
        
        # Get anomaly scores (higher is more anomalous)
        # Convert to positive values where higher = more divergent
        scores = -iso_forest.score_samples(X)
        z_scores = self.get_z_scores(scores)
        
        # Prepare results
        results = []
        for i, (score, z) in enumerate(zip(scores, z_scores)):
            results.append({
                "agent_id": self.data.iloc[i]['agent_id'],
                "distance": float(score),
                "z_score": float(z),
                "is_divergent": bool(z > threshold)
            })
        
        return {
            "metric": "isolation_forest",
            "results": results,
            "threshold": threshold
        }

class PCAReconstructionError(DivergenceDetector):
    """Detect divergent agents using PCA reconstruction error."""
    
    def detect_divergent(self, threshold: float = 2.0) -> Dict[str, Any]:
        """Detect divergent agents using PCA reconstruction error."""
        X = self.preprocess()
        
        # Choose number of components to retain 90% of variance
        pca = PCA(n_components=0.9, random_state=42)
        X_pca = pca.fit_transform(X)
        
        # Reconstruct the data
        X_reconstructed = pca.inverse_transform(X_pca)
        
        # Calculate reconstruction error
        errors = np.sum(np.square(X - X_reconstructed), axis=1)
        z_scores = self.get_z_scores(errors)
        
        # Prepare results
        results = []
        for i, (error, z) in enumerate(zip(errors, z_scores)):
            results.append({
                "agent_id": self.data.iloc[i]['agent_id'],
                "distance": float(error),
                "z_score": float(z),
                "is_divergent": bool(z > threshold)
            })
        
        return {
            "metric": "pca_reconstruction_error",
            "results": results,
            "threshold": threshold
        }

class LocalOutlierFactor(DivergenceDetector):
    """Detect divergent agents using Local Outlier Factor."""
    
    def detect_divergent(self, threshold: float = 2.0) -> Dict[str, Any]:
        """Detect divergent agents using Local Outlier Factor (LOF)."""
        X = self.preprocess()
        
        # Calculate LOF scores
        # We implement a simplified version here
        n_neighbors = min(20, len(X) - 1)
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(X)
        
        # Get k-distances and k-distance neighborhood
        distances, indices = nn.kneighbors(X)
        
        # Calculate local reachability density (LRD)
        lrd = np.zeros(len(X))
        for i in range(len(X)):
            # Average reachability distance
            reach_dist = np.max([distances[i], distances[indices[i, :], indices[i, :].T].max(axis=0)], axis=0)
            lrd[i] = n_neighbors / reach_dist.sum()
        
        # Calculate LOF scores
        lof = np.zeros(len(X))
        for i in range(len(X)):
            lof[i] = np.mean([lrd[j] for j in indices[i]]) / lrd[i]
        
        z_scores = self.get_z_scores(lof)
        
        # Prepare results
        results = []
        for i, (score, z) in enumerate(zip(lof, z_scores)):
            results.append({
                "agent_id": self.data.iloc[i]['agent_id'],
                "distance": float(score),
                "z_score": float(z),
                "is_divergent": bool(z > threshold)
            })
        
        return {
            "metric": "local_outlier_factor",
            "results": results,
            "threshold": threshold
        }

def load_data(file_path: str) -> pd.DataFrame:
    """Load and prepare agent data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
        
        # Ensure agent_id column exists
        if 'agent_id' not in df.columns:
            if df.columns[0].lower() in ('id', 'agent', 'agent_id', 'agentid'):
                # Rename the first column to agent_id
                df = df.rename(columns={df.columns[0]: 'agent_id'})
            else:
                # Create sequential agent_id
                df['agent_id'] = [f"agent_{i}" for i in range(len(df))]
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def get_detector(metric: str, data: pd.DataFrame) -> DivergenceDetector:
    """Get the appropriate detector based on the metric name."""
    detectors = {
        "mahalanobis_distance": MahalanobisDistance,
        "euclidean_distance": EuclideanDistance,
        "isolation_forest": IsolationForestDetector,
        "pca_reconstruction_error": PCAReconstructionError,
        "local_outlier_factor": LocalOutlierFactor
    }
    
    if metric not in detectors:
        raise ValueError(f"Unknown metric: {metric}")
    
    return detectors[metric](data)

def find_neighbors(agent_id: str, n_neighbors: int = 5) -> List[Dict[str, Any]]:
    """Find the nearest neighbors for a given agent."""
    global agent_data
    
    # Get agent index
    agent_idx = agent_data[agent_data['agent_id'] == agent_id].index
    if len(agent_idx) == 0:
        return []
    
    agent_idx = agent_idx[0]
    
    # Get feature data
    X = agent_data.drop(columns=['agent_id']).values
    
    # Find nearest neighbors
    n_neighbors = min(n_neighbors + 1, len(X))  # +1 because the agent itself will be included
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X)
    
    distances, indices = nn.kneighbors(X[agent_idx].reshape(1, -1))
    
    # Format results
    neighbors = []
    for i, (dist, idx) in enumerate(zip(distances[0][1:], indices[0][1:])):  # Skip the first one (the agent itself)
        neighbors.append({
            "agent_id": agent_data.iloc[idx]['agent_id'],
            "distance": float(dist)
        })
    
    return neighbors

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "version": VERSION,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get available metrics."""
    return jsonify({
        "metrics": metrics
    })

@app.route('/api/divergent', methods=['GET'])
def get_divergent_agents():
    """Get divergent agents based on a specific metric."""
    global agent_data
    
    if agent_data is None:
        return jsonify({"error": "No data loaded"}), 500
    
    # Parse request parameters
    metric = request.args.get('metric', 'mahalanobis_distance')
    top = int(request.args.get('top', 10))
    threshold = float(request.args.get('threshold', 2.0))
    
    try:
        # Get detector and results
        detector = get_detector(metric, agent_data)
        results = detector.detect_divergent(threshold)
        
        # Sort by z-score (descending)
        sorted_results = sorted(results['results'], key=lambda x: x['z_score'], reverse=True)
        
        # Filter divergent agents
        divergent_agents = [r for r in sorted_results if r['is_divergent']]
        
        return jsonify({
            "metric": metric,
            "threshold": threshold,
            "total_agents": len(agent_data),
            "divergent_count": len(divergent_agents),
            "top_divergent_agents": divergent_agents[:top]
        })
    
    except Exception as e:
        logger.error(f"Error in get_divergent_agents: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/common-divergent', methods=['GET'])
def get_common_divergent():
    """Get agents that are divergent across multiple metrics."""
    global agent_data
    
    if agent_data is None:
        return jsonify({"error": "No data loaded"}), 500
    
    # Parse request parameters
    metrics_list = request.args.get('metrics', None)
    if metrics_list:
        metrics_to_check = metrics_list.split(',')
    else:
        metrics_to_check = list(metrics.keys())
    
    top = int(request.args.get('top', 10))
    threshold = float(request.args.get('threshold', 2.0))
    
    try:
        # Get results for each metric
        divergent_agents_by_metric = {}
        for metric in metrics_to_check:
            detector = get_detector(metric, agent_data)
            results = detector.detect_divergent(threshold)
            
            # Add divergent agents to the dictionary
            divergent_agents = [r['agent_id'] for r in results['results'] if r['is_divergent']]
            divergent_agents_by_metric[metric] = divergent_agents
        
        # Find agents that appear in multiple metrics
        agent_counts = {}
        for metric, agents in divergent_agents_by_metric.items():
            for agent in agents:
                if agent not in agent_counts:
                    agent_counts[agent] = []
                agent_counts[agent].append(metric)
        
        # Filter agents that appear in at least 2 metrics
        common_divergent = {k: v for k, v in agent_counts.items() if len(v) >= 2}
        
        # Sort by number of metrics (descending)
        sorted_common_divergent = dict(sorted(common_divergent.items(), 
                                             key=lambda x: len(x[1]), 
                                             reverse=True))
        
        # Get top results
        top_results = dict(list(sorted_common_divergent.items())[:top])
        
        return jsonify({
            "threshold": threshold,
            "metrics_used": metrics_to_check,
            "common_divergent_count": len(common_divergent),
            "common_divergent_agents": top_results
        })
    
    except Exception as e:
        logger.error(f"Error in get_common_divergent: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/agent/<agent_id>', methods=['GET'])
def get_agent_details(agent_id):
    """Get detailed information about a specific agent."""
    global agent_data
    
    if agent_data is None:
        return jsonify({"error": "No data loaded"}), 500
    
    # Check if agent exists
    agent_row = agent_data[agent_data['agent_id'] == agent_id]
    if len(agent_row) == 0:
        return jsonify({"error": f"Agent {agent_id} not found"}), 404
    
    try:
        # Get basic information
        agent_row = agent_row.iloc[0]
        basic_info = agent_row.to_dict()
        
        # Get divergence metrics
        divergence_metrics = {}
        for metric_name in metrics.keys():
            detector = get_detector(metric_name, agent_data)
            results = detector.detect_divergent()
            
            # Find this agent's result
            agent_result = next((r for r in results['results'] if r['agent_id'] == agent_id), None)
            if agent_result:
                divergence_metrics[metric_name] = {
                    "distance": agent_result['distance'],
                    "z_score": agent_result['z_score'],
                    "is_divergent": agent_result['is_divergent']
                }
        
        # Get nearest neighbors
        neighbors = find_neighbors(agent_id)
        
        return jsonify({
            "agent_id": agent_id,
            "basic_info": basic_info,
            "divergence_metrics": divergence_metrics,
            "nearest_neighbors": neighbors
        })
    
    except Exception as e:
        logger.error(f"Error in get_agent_details: {str(e)}")
        return jsonify({"error": str(e)}), 500

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Agent Divergence Analysis Server")
    parser.add_argument('--data', type=str, required=True, help='Path to agent data CSV file')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    global agent_data
    try:
        agent_data = load_data(args.data)
        app.run(host='0.0.0.0', port=args.port, debug=args.debug)
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main() 