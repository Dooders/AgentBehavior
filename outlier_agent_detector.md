# Outlier Agent Detector

## Overview

The Outlier Agent Detector is a tool for identifying agents that are significantly different ("lone weirdos") from the majority of agents in your simulation, without relying on visual methods like t-SNE or PCA. This tool analyzes agent embedding vectors to calculate various isolation metrics and identify statistical outliers.

## Key Features

- **Multiple Isolation Metrics**: Choose from six different distance/isolation metrics
- **Statistical Outlier Detection**: Automatically identify agents with significantly different behavior
- **Detailed Reporting**: Get comprehensive information about the most isolated agents
- **Optional Visualization**: Generate plots of isolation metric distributions (can be disabled)
- **Flexible Output**: Save results to JSON for further analysis

## Isolation Metrics

The tool provides six different metrics to measure isolation:

1. **Mahalanobis Distance** (default): Accounts for correlation structure in the data, measuring distance while considering the shape of the distribution. Best for high-dimensional data with correlations.

2. **Euclidean Distance**: Straight-line distance from an agent's vector to the population centroid. Simple and intuitive, but doesn't account for data distribution.

3. **Cosine Distance**: Measures differences in vector direction rather than magnitude. Useful when the pattern of behavior matters more than its intensity.

4. **Mean Pairwise Distance**: Average distance to all other agents. Identifies agents that are generally far from others.

5. **Nearest Neighbor Distance**: Distance to the closest agent. Identifies agents in sparse regions.

6. **Local Density**: Number of agents within a threshold distance. Low density indicates isolation.

## Usage

```
python outlier_agent_detector.py [--top N] [--metric METRIC] [--threshold Z] [--output FILE] [--visualize] [--viz_output FILE]
```

### Arguments

- `--top N`: Number of top outliers to report (default: 10)
- `--metric METRIC`: Isolation metric to use (default: mahalanobis_distance)
  - Options: mahalanobis_distance, euclidean_distance, cosine_distance, mean_pairwise_distance, nearest_neighbor_distance, local_density
- `--threshold Z`: Z-score threshold for statistical outlier detection (default: 2.0)
- `--output FILE`: JSON file to save results to
- `--visualize`: Generate visualization of the isolation metric distribution
- `--viz_output FILE`: Image file to save visualization to

### Examples

Basic usage with default settings:
```
python outlier_agent_detector.py
```

Find top 5 outliers using cosine distance:
```
python outlier_agent_detector.py --top 5 --metric cosine_distance
```

Save results to JSON file:
```
python outlier_agent_detector.py --output outliers.json
```

Generate and save visualization:
```
python outlier_agent_detector.py --visualize --viz_output isolation_distribution.png
```

## Output Format

The tool outputs:

1. A ranked list of the most isolated agents
2. Metadata about each agent (type, generation, reward, etc.)
3. Statistical outliers (agents with z-scores above threshold)

When saving to JSON, the output includes:
- Rank of each agent
- Full and shortened agent IDs
- Isolation score for the chosen metric
- All available metrics for each agent
- Agent metadata

## Use Cases

- Identifying agents with unusual behavior patterns
- Finding strategic innovations in agent populations
- Detecting agents that have developed unique adaptations
- Exploring behavioral diversity in simulations

## Requirements

- Python 3.6+
- NumPy
- SciPy
- Pandas
- FAISS
- Matplotlib (for visualization)
- Seaborn (for visualization)

## Integration with Analysis Workflow

This tool complements the visual identification methods described in the Agent Behavioral Analysis Framework by providing a quantitative, non-visual approach to finding outlier agents. It can be used as part of the "Statistical screening" step in the Initial Identification phase of the analysis process. 