# Agent Divergence Analysis

## Overview

Agent Divergence Analysis is a methodology for identifying agents that diverge from population norms within multi-agent systems. This approach helps detect agents that behave differently from the majority, which may indicate:

- Innovative strategies
- System exploits
- Buggy implementations
- Edge case handling
- Unique optimization approaches

## Key Concepts

### Divergent Agent

A "divergent agent" is an agent whose behavior or characteristics significantly differ from the population average. These agents can be identified through various statistical and machine learning techniques that measure their distance from the population centroid.

### Divergence Metrics

The system supports multiple metrics to detect divergent agents:

1. **Mahalanobis Distance**: Measures distance from the population centroid, accounting for feature correlation
2. **Euclidean Distance**: Straight-line distance in feature space
3. **Isolation Forest**: Detects anomalies using random forests
4. **Local Outlier Factor**: Compares local density of a point to its neighbors
5. **PCA Reconstruction Error**: Measures error when reconstructing data after PCA dimensionality reduction

### Z-Score Threshold

Divergence is typically measured using z-scores, which represent how many standard deviations an agent is from the mean. The default threshold is 2.0 (approximately the 95th percentile).

## Use Cases

### Primary Applications

1. **Quality Assurance**: Identify agents that behave abnormally during testing
2. **Algorithm Development**: Discover novel strategies through divergent agent analysis
3. **System Monitoring**: Detect when agents begin to diverge from expected behavior
4. **Competitive Analysis**: Identify uniquely successful agents in competitive environments
5. **Bug Detection**: Find agents that exploit unintended system behaviors

### Example Scenarios

- **Reinforcement Learning**: Identify agents that have discovered unusual strategies
- **Multi-agent Simulations**: Detect agents that have developed specialized behaviors
- **Trading Systems**: Find trading agents with unique patterns
- **Game AI**: Discover NPCs with unexpected behaviors
- **Robotics**: Identify robots that move or operate differently

## Workflow

1. **Data Collection**: Gather metrics from all agents in your population
2. **Feature Selection**: Choose relevant features that capture agent behavior
3. **Analysis**: Apply divergence detection techniques
4. **Threshold Selection**: Set appropriate z-score thresholds for your domain
5. **Investigation**: Examine divergent agents to understand their unique behaviors
6. **Iteration**: Refine your metrics and thresholds based on findings

## API Reference

The Agent Divergence Analysis system includes a REST API with the following endpoints:

- `GET /api/health`: Check server health
- `GET /api/metrics`: List available divergence metrics
- `GET /api/divergent`: Find divergent agents using specified metrics
- `GET /api/common-divergent`: Find agents that are divergent across multiple metrics
- `GET /api/agent/<agent_id>`: Get detailed information about a specific agent

## Command Line Interface

The system includes a command-line client for interacting with the server:

```
python agent_divergence_client.py health
python agent_divergence_client.py metrics
python agent_divergence_client.py find-divergent --metric mahalanobis_distance --top 5
python agent_divergence_client.py find-common-divergent --metrics mahalanobis_distance,isolation_forest
python agent_divergence_client.py agent-details <agent_id>
```

## Installation

```
pip install -r requirements.txt
```

## Running the Server

```
python agent_divergence_detector.py --data your_agent_data.csv --port 5000
```

## Best Practices

1. **Combine Multiple Metrics**: Agents that appear divergent across multiple metrics are more likely to be genuinely interesting
2. **Context Matters**: The significance of divergence depends on your specific domain
3. **Investigate Thoroughly**: Always examine divergent agents in detail to understand why they differ
4. **Adjust Thresholds**: Different applications may require different sensitivity levels
5. **Normalize Features**: Ensure your features are properly scaled for meaningful comparisons

## Related Concepts

- **Outlier Detection**: Traditional statistical approaches to finding anomalous data points
- **Anomaly Detection**: Machine learning techniques for identifying unusual patterns
- **Novelty Detection**: Finding new or unknown patterns in data
- **Cluster Analysis**: Grouping similar agents to identify distinct strategies or behaviors

## Common Pitfalls

1. **Overfitting**: Being too sensitive to minor variations in agent behavior
2. **Confirmation Bias**: Only looking for divergence that matches preconceptions
3. **Feature Selection**: Using irrelevant features that obscure meaningful divergence
4. **Threshold Selection**: Setting thresholds too high or too low for your application
5. **Sample Size**: Drawing conclusions from too small a population 