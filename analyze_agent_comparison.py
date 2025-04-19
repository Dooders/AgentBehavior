#!/usr/bin/env python
"""
Script to analyze and compare agent behavior with other agents in the simulation.
"""

import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict
import json

# Target agent ID
TARGET_AGENT = "56q2nhmuN2SqH9beAEmVqo"
DB_PATH = "data/simulation.db"

def get_agent_actions(conn, agent_id):
    """Get all actions for an agent with their outcomes."""
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            action_type,
            COUNT(*) as count,
            AVG(resources_after - resources_before) as avg_resource_change,
            AVG(reward) as avg_reward
        FROM agent_actions
        WHERE agent_id = ?
        GROUP BY action_type
    """, (agent_id,))
    
    actions = {}
    for row in cursor.fetchall():
        actions[row[0]] = {
            'count': row[1],
            'avg_resource_change': row[2],
            'avg_reward': row[3]
        }
    
    return actions

def get_agent_social_stats(conn, agent_id):
    """Get social interaction statistics for an agent."""
    cursor = conn.cursor()
    
    # Get initiated interactions
    cursor.execute("""
        SELECT 
            interaction_type,
            COUNT(*) as count,
            AVG(resources_transferred) as avg_resources,
            COUNT(CASE WHEN outcome = 'successful' THEN 1 END) as successes
        FROM social_interactions
        WHERE initiator_id = ?
        GROUP BY interaction_type
    """, (agent_id,))
    
    initiated = defaultdict(lambda: {'count': 0, 'avg_resources': 0, 'success_rate': 0})
    for row in cursor.fetchall():
        initiated[row[0]] = {
            'count': row[1],
            'avg_resources': row[2] or 0,
            'success_rate': (row[3] / row[1]) * 100 if row[1] > 0 else 0
        }
    
    # Get received interactions
    cursor.execute("""
        SELECT 
            interaction_type,
            COUNT(*) as count,
            AVG(resources_transferred) as avg_resources,
            COUNT(CASE WHEN outcome = 'successful' THEN 1 END) as successes
        FROM social_interactions
        WHERE recipient_id = ?
        GROUP BY interaction_type
    """, (agent_id,))
    
    received = defaultdict(lambda: {'count': 0, 'avg_resources': 0, 'success_rate': 0})
    for row in cursor.fetchall():
        received[row[0]] = {
            'count': row[1],
            'avg_resources': row[2] or 0,
            'success_rate': (row[3] / row[1]) * 100 if row[1] > 0 else 0
        }
    
    return {
        'initiated': dict(initiated),
        'received': dict(received)
    }

def get_agent_resource_stats(conn, agent_id):
    """Get resource management statistics for an agent."""
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            AVG(resource_level) as avg_resources,
            MIN(resource_level) as min_resources,
            MAX(resource_level) as max_resources,
            AVG(current_health) as avg_health,
            COUNT(*) as total_states
        FROM agent_states
        WHERE agent_id = ?
    """, (agent_id,))
    
    row = cursor.fetchone()
    return {
        'avg_resources': row[0],
        'min_resources': row[1],
        'max_resources': row[2],
        'avg_health': row[3],
        'total_states': row[4]
    }

def get_population_stats(conn):
    """Get population-wide statistics for comparison."""
    cursor = conn.cursor()
    
    # Get action distribution across all agents
    cursor.execute("""
        SELECT 
            action_type,
            COUNT(*) as count,
            AVG(resources_after - resources_before) as avg_resource_change,
            AVG(reward) as avg_reward
        FROM agent_actions
        GROUP BY action_type
    """)
    
    actions = {}
    total_actions = 0
    for row in cursor.fetchall():
        actions[row[0]] = {
            'count': row[1],
            'avg_resource_change': row[2],
            'avg_reward': row[3]
        }
        total_actions += row[1]
    
    # Convert counts to percentages
    for action_type in actions:
        actions[action_type]['percentage'] = (actions[action_type]['count'] / total_actions) * 100
    
    # Get average resource and health stats
    cursor.execute("""
        SELECT 
            AVG(resource_level) as avg_resources,
            AVG(current_health) as avg_health
        FROM agent_states
    """)
    
    row = cursor.fetchone()
    resources = {
        'avg_resources': row[0],
        'avg_health': row[1]
    }
    
    # Get social interaction stats
    cursor.execute("""
        SELECT 
            interaction_type,
            COUNT(*) as count,
            AVG(resources_transferred) as avg_resources,
            COUNT(CASE WHEN outcome = 'successful' THEN 1 END) * 100.0 / COUNT(*) as success_rate
        FROM social_interactions
        GROUP BY interaction_type
    """)
    
    social = {}
    for row in cursor.fetchall():
        social[row[0]] = {
            'count': row[1],
            'avg_resources': row[2],
            'success_rate': row[3]
        }
    
    return {
        'actions': actions,
        'resources': resources,
        'social': social
    }

def analyze_agent_behavior():
    """Analyze target agent's behavior compared to population averages."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Get target agent's statistics
    target_actions = get_agent_actions(conn, TARGET_AGENT)
    target_social = get_agent_social_stats(conn, TARGET_AGENT)
    target_resources = get_agent_resource_stats(conn, TARGET_AGENT)
    
    # Get population statistics
    pop_stats = get_population_stats(conn)
    
    # Calculate deviations from population averages
    analysis = {
        'action_patterns': {},
        'resource_management': {},
        'social_behavior': {},
        'anomaly_scores': {}
    }
    
    # Analyze action patterns
    total_target_actions = sum(a['count'] for a in target_actions.values())
    for action_type, stats in target_actions.items():
        if action_type in pop_stats['actions']:
            pop_stats_action = pop_stats['actions'][action_type]
            target_percentage = (stats['count'] / total_target_actions) * 100
            
            analysis['action_patterns'][action_type] = {
                'target_percentage': target_percentage,
                'population_percentage': pop_stats_action['percentage'],
                'deviation_percentage': target_percentage - pop_stats_action['percentage'],
                'reward_ratio': stats['avg_reward'] / pop_stats_action['avg_reward'] if pop_stats_action['avg_reward'] != 0 else float('inf'),
                'resource_efficiency': stats['avg_resource_change'] / pop_stats_action['avg_resource_change'] if pop_stats_action['avg_resource_change'] != 0 else float('inf')
            }
    
    # Analyze resource management
    analysis['resource_management'] = {
        'resource_ratio': target_resources['avg_resources'] / pop_stats['resources']['avg_resources'] if pop_stats['resources']['avg_resources'] != 0 else float('inf'),
        'health_ratio': target_resources['avg_health'] / pop_stats['resources']['avg_health'] if pop_stats['resources']['avg_health'] != 0 else float('inf')
    }
    
    # Analyze social behavior
    for interaction_type, stats in target_social['initiated'].items():
        if interaction_type in pop_stats['social']:
            pop_stats_social = pop_stats['social'][interaction_type]
            analysis['social_behavior'][interaction_type] = {
                'initiation_ratio': stats['count'] / pop_stats_social['count'] if pop_stats_social['count'] != 0 else float('inf'),
                'success_ratio': stats['success_rate'] / pop_stats_social['success_rate'] if pop_stats_social['success_rate'] != 0 else float('inf'),
                'resource_transfer_ratio': stats['avg_resources'] / pop_stats_social['avg_resources'] if pop_stats_social['avg_resources'] != 0 else float('inf')
            }
    
    # Calculate anomaly scores
    action_anomaly = np.mean([abs(stats['deviation_percentage']) for stats in analysis['action_patterns'].values()])
    resource_anomaly = abs(1 - analysis['resource_management']['resource_ratio']) + abs(1 - analysis['resource_management']['health_ratio'])
    
    social_ratios = []
    for stats in analysis['social_behavior'].values():
        social_ratios.extend([
            abs(1 - stats['initiation_ratio']),
            abs(1 - stats['success_ratio']),
            abs(1 - stats['resource_transfer_ratio'])
        ])
    social_anomaly = np.mean(social_ratios) if social_ratios else 0
    
    analysis['anomaly_scores'] = {
        'action_pattern_anomaly': action_anomaly,
        'resource_management_anomaly': resource_anomaly,
        'social_behavior_anomaly': social_anomaly,
        'overall_anomaly': (action_anomaly + resource_anomaly + social_anomaly) / 3
    }
    
    conn.close()
    return analysis

def main():
    """Main function to run the analysis and print results."""
    analysis = analyze_agent_behavior()
    
    print(f"\nBehavioral Analysis Report for Agent {TARGET_AGENT}")
    print("=" * 80)
    
    print("\n1. Action Pattern Analysis")
    print("-" * 30)
    for action_type, stats in analysis['action_patterns'].items():
        print(f"\n{action_type}:")
        print(f"  Target: {stats['target_percentage']:.1f}% vs Population: {stats['population_percentage']:.1f}%")
        print(f"  Deviation: {stats['deviation_percentage']:+.1f}%")
        print(f"  Reward Efficiency: {stats['reward_ratio']:.2f}x population average")
        print(f"  Resource Efficiency: {stats['resource_efficiency']:.2f}x population average")
    
    print("\n2. Resource Management Analysis")
    print("-" * 30)
    resource_stats = analysis['resource_management']
    print(f"Resource Level: {resource_stats['resource_ratio']:.2f}x population average")
    print(f"Health Level: {resource_stats['health_ratio']:.2f}x population average")
    
    print("\n3. Social Behavior Analysis")
    print("-" * 30)
    for interaction_type, stats in analysis['social_behavior'].items():
        print(f"\n{interaction_type}:")
        print(f"  Initiation Rate: {stats['initiation_ratio']:.2f}x population average")
        print(f"  Success Rate: {stats['success_ratio']:.2f}x population average")
        print(f"  Resource Transfer: {stats['resource_transfer_ratio']:.2f}x population average")
    
    print("\n4. Anomaly Scores")
    print("-" * 30)
    scores = analysis['anomaly_scores']
    print(f"Action Pattern Anomaly: {scores['action_pattern_anomaly']:.2f}")
    print(f"Resource Management Anomaly: {scores['resource_management_anomaly']:.2f}")
    print(f"Social Behavior Anomaly: {scores['social_behavior_anomaly']:.2f}")
    print(f"Overall Anomaly Score: {scores['overall_anomaly']:.2f}")
    
    # Save analysis to file
    with open(f'agent_{TARGET_AGENT}_comparison.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nDetailed analysis saved to agent_{TARGET_AGENT}_comparison.json")

if __name__ == "__main__":
    main() 