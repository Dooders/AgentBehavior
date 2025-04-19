#!/usr/bin/env python
"""
Script to generate a detailed validation report about agent 56q2nhmuN2SqH9beAEmVqo's behavior.
"""

import os
import sqlite3
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

# Set data directory
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "simulation.db")
TARGET_AGENT = "56q2nhmuN2SqH9beAEmVqo"

def connect_db():
    """Create a connection to the database with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_agent_basic_info(conn, agent_id):
    """Get basic information about the agent."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT *
        FROM agents
        WHERE agent_id = ?
    """, (agent_id,))
    
    return dict(cursor.fetchone())

def get_agent_lifetime_stats(conn, agent_id):
    """Get lifetime statistics for the agent."""
    cursor = conn.cursor()
    
    # Get total actions and states
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT s.id) as total_states,
            COUNT(DISTINCT a.action_id) as total_actions,
            MIN(s.step_number) as first_step,
            MAX(s.step_number) as last_step,
            AVG(s.resource_level) as avg_resources,
            AVG(s.current_health) as avg_health,
            MAX(s.total_reward) as max_reward
        FROM agents ag
        LEFT JOIN agent_states s ON ag.agent_id = s.agent_id
        LEFT JOIN agent_actions a ON ag.agent_id = a.agent_id
        WHERE ag.agent_id = ?
    """, (agent_id,))
    
    return dict(cursor.fetchone())

def analyze_action_patterns(conn, agent_id):
    """Analyze action patterns and transitions."""
    cursor = conn.cursor()
    
    # Get action sequence
    cursor.execute("""
        SELECT action_type, step_number, resources_before, resources_after, reward
        FROM agent_actions
        WHERE agent_id = ?
        ORDER BY step_number ASC
    """, (agent_id,))
    
    actions = cursor.fetchall()
    
    # Calculate action transitions
    transitions = defaultdict(lambda: defaultdict(int))
    prev_action = None
    
    for action in actions:
        if prev_action:
            transitions[prev_action['action_type']][action['action_type']] += 1
        prev_action = action
    
    # Calculate action outcomes
    outcomes = defaultdict(lambda: {
        'count': 0,
        'avg_resource_gain': 0,
        'avg_reward': 0,
        'success_rate': 0
    })
    
    for action in actions:
        action_type = action['action_type']
        outcomes[action_type]['count'] += 1
        
        # Calculate resource change
        resource_change = action['resources_after'] - action['resources_before'] if action['resources_after'] is not None else 0
        outcomes[action_type]['avg_resource_gain'] += resource_change
        
        # Add reward
        outcomes[action_type]['avg_reward'] += action['reward'] if action['reward'] is not None else 0
    
    # Calculate averages
    for action_type in outcomes:
        count = outcomes[action_type]['count']
        if count > 0:
            outcomes[action_type]['avg_resource_gain'] /= count
            outcomes[action_type]['avg_reward'] /= count
    
    return {
        'transitions': dict(transitions),
        'outcomes': dict(outcomes)
    }

def analyze_resource_management(conn, agent_id):
    """Analyze resource management patterns."""
    cursor = conn.cursor()
    
    # Get resource state changes
    cursor.execute("""
        SELECT 
            step_number,
            resource_level,
            current_health,
            total_reward
        FROM agent_states
        WHERE agent_id = ?
        ORDER BY step_number ASC
    """, (agent_id,))
    
    states = cursor.fetchall()
    
    # Calculate resource metrics
    resource_metrics = {
        'resource_levels': [],
        'health_levels': [],
        'rewards': [],
        'steps': []
    }
    
    for state in states:
        resource_metrics['resource_levels'].append(state['resource_level'])
        resource_metrics['health_levels'].append(state['current_health'])
        resource_metrics['rewards'].append(state['total_reward'])
        resource_metrics['steps'].append(state['step_number'])
    
    # Convert to numpy arrays for analysis
    resource_levels = np.array(resource_metrics['resource_levels'])
    health_levels = np.array(resource_metrics['health_levels'])
    rewards = np.array(resource_metrics['rewards'])
    
    # Calculate statistics
    stats = {
        'resource_stats': {
            'mean': np.mean(resource_levels),
            'std': np.std(resource_levels),
            'min': np.min(resource_levels),
            'max': np.max(resource_levels),
            'periods_below_mean': np.sum(resource_levels < np.mean(resource_levels))
        },
        'health_stats': {
            'mean': np.mean(health_levels),
            'std': np.std(health_levels),
            'min': np.min(health_levels),
            'max': np.max(health_levels)
        },
        'reward_stats': {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'total_gain': rewards[-1] - rewards[0] if len(rewards) > 0 else 0
        }
    }
    
    return stats

def analyze_social_behavior(conn, agent_id):
    """Analyze social interactions with other agents."""
    cursor = conn.cursor()
    
    # Get social interactions
    cursor.execute("""
        SELECT 
            interaction_type,
            subtype,
            outcome,
            resources_transferred,
            initiator_id,
            recipient_id
        FROM social_interactions
        WHERE initiator_id = ? OR recipient_id = ?
    """, (agent_id, agent_id))
    
    interactions = cursor.fetchall()
    
    # Analyze interaction patterns
    interaction_stats = defaultdict(lambda: {
        'initiated': 0,
        'received': 0,
        'successful': 0,
        'resources_given': 0,
        'resources_received': 0
    })
    
    for interaction in interactions:
        int_type = interaction['interaction_type']
        is_initiator = interaction['initiator_id'] == agent_id
        
        if is_initiator:
            interaction_stats[int_type]['initiated'] += 1
        else:
            interaction_stats[int_type]['received'] += 1
            
        if interaction['outcome'] == 'successful':
            interaction_stats[int_type]['successful'] += 1
            
        resources = interaction['resources_transferred'] or 0
        if is_initiator:
            interaction_stats[int_type]['resources_given'] += resources
        else:
            interaction_stats[int_type]['resources_received'] += resources
    
    return dict(interaction_stats)

def generate_report(agent_id=TARGET_AGENT):
    """Generate a comprehensive analysis report."""
    conn = connect_db()
    
    # Collect all analysis data
    basic_info = get_agent_basic_info(conn, agent_id)
    lifetime_stats = get_agent_lifetime_stats(conn, agent_id)
    action_analysis = analyze_action_patterns(conn, agent_id)
    resource_analysis = analyze_resource_management(conn, agent_id)
    social_analysis = analyze_social_behavior(conn, agent_id)
    
    # Generate report
    report = f"""
BEHAVIORAL ANALYSIS REPORT
Agent ID: {agent_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

1. BASIC INFORMATION
-------------------
Type: {basic_info['agent_type']}
Generation: {basic_info['generation']}
Initial Resources: {basic_info['initial_resources']}
Starting Health: {basic_info['starting_health']}

2. LIFETIME STATISTICS
---------------------
Total States: {lifetime_stats['total_states']}
Total Actions: {lifetime_stats['total_actions']}
Lifespan: {lifetime_stats['last_step'] - lifetime_stats['first_step']} steps
Average Resources: {lifetime_stats['avg_resources']:.2f}
Average Health: {lifetime_stats['avg_health']:.2f}
Maximum Reward: {lifetime_stats['max_reward']:.2f}

3. ACTION PATTERNS
-----------------
Most Common Action Transitions:
"""
    
    # Add action transitions
    transitions = action_analysis['transitions']
    for from_action, to_actions in transitions.items():
        report += f"\nFrom {from_action}:\n"
        total = sum(to_actions.values())
        for to_action, count in sorted(to_actions.items(), key=lambda x: x[1], reverse=True):
            report += f"  -> {to_action}: {count/total*100:.1f}%\n"
    
    report += "\nAction Outcomes:\n"
    for action, stats in action_analysis['outcomes'].items():
        report += f"""
{action}:
  Count: {stats['count']}
  Avg Resource Gain: {stats['avg_resource_gain']:.2f}
  Avg Reward: {stats['avg_reward']:.2f}
"""
    
    report += f"""
4. RESOURCE MANAGEMENT
---------------------
Resource Statistics:
  Mean: {resource_analysis['resource_stats']['mean']:.2f}
  Standard Deviation: {resource_analysis['resource_stats']['std']:.2f}
  Minimum: {resource_analysis['resource_stats']['min']:.2f}
  Maximum: {resource_analysis['resource_stats']['max']:.2f}
  Periods Below Mean: {resource_analysis['resource_stats']['periods_below_mean']}

Health Statistics:
  Mean: {resource_analysis['health_stats']['mean']:.2f}
  Standard Deviation: {resource_analysis['health_stats']['std']:.2f}
  Minimum: {resource_analysis['health_stats']['min']:.2f}
  Maximum: {resource_analysis['health_stats']['max']:.2f}

Reward Progress:
  Total Gain: {resource_analysis['reward_stats']['total_gain']:.2f}
  Mean: {resource_analysis['reward_stats']['mean']:.2f}
  Standard Deviation: {resource_analysis['reward_stats']['std']:.2f}

5. SOCIAL BEHAVIOR
-----------------"""
    
    for int_type, stats in social_analysis.items():
        report += f"""
{int_type}:
  Initiated: {stats['initiated']}
  Received: {stats['received']}
  Success Rate: {stats['successful']/(stats['initiated'] + stats['received'])*100:.1f}%
  Resources Given: {stats['resources_given']:.2f}
  Resources Received: {stats['resources_received']:.2f}
"""
    
    report += "\n6. VALIDATION OF FINDINGS\n----------------------\n"
    
    # Validate gathering focus
    gather_actions = action_analysis['outcomes'].get('gather', {}).get('count', 0)
    total_actions = sum(stats['count'] for stats in action_analysis['outcomes'].values())
    gather_percentage = (gather_actions / total_actions * 100) if total_actions > 0 else 0
    
    report += f"""
1. Gathering Focus:
   - Gather actions: {gather_percentage:.1f}% of total actions
   - Average resource gain per gather: {action_analysis['outcomes'].get('gather', {}).get('avg_resource_gain', 0):.2f}
   
2. Resource Management:
   - Maintained health above {resource_analysis['health_stats']['mean']:.1f} on average
   - Resource efficiency (reward/resource): {resource_analysis['reward_stats']['total_gain']/resource_analysis['resource_stats']['mean']:.2f}
   
3. Social Interaction:
   - Share actions: {action_analysis['outcomes'].get('share', {}).get('count', 0)} ({action_analysis['outcomes'].get('share', {}).get('count', 0)/total_actions*100:.1f}% of total)
   - Attack actions: {action_analysis['outcomes'].get('attack', {}).get('count', 0)} ({action_analysis['outcomes'].get('attack', {}).get('count', 0)/total_actions*100:.1f}% of total)
"""
    
    # Save report
    report_file = f"agent_{agent_id}_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"Report generated and saved to {report_file}")
    conn.close()

if __name__ == "__main__":
    generate_report() 