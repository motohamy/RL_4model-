from visualizer import MetricsVisualizer
import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_all_results():
    viz = MetricsVisualizer()
    
    # 1. Compare all algorithms
    print("Generating comparison plots...")
    viz.plot_training_curves()
    
    # 2. Generate individual analysis for each agent
    agents = ['dqn', 'ddpg', 'ppo', 'sac']
    summaries = {}
    
    for agent in agents:
        print(f"\nAnalyzing {agent.upper()} performance...")
        try:
            # Plot detailed performance
            viz.plot_agent_performance(agent)
            
            # Get summary statistics
            summary = viz.save_summary(agent)
            summaries[agent] = summary
            
            print(f"{agent.upper()} Summary:")
            for metric, value in summary.items():
                print(f"{metric}: {value:.2f}")
                
        except Exception as e:
            print(f"Could not analyze {agent}: {e}")
    
    # 3. Create comparison table
    df = pd.DataFrame(summaries).T
    print("\nComparison Table:")
    print(df)
    
    # 4. Save results
    df.to_csv("algorithm_comparison.csv")
    
    return df

if __name__ == "__main__":
    results = analyze_all_results()
