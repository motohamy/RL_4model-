import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from visualizer import MetricsVisualizer

class StatisticalAnalyzer:
    def __init__(self, log_dir='logs'):
        self.viz = MetricsVisualizer(log_dir)
        
    def perform_analysis(self, agents=['dqn', 'ddpg', 'ppo', 'sac']):
        """Perform comprehensive statistical analysis"""
        results = {}
        
        # Collect rewards for each agent
        for agent in agents:
            try:
                metrics = self.viz.load_metrics(agent)
                results[agent] = metrics['rewards']
            except Exception as e:
                print(f"Could not load metrics for {agent}: {e}")
        
        # Statistical Tests
        stats_results = {
            'basic_stats': self._compute_basic_stats(results),
            'anova_results': self._perform_anova(results),
            'pairwise_tests': self._perform_pairwise_tests(results)
        }
        
        # Visualizations
        self._plot_statistical_comparisons(results)
        
        return stats_results
    
    def _compute_basic_stats(self, results):
        """Compute basic statistics for each agent"""
        stats_dict = {}
        for agent, rewards in results.items():
            stats_dict[agent] = {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'median': np.median(rewards),
                'q25': np.percentile(rewards, 25),
                'q75': np.percentile(rewards, 75),
                'min': np.min(rewards),
                'max': np.max(rewards)
            }
        return pd.DataFrame(stats_dict).T
    
    def _perform_anova(self, results):
        """Perform one-way ANOVA test"""
        groups = [rewards for rewards in results.values()]
        f_stat, p_value = stats.f_oneway(*groups)
        return {'f_statistic': f_stat, 'p_value': p_value}
    
    def _perform_pairwise_tests(self, results):
        """Perform pairwise t-tests between all agents"""
        agents = list(results.keys())
        n_agents = len(agents)
        pairwise_results = pd.DataFrame(
            index=agents,
            columns=agents,
            dtype=float
        )
        
        for i in range(n_agents):
            for j in range(i+1, n_agents):
                t_stat, p_value = stats.ttest_ind(
                    results[agents[i]],
                    results[agents[j]]
                )
                pairwise_results.iloc[i,j] = p_value
                pairwise_results.iloc[j,i] = p_value
        
        return pairwise_results
    
    def _plot_statistical_comparisons(self, results):
        """Create statistical visualization plots"""
        plt.figure(figsize=(15, 10))
        
        # Box plot
        plt.subplot(2, 2, 1)
        data = []
        labels = []
        for agent, rewards in results.items():
            data.extend(rewards)
            labels.extend([agent.upper()] * len(rewards))
        
        df = pd.DataFrame({'Agent': labels, 'Reward': data})
        sns.boxplot(x='Agent', y='Reward', data=df)
        plt.title('Reward Distribution by Agent')
        
        # Violin plot
        plt.subplot(2, 2, 2)
        sns.violinplot(x='Agent', y='Reward', data=df)
        plt.title('Reward Density by Agent')
        
        # Learning curve comparison
        plt.subplot(2, 2, 3)
        for agent, rewards in results.items():
            plt.plot(
                np.convolve(rewards, np.ones(10)/10, mode='valid'),
                label=agent.upper()
            )
        plt.title('Smoothed Learning Curves')
        plt.xlabel('Episode')
        plt.ylabel('Reward (10-episode moving average)')
        plt.legend()
        
        # Success rate comparison
        plt.subplot(2, 2, 4)
        success_rates = {}
        for agent, rewards in results.items():
            success_rates[agent] = np.mean(np.array(rewards) > np.median(rewards))
        
        plt.bar(success_rates.keys(), success_rates.values())
        plt.title('Success Rate by Agent')
        plt.ylabel('Success Rate')
        
        plt.tight_layout()
        plt.savefig('statistical_comparison.png')
        plt.close()

def main():
    analyzer = StatisticalAnalyzer()
    results = analyzer.perform_analysis()
    
    # Print results
    print("\nBasic Statistics:")
    print(results['basic_stats'])
    
    print("\nANOVA Results:")
    print(f"F-statistic: {results['anova_results']['f_statistic']:.4f}")
    print(f"p-value: {results['anova_results']['p_value']:.4f}")
    
    print("\nPairwise Test Results (p-values):")
    print(results['pairwise_tests'])
    
    # Save results to file
    with pd.ExcelWriter('statistical_analysis.xlsx') as writer:
        results['basic_stats'].to_excel(writer, sheet_name='Basic Stats')
        results['pairwise_tests'].to_excel(writer, sheet_name='Pairwise Tests')
        
        anova_df = pd.DataFrame([results['anova_results']])
        anova_df.to_excel(writer, sheet_name='ANOVA Results')

if __name__ == "__main__":
    main()