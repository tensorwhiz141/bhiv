#!/usr/bin/env python3
"""
Learning Dashboard CLI Tool - Analyze model performance, fallback frequency, and token costs.
"""

import argparse
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from collections import defaultdict, Counter
import statistics

# Simple logger setup for dashboard
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningDashboard:
    """CLI tool for analyzing RL performance and model statistics."""
    
    def __init__(self):
        self.learning_log_path = "logs/learning_log.json"
        self.model_log_path = "logs/model_logs.json"
        self.agent_log_path = "logs/agent_logs.json"
        
    def load_learning_log(self) -> List[Dict[str, Any]]:
        """Load learning log data."""
        try:
            if os.path.exists(self.learning_log_path):
                with open(self.learning_log_path, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading learning log: {e}")
            return []
    
    def load_model_logs(self) -> List[Dict[str, Any]]:
        """Load model selection logs."""
        try:
            if os.path.exists(self.model_log_path):
                with open(self.model_log_path, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.warning(f"Model logs not found: {e}")
            return []
    
    def load_agent_logs(self) -> List[Dict[str, Any]]:
        """Load agent selection logs."""
        try:
            if os.path.exists(self.agent_log_path):
                with open(self.agent_log_path, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.warning(f"Agent logs not found: {e}")
            return []
    
    def analyze_model_performance(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze model performance from learning log."""
        model_stats = defaultdict(lambda: {'rewards': [], 'count': 0, 'total_cost': 0.0, 'total_tokens': 0})
        
        for entry in data:
            model = entry.get('model', 'unknown')
            reward = entry.get('reward', 0.0)
            output = entry.get('output', {})
            
            model_stats[model]['rewards'].append(reward)
            model_stats[model]['count'] += 1
            
            # Extract cost and token information
            if isinstance(output, dict):
                model_stats[model]['total_cost'] += output.get('cost_estimate', 0.0)
                model_stats[model]['total_tokens'] += output.get('tokens_used', 0)
        
        # Calculate statistics
        performance = {}
        for model, stats in model_stats.items():
            if stats['rewards']:
                performance[model] = {
                    'avg_reward': statistics.mean(stats['rewards']),
                    'max_reward': max(stats['rewards']),
                    'min_reward': min(stats['rewards']),
                    'std_reward': statistics.stdev(stats['rewards']) if len(stats['rewards']) > 1 else 0.0,
                    'count': stats['count'],
                    'total_cost': stats['total_cost'],
                    'avg_cost': stats['total_cost'] / stats['count'] if stats['count'] > 0 else 0.0,
                    'total_tokens': stats['total_tokens'],
                    'avg_tokens': stats['total_tokens'] / stats['count'] if stats['count'] > 0 else 0.0
                }
        
        return performance
    
    def analyze_fallback_frequency(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze fallback usage patterns."""
        total_tasks = len(data)
        error_count = 0
        fallback_patterns = Counter()
        
        for entry in data:
            output = entry.get('output', {})
            if isinstance(output, dict) and 'error' in output:
                error_count += 1
                model = entry.get('model', 'unknown')
                fallback_patterns[model] += 1
        
        return {
            'total_tasks': total_tasks,
            'error_count': error_count,
            'error_rate': error_count / total_tasks if total_tasks > 0 else 0.0,
            'fallback_patterns': dict(fallback_patterns)
        }
    
    def create_text_heatmap(self, data: Dict[str, float], title: str, width: int = 50) -> str:
        """Create a simple text-based heatmap."""
        if not data:
            return f"{title}: No data available\n"
        
        max_val = max(data.values())
        min_val = min(data.values())
        range_val = max_val - min_val if max_val != min_val else 1
        
        heatmap = [f"\n{title}:"]
        heatmap.append("=" * len(title))
        
        for key, value in sorted(data.items(), key=lambda x: x[1], reverse=True):
            # Normalize value to 0-1 range
            normalized = (value - min_val) / range_val if range_val > 0 else 0
            bar_length = int(normalized * width)
            bar = "█" * bar_length + "░" * (width - bar_length)
            heatmap.append(f"{key:20} │{bar}│ {value:.3f}")
        
        return "\n".join(heatmap) + "\n"
    
    def display_top_performers(self, performance: Dict[str, Any], metric: str = 'avg_reward', top_n: int = 5) -> str:
        """Display top performing models."""
        if not performance:
            return "No performance data available.\n"
        
        sorted_models = sorted(performance.items(), key=lambda x: x[1].get(metric, 0), reverse=True)
        
        output = [f"\n🏆 Top {top_n} Models by {metric.replace('_', ' ').title()}:"]
        output.append("=" * 50)
        
        for i, (model, stats) in enumerate(sorted_models[:top_n], 1):
            output.append(f"{i}. {model}")
            output.append(f"   {metric.replace('_', ' ').title()}: {stats.get(metric, 0):.3f}")
            output.append(f"   Tasks: {stats.get('count', 0)}")
            output.append(f"   Total Cost: ${stats.get('total_cost', 0):.4f}")
            output.append("")
        
        return "\n".join(output)
    
    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report."""
        learning_data = self.load_learning_log()
        model_logs = self.load_model_logs()
        agent_logs = self.load_agent_logs()
        
        if not learning_data:
            return "No learning data available. Run some tasks first!"
        
        # Analyze performance
        performance = self.analyze_model_performance(learning_data)
        fallback_stats = self.analyze_fallback_frequency(learning_data)
        
        # Generate report
        report = []
        report.append("🤖 BHIV Core Learning Dashboard")
        report.append("=" * 40)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Tasks Analyzed: {len(learning_data)}")
        report.append("")
        
        # Model performance summary
        if performance:
            avg_rewards = {model: stats['avg_reward'] for model, stats in performance.items()}
            report.append(self.create_text_heatmap(avg_rewards, "Model Performance (Average Reward)"))
            
            total_costs = {model: stats['total_cost'] for model, stats in performance.items()}
            report.append(self.create_text_heatmap(total_costs, "Total Cost by Model"))
            
            report.append(self.display_top_performers(performance, 'avg_reward'))
        
        # Fallback analysis
        report.append("\n📊 Fallback Analysis:")
        report.append("=" * 20)
        report.append(f"Error Rate: {fallback_stats['error_rate']:.1%}")
        report.append(f"Total Errors: {fallback_stats['error_count']}/{fallback_stats['total_tasks']}")
        
        if fallback_stats['fallback_patterns']:
            report.append("\nModels with Errors:")
            for model, count in fallback_stats['fallback_patterns'].items():
                report.append(f"  {model}: {count} errors")
        
        # RL Statistics
        report.append(f"\n🧠 RL Statistics:")
        report.append("=" * 15)
        try:
            from reinforcement.model_selector import model_selector
            from reinforcement.agent_selector import agent_selector
            report.append(f"Model Selector History: {len(model_selector.history)} models tracked")
            report.append(f"Agent Selector History: {len(agent_selector.agent_scores)} agents tracked")
        except ImportError as e:
            report.append(f"RL modules not available: {e}")
            report.append("Run tasks with RL enabled to see statistics")
        
        # Recent performance trend
        if len(learning_data) >= 10:
            recent_data = learning_data[-10:]
            recent_rewards = [entry.get('reward', 0) for entry in recent_data]
            report.append(f"\nRecent Performance (last 10 tasks):")
            report.append(f"  Average Reward: {statistics.mean(recent_rewards):.3f}")
            report.append(f"  Trend: {'📈 Improving' if recent_rewards[-1] > recent_rewards[0] else '📉 Declining'}")
        
        return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(
        description="BHIV Core Learning Dashboard - Analyze RL performance and model statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python learning_dashboard.py --summary
  python learning_dashboard.py --models --top 10
  python learning_dashboard.py --fallbacks
  python learning_dashboard.py --export dashboard_report.txt
        """
    )
    
    parser.add_argument("--summary", action="store_true", help="Show comprehensive summary report")
    parser.add_argument("--models", action="store_true", help="Show model performance analysis")
    parser.add_argument("--fallbacks", action="store_true", help="Show fallback frequency analysis")
    parser.add_argument("--top", type=int, default=5, help="Number of top performers to show")
    parser.add_argument("--export", help="Export report to file")
    parser.add_argument("--metric", default="avg_reward", choices=["avg_reward", "total_cost", "count"], 
                       help="Metric for ranking models")
    
    args = parser.parse_args()
    
    dashboard = LearningDashboard()
    
    if args.summary or (not args.models and not args.fallbacks):
        report = dashboard.generate_summary_report()
    else:
        # Custom report based on flags
        learning_data = dashboard.load_learning_log()
        report_parts = []
        
        if args.models:
            performance = dashboard.analyze_model_performance(learning_data)
            report_parts.append(dashboard.display_top_performers(performance, args.metric, args.top))
            
            if performance:
                metric_data = {model: stats[args.metric] for model, stats in performance.items()}
                report_parts.append(dashboard.create_text_heatmap(metric_data, f"Models by {args.metric}"))
        
        if args.fallbacks:
            fallback_stats = dashboard.analyze_fallback_frequency(learning_data)
            report_parts.append(f"Fallback Analysis:\nError Rate: {fallback_stats['error_rate']:.1%}")
        
        report = "\n".join(report_parts)
    
    # Output report
    if args.export:
        try:
            with open(args.export, 'w') as f:
                f.write(report)
            print(f"Report exported to {args.export}")
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
    else:
        print(report)

if __name__ == "__main__":
    main()
