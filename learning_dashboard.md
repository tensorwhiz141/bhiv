# Learning Dashboard Documentation

## Overview

The Learning Dashboard is a comprehensive CLI tool for analyzing BHIV Core's Reinforcement Learning performance, model statistics, and system behavior. It provides insights into model performance, fallback frequency, token costs, and RL decision patterns.

## Features

- **Model Performance Analysis**: Track reward trends, success rates, and cost efficiency
- **Fallback Analytics**: Monitor error patterns and fallback effectiveness
- **Text-based Visualizations**: ASCII charts and heatmaps for terminal display
- **Export Capabilities**: Save reports in multiple formats
- **Real-time Metrics**: Current system performance and trends
- **Historical Analysis**: Long-term performance tracking

## Installation

The dashboard is included with BHIV Core. No additional installation required.

## Usage

### Basic Commands

```bash
# Show comprehensive summary report
python learning_dashboard.py --summary

# Analyze model performance
python learning_dashboard.py --models

# Show fallback frequency analysis
python learning_dashboard.py --fallbacks

# Export report to file
python learning_dashboard.py --summary --export dashboard_report.txt
```

### Advanced Options

```bash
# Show top 10 performers by average reward
python learning_dashboard.py --models --top 10 --metric avg_reward

# Show top performers by total cost
python learning_dashboard.py --models --top 5 --metric total_cost

# Show top performers by task count
python learning_dashboard.py --models --top 8 --metric count

# Combined analysis
python learning_dashboard.py --models --fallbacks --top 15
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--summary` | Show comprehensive summary report | False |
| `--models` | Show model performance analysis | False |
| `--fallbacks` | Show fallback frequency analysis | False |
| `--top N` | Number of top performers to show | 5 |
| `--export FILE` | Export report to file | None |
| `--metric METRIC` | Metric for ranking models | avg_reward |

### Available Metrics

- `avg_reward`: Average reward per task
- `total_cost`: Total estimated cost
- `count`: Number of tasks processed

## Output Examples

### Summary Report

```
🤖 BHIV Core Learning Dashboard
========================================
Generated: 2025-07-22 14:30:15
Total Tasks Analyzed: 156

Model Performance (Average Reward):
===================================
edumentor_agent      │████████████████████████████████████████████████│ 1.425
vedas_agent          │████████████████████████████████████████████████│ 1.398
wellness_agent       │████████████████████████████████████████████████│ 1.387
llama                │████████████████████████████████████████████████│ 1.245

Total Cost by Model:
===================
edumentor_agent      │████████████████████████████████████████████████│ 0.245
vedas_agent          │████████████████████████████████████████████████│ 0.198
wellness_agent       │████████████████████████████████████████████████│ 0.156
llama                │████████████████████████████████████████████████│ 0.089

🏆 Top 5 Models by Average Reward:
==================================================
1. edumentor_agent
   Average Reward: 1.425
   Tasks: 67
   Total Cost: $0.2450

2. vedas_agent
   Average Reward: 1.398
   Tasks: 45
   Total Cost: $0.1980

📊 Fallback Analysis:
====================
Error Rate: 8.3%
Total Errors: 13/156

Models with Errors:
  edumentor_agent: 5 errors
  llama: 4 errors
  vedas_agent: 4 errors

🧠 RL Statistics:
===============
Model Selector History: 4 models tracked
Agent Selector History: 3 agents tracked

Recent Performance (last 10 tasks):
  Average Reward: 1.456
  Trend: 📈 Improving
```

### Model Performance Analysis

```
🏆 Top 5 Models by Average Reward:
==================================================
1. edumentor_agent
   Average Reward: 1.425
   Tasks: 67
   Total Cost: $0.2450

2. vedas_agent
   Average Reward: 1.398
   Tasks: 45
   Total Cost: $0.1980

3. wellness_agent
   Average Reward: 1.387
   Tasks: 32
   Total Cost: $0.1560

4. llama
   Average Reward: 1.245
   Tasks: 12
   Total Cost: $0.0890

Models by avg_reward:
====================
edumentor_agent      │████████████████████████████████████████████████│ 1.425
vedas_agent          │████████████████████████████████████████████████│ 1.398
wellness_agent       │████████████████████████████████████████████████│ 1.387
llama                │████████████████████████████████████████████████│ 1.245
```

### Fallback Analysis

```
Fallback Analysis:
Error Rate: 8.3%
Total Errors: 13/156

Models with Errors:
  edumentor_agent: 5 errors
  llama: 4 errors
  vedas_agent: 4 errors
```

## Data Sources

The dashboard analyzes data from multiple sources:

### Log Files

- `logs/learning_log.json`: Complete task history with rewards
- `logs/model_logs.json`: Model selection decisions (if available)
- `logs/agent_logs.json`: Agent selection decisions (if available)

### Data Structure

```json
{
  "task_id": "uuid-string",
  "input": "user input text",
  "output": {
    "result": "model response",
    "model": "model_name",
    "tokens_used": 150,
    "cost_estimate": 0.0045,
    "status": 200
  },
  "agent": "agent_name",
  "model": "model_name",
  "reward": 1.425,
  "timestamp": "2025-07-22T14:30:15.123456"
}
```

## Metrics Explained

### Model Performance Metrics

- **Average Reward**: Mean reward across all tasks for the model
- **Max/Min Reward**: Highest and lowest rewards achieved
- **Standard Deviation**: Consistency of performance
- **Task Count**: Total number of tasks processed
- **Total Cost**: Cumulative estimated cost
- **Average Cost**: Mean cost per task
- **Total Tokens**: Cumulative token usage
- **Average Tokens**: Mean tokens per task

### Fallback Metrics

- **Error Rate**: Percentage of tasks that resulted in errors
- **Total Errors**: Absolute number of failed tasks
- **Fallback Patterns**: Which models fail most frequently

### RL Statistics

- **Model Selector History**: Number of models tracked by RL
- **Agent Selector History**: Number of agents tracked by RL
- **Recent Performance**: Trend analysis of last N tasks

## Text-based Visualizations

### Heatmaps

The dashboard creates ASCII heatmaps using Unicode block characters:

```
Model Performance (Average Reward):
===================================
edumentor_agent      │████████████████████████████████████████████████│ 1.425
vedas_agent          │████████████████████████████████████████████████│ 1.398
wellness_agent       │████████████████████████████████████████████████│ 1.387
llama                │████████████████████████████████████████████████│ 1.245
```

- `█` represents filled portions
- `░` represents empty portions
- Width is configurable (default: 50 characters)

### Bar Charts

Performance comparisons use proportional bars based on normalized values.

## Export Formats

### Text Format

Human-readable format suitable for reports and documentation.

### JSON Format

Machine-readable format for further analysis:

```json
{
  "generated_at": "2025-07-22T14:30:15",
  "total_tasks": 156,
  "model_performance": {
    "edumentor_agent": {
      "avg_reward": 1.425,
      "count": 67,
      "total_cost": 0.245
    }
  },
  "fallback_analysis": {
    "error_rate": 0.083,
    "total_errors": 13
  }
}
```

## Integration with Other Tools

### CLI Runner Integration

```bash
# Show RL stats after batch processing
python cli_runner.py summarize "Process files" edumentor_agent --batch ./files --rl-stats
```

### Programmatic Access

```python
from learning_dashboard import LearningDashboard

dashboard = LearningDashboard()
performance = dashboard.analyze_model_performance(dashboard.load_learning_log())
fallback_stats = dashboard.analyze_fallback_frequency(dashboard.load_learning_log())
```

## Troubleshooting

### No Data Available

```
No learning data available. Run some tasks first!
```

**Solution**: Execute some tasks through the CLI or API to generate learning data.

### Missing Log Files

```
Model logs not found: [Errno 2] No such file or directory: 'logs/model_logs.json'
```

**Solution**: This is normal if RL logging hasn't been fully implemented. The dashboard will work with available data.

### Empty Visualizations

If heatmaps show no data, check:
1. Learning log file exists and contains valid JSON
2. Tasks have been executed with reward calculation
3. File permissions allow reading log files

## Performance Considerations

- **Large Log Files**: Dashboard loads entire log into memory
- **Processing Time**: Analysis time increases with log size
- **Memory Usage**: Proportional to number of tasks in log

### Optimization Tips

1. **Regular Cleanup**: Archive old logs periodically
2. **Filtered Analysis**: Focus on recent time periods
3. **Batch Processing**: Analyze in chunks for very large datasets

## Future Enhancements

- **Time-based Filtering**: Analyze specific date ranges
- **Interactive Mode**: Real-time dashboard updates
- **Web Interface**: Browser-based dashboard
- **Advanced Analytics**: Trend prediction and anomaly detection
- **Comparative Analysis**: Compare different time periods
- **Custom Metrics**: User-defined performance indicators

## API Reference

### LearningDashboard Class

```python
class LearningDashboard:
    def __init__(self):
        """Initialize dashboard with default log paths."""
    
    def load_learning_log(self) -> List[Dict[str, Any]]:
        """Load learning log data."""
    
    def analyze_model_performance(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze model performance from learning log."""
    
    def analyze_fallback_frequency(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze fallback usage patterns."""
    
    def create_text_heatmap(self, data: Dict[str, float], title: str, width: int = 50) -> str:
        """Create a simple text-based heatmap."""
    
    def display_top_performers(self, performance: Dict[str, Any], metric: str = 'avg_reward', top_n: int = 5) -> str:
        """Display top performing models."""
    
    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report."""
```
