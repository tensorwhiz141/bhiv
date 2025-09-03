# Reinforcement Learning Layer Documentation

## Overview
The Reinforcement Learning (RL) layer in BHIV Core logs decisions, tracks rewards, and prepares for future learning without disrupting the core pipeline. It acts as a lightweight observer, recording agent and model selections, task outcomes, and computed rewards.

## Components
1. **rl_context.py**:
   - Centralized logging for RL actions, rewards, and task details.
   - Methods: `log_action`, `log_reward`, `log_task`.

2. **agent_selector.py**:
   - Implements RL-based agent selection with random exploration (20% chance).
   - Falls back to keyword/pattern-based selection if RL is disabled or no history exists.
   - Updates agent performance history with rewards.

3. **model_selector.py**:
   - Implements RL-based model selection using bandit logic.
   - Supports exploration and history-based selection.
   - Updates model performance history with rewards.

4. **reward_functions.py**:
   - Computes rewards based on output status, clarity (word count), and tag count.
   - Logs rewards to `rl_context`.

5. **replay_buffer.py**:
   - Stores past task runs in `logs/learning_log.json` for future RL training.
   - Includes input, output, agent, model, reward, and timestamp.

6. **learning_log.json**:
   - JSON file storing task runs for RL analysis.
   - Populated by `replay_buffer`.

## Configuration
- **RL_CONFIG** in `settings.py`:
  - `use_rl`: Enable/disable RL (default: true).
  - `exploration_rate`: Probability of random agent/model selection (default: 0.2).
  - `buffer_file`: Path to `learning_log.json`.

## Usage
1. **Enable RL**:
   - Set `USE_RL=true` in environment variables.
   - Ensure `rl_enabled: true` in `agent_configs.json` for relevant agents.

2. **Task Execution**:
   - Run tasks via `cli_runner.py` or `mcp_bridge.py`.
   - RL layer logs agent/model selections and rewards automatically.

3. **Review Logs**:
   - Check `logs/learning_log.json` for task runs.
   - Query MongoDB `task_logs` collection for detailed logs.

## Future Enhancements
- Transition from logging to active RL training using `replay_buffer`.
- Implement policy-based learning for agent/model selection.
- Enhance reward functions with domain-specific metrics.