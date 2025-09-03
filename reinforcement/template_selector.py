"""
Template Selection Policy with Epsilon-Greedy Fallback

Implements intelligent template selection using epsilon-greedy policy
with automatic fallback to more extractive templates when grounding fails.
"""

import random
import math
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from utils.logger import get_logger
from config.template_config import (
    ResponseTemplate, RESPONSE_TEMPLATES, DEFAULT_TEMPLATE_ORDER,
    TEMPLATE_POLICY_CONFIG, get_template_by_id, get_fallback_template
)
from utils.grounding_verifier import GroundingMetrics

logger = get_logger(__name__)

@dataclass
class TemplatePerformance:
    """Performance tracking for template selection."""
    template_id: str
    total_reward: float = 0.0
    count: int = 0
    avg_reward: float = 0.0
    grounding_success_rate: float = 0.0
    grounding_successes: int = 0
    last_used: Optional[str] = None

class TemplateSelector:
    """
    Epsilon-greedy template selector with grounding-based fallback.
    
    Features:
    - Epsilon-greedy exploration/exploitation
    - Automatic fallback to extractive templates on grounding failure
    - Performance tracking and adaptive selection
    - RL integration for template action logging
    """
    
    def __init__(self):
        self.epsilon = TEMPLATE_POLICY_CONFIG["epsilon"]
        self.epsilon_decay = TEMPLATE_POLICY_CONFIG["epsilon_decay"]
        self.min_epsilon = TEMPLATE_POLICY_CONFIG["min_epsilon"]
        self.fallback_threshold = TEMPLATE_POLICY_CONFIG["fallback_threshold"]
        self.memory_size = TEMPLATE_POLICY_CONFIG["memory_size"]
        
        # Performance tracking
        self.template_performance: Dict[str, TemplatePerformance] = {}
        self.total_selections = 0
        self.selection_history: List[Dict[str, Any]] = []
        
        # Initialize performance tracking for all templates
        for template_id in RESPONSE_TEMPLATES.keys():
            self.template_performance[template_id] = TemplatePerformance(template_id=template_id)
        
        # Load historical data if available
        self._load_performance_data()
        
        logger.info(f"TemplateSelector initialized with epsilon={self.epsilon}")
    
    def select_template(self, 
                       task_context: Dict[str, Any],
                       grounding_context: Optional[Dict[str, Any]] = None) -> ResponseTemplate:
        """
        Select template using epsilon-greedy policy.
        
        Args:
            task_context: Context about the current task
            grounding_context: Optional context for grounding requirements
            
        Returns:
            Selected ResponseTemplate
        """
        task_id = task_context.get("task_id", "unknown")
        task_type = task_context.get("input_type", "text")
        
        self.total_selections += 1
        
        # Calculate current exploration rate
        current_epsilon = max(
            self.min_epsilon, 
            self.epsilon * (self.epsilon_decay ** self.total_selections)
        )
        
        selected_template = None
        selection_reason = ""
        
        # Epsilon-greedy selection
        if random.random() < current_epsilon:
            # Exploration: random selection
            template_id = random.choice(list(RESPONSE_TEMPLATES.keys()))
            selected_template = RESPONSE_TEMPLATES[template_id]
            selection_reason = f"exploration (ε={current_epsilon:.3f})"
            logger.info(f"Template exploration: selected {template_id} for task {task_id}")
        else:
            # Exploitation: select best performing template
            best_template_id = self._select_best_template(task_type)
            selected_template = RESPONSE_TEMPLATES[best_template_id]
            selection_reason = "exploitation (UCB)"
            logger.info(f"Template exploitation: selected {best_template_id} for task {task_id}")
        
        # Log selection
        self._log_selection(task_id, selected_template.template_id, selection_reason, task_context)
        
        return selected_template
    
    def handle_grounding_failure(self, 
                                task_id: str,
                                failed_template_id: str, 
                                grounding_metrics: GroundingMetrics) -> ResponseTemplate:
        """
        Handle grounding failure by selecting more extractive fallback template.
        
        Args:
            task_id: Task identifier
            failed_template_id: Template that failed grounding
            grounding_metrics: Grounding verification results
            
        Returns:
            Fallback template with higher extractiveness
        """
        logger.warning(f"Grounding failure for template {failed_template_id} in task {task_id}. "
                      f"Score: {grounding_metrics.overall_score:.3f}")
        
        # Find more extractive template in fallback order
        current_index = DEFAULT_TEMPLATE_ORDER.index(failed_template_id) if failed_template_id in DEFAULT_TEMPLATE_ORDER else -1
        
        # Select next more extractive template
        if current_index < len(DEFAULT_TEMPLATE_ORDER) - 1:
            fallback_template_id = DEFAULT_TEMPLATE_ORDER[current_index + 1]
            fallback_template = RESPONSE_TEMPLATES[fallback_template_id]
            logger.info(f"Fallback to template {fallback_template_id} for task {task_id}")
        else:
            # Use most extractive template as last resort
            fallback_template = get_fallback_template()
            logger.info(f"Last resort fallback to {fallback_template.template_id} for task {task_id}")
        
        # Log fallback action
        self._log_selection(task_id, fallback_template.template_id, 
                          f"grounding_fallback_from_{failed_template_id}", 
                          {"grounding_score": grounding_metrics.overall_score})
        
        return fallback_template
    
    def update_performance(self, 
                          task_id: str,
                          template_id: str, 
                          reward: float,
                          grounding_metrics: Optional[GroundingMetrics] = None):
        """
        Update template performance based on task results.
        
        Args:
            task_id: Task identifier
            template_id: Template used
            reward: Reward received
            grounding_metrics: Optional grounding verification results
        """
        if template_id not in self.template_performance:
            self.template_performance[template_id] = TemplatePerformance(template_id=template_id)
        
        perf = self.template_performance[template_id]
        perf.total_reward += reward
        perf.count += 1
        perf.avg_reward = perf.total_reward / perf.count
        perf.last_used = datetime.now().isoformat()
        
        # Update grounding success tracking
        if grounding_metrics:
            if grounding_metrics.is_grounded:
                perf.grounding_successes += 1
            perf.grounding_success_rate = perf.grounding_successes / perf.count
        
        logger.info(f"Updated template {template_id} performance: "
                   f"avg_reward={perf.avg_reward:.3f}, "
                   f"grounding_rate={perf.grounding_success_rate:.3f}, "
                   f"count={perf.count}")
        
        # Save performance data periodically
        if self.total_selections % 10 == 0:
            self._save_performance_data()
    
    def _select_best_template(self, task_type: str = "text") -> str:
        """Select best template using Upper Confidence Bound (UCB)."""
        if not any(perf.count > 0 for perf in self.template_performance.values()):
            # No history, return default
            return list(RESPONSE_TEMPLATES.keys())[0]
        
        # Calculate UCB scores
        ucb_scores = {}
        for template_id, perf in self.template_performance.items():
            if perf.count == 0:
                ucb_scores[template_id] = float('inf')  # Unexplored templates
            else:
                # UCB formula with grounding success rate bonus
                confidence_interval = math.sqrt(2 * math.log(self.total_selections) / perf.count)
                grounding_bonus = perf.grounding_success_rate * 0.2  # 20% bonus for reliable grounding
                ucb_scores[template_id] = perf.avg_reward + confidence_interval + grounding_bonus
        
        # Select template with highest UCB score
        best_template_id = max(ucb_scores.keys(), key=lambda x: ucb_scores[x])
        ucb_score = ucb_scores[best_template_id]
        
        logger.debug(f"UCB scores: {ucb_scores}")
        logger.info(f"Best template: {best_template_id} (UCB={ucb_score:.3f})")
        
        return best_template_id
    
    def _log_selection(self, task_id: str, template_id: str, reason: str, context: Dict[str, Any]):
        """Log template selection for analysis."""
        selection_entry = {
            "task_id": task_id,
            "template_id": template_id,
            "selection_reason": reason,
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "epsilon": max(self.min_epsilon, self.epsilon * (self.epsilon_decay ** self.total_selections))
        }
        
        self.selection_history.append(selection_entry)
        
        # Maintain memory size limit
        if len(self.selection_history) > self.memory_size:
            self.selection_history = self.selection_history[-self.memory_size:]
        
        logger.debug(f"Logged template selection: {selection_entry}")
    
    def _save_performance_data(self):
        """Save performance data to disk."""
        try:
            os.makedirs("logs", exist_ok=True)
            data = {
                "template_performance": {k: asdict(v) for k, v in self.template_performance.items()},
                "total_selections": self.total_selections,
                "epsilon": self.epsilon,
                "last_updated": datetime.now().isoformat()
            }
            
            with open("logs/template_performance.json", "w") as f:
                json.dump(data, f, indent=2)
                
            logger.debug("Saved template performance data")
        except Exception as e:
            logger.error(f"Failed to save template performance data: {e}")
    
    def _load_performance_data(self):
        """Load performance data from disk."""
        try:
            if os.path.exists("logs/template_performance.json"):
                with open("logs/template_performance.json", "r") as f:
                    data = json.load(f)
                
                # Restore performance tracking
                for template_id, perf_data in data.get("template_performance", {}).items():
                    self.template_performance[template_id] = TemplatePerformance(**perf_data)
                
                self.total_selections = data.get("total_selections", 0)
                
                logger.info(f"Loaded template performance data: {len(self.template_performance)} templates, "
                           f"{self.total_selections} total selections")
        except Exception as e:
            logger.warning(f"Failed to load template performance data: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all templates."""
        return {
            "total_selections": self.total_selections,
            "current_epsilon": max(self.min_epsilon, self.epsilon * (self.epsilon_decay ** self.total_selections)),
            "template_performance": {
                template_id: {
                    "avg_reward": perf.avg_reward,
                    "count": perf.count,
                    "grounding_success_rate": perf.grounding_success_rate,
                    "last_used": perf.last_used
                }
                for template_id, perf in self.template_performance.items()
            }
        }

# Global instance
template_selector = TemplateSelector()

def select_response_template(task_context: Dict[str, Any], 
                           grounding_context: Optional[Dict[str, Any]] = None) -> ResponseTemplate:
    """Convenience function for template selection."""
    return template_selector.select_template(task_context, grounding_context)

def handle_template_grounding_failure(task_id: str,
                                     failed_template_id: str, 
                                     grounding_metrics: GroundingMetrics) -> ResponseTemplate:
    """Convenience function for grounding failure handling."""
    return template_selector.handle_grounding_failure(task_id, failed_template_id, grounding_metrics)

def update_template_performance(task_id: str,
                              template_id: str, 
                              reward: float,
                              grounding_metrics: Optional[GroundingMetrics] = None):
    """Convenience function for performance updates."""
    return template_selector.update_performance(task_id, template_id, reward, grounding_metrics)