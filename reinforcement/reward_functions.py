from typing import Dict, Any
from utils.logger import get_logger
from reinforcement.rl_context import rl_context
from datetime import datetime

logger = get_logger(__name__)

def get_reward_from_output(output: Dict[str, Any], task_id: str) -> float:
    """Compute reward based on output quality, including template and grounding metrics."""
    try:
        result = output.get('result') or output.get('response', '')
        status = output.get('status', 200)  # Use status from API response
        keywords = output.get('keywords') or output.get('sources', [])
        
        # Base reward
        reward = 1.0 if status == 200 else 0.0
        
        # Content quality scoring
        clarity_score = 0.0
        tag_count = 0
        if result:
            word_count = len(result.split())
            clarity_score = min(word_count / 100.0, 1.0)
            reward += clarity_score * 0.3  # Reduced weight to make room for template metrics
        if keywords:
            tag_count = len(keywords)
            reward += tag_count * 0.1
        
        # Template and grounding bonuses
        template_bonus = 0.0
        grounding_bonus = 0.0
        
        # Template selection bonus
        if output.get('template_enhanced'):
            template_bonus = 0.2  # Base bonus for using template
            template_id = output.get('template_id', '')
            
            # Bonus for extractive templates when they work well
            if 'extractive' in template_id and output.get('grounded', False):
                template_bonus += 0.1
                
        # Grounding verification bonus
        if 'grounded' in output:
            grounded = output.get('grounded', False)
            grounding_score = output.get('grounding_score', 0.0)
            
            if grounded:
                grounding_bonus = 0.3 * grounding_score  # Scale bonus by grounding quality
            else:
                # Penalty for poor grounding
                grounding_bonus = -0.2 * (1.0 - grounding_score)
        
        # Fallback penalty
        fallback_penalty = 0.0
        if output.get('composition_trace', {}).get('fallback_used', False):
            fallback_penalty = -0.1  # Small penalty for needing fallback
            
        total_reward = reward + template_bonus + grounding_bonus + fallback_penalty
        
        # Ensure reward is within reasonable bounds
        total_reward = max(0.0, min(total_reward, 3.0))
            
        metrics = {
            "base_reward": reward,
            "clarity_score": clarity_score,
            "tag_count": tag_count,
            "template_bonus": template_bonus,
            "grounding_bonus": grounding_bonus,
            "fallback_penalty": fallback_penalty,
            "total_reward": total_reward,
            "status": status,
            "template_enhanced": output.get('template_enhanced', False),
            "grounded": output.get('grounded'),
            "grounding_score": output.get('grounding_score'),
            "template_id": output.get('template_id')
        }
        
        logger.info(f"Computed reward {total_reward:.3f} for task {task_id} "
                   f"(base={reward:.3f}, template_bonus={template_bonus:.3f}, "
                   f"grounding_bonus={grounding_bonus:.3f})")
        
        rl_context.log_reward(task_id, total_reward, metrics)
        return total_reward
        
    except Exception as e:
        logger.error(f"Error computing reward for task {task_id}: {e}")
        metrics = {"error": str(e), "status": output.get('status', 500)}
        rl_context.log_reward(task_id, 0.0, metrics)
        return 0.0

# from typing import Dict, Any
# from utils.logger import get_logger
# from reinforcement.rl_context import rl_context

# logger = get_logger(__name__)

# def get_reward_from_output(output: Dict[str, Any], task_id: str) -> float:
#     """Compute reward based on output quality."""
#     try:
#         # Handle different output structures (e.g., vedas_agent uses 'response' instead of 'result')
#         result = output.get('result') or output.get('response', '')
#         status = output.get('status', 500)
#         keywords = output.get('keywords') or output.get('sources', [])  # Fallback to sources if keywords missing
        
#         # Base reward: 1.0 for success, 0.0 for failure
#         reward = 1.0 if status == 200 else 0.0
        
#         # Bonus for content quality
#         clarity_score = 0.0
#         tag_count = 0
#         if result:
#             word_count = len(result.split())
#             clarity_score = min(word_count / 100.0, 1.0)  # Reward based on output length
#             reward += clarity_score * 0.5
#         if keywords:
#             tag_count = len(keywords)
#             reward += tag_count * 0.1  # Weight tags
            
#         metrics = {
#             "clarity_score": clarity_score,
#             "tag_count": tag_count,
#             "status": status
#         }
#         logger.info(f"Computed reward {reward} for task {task_id}")
#         rl_context.log_reward(task_id, reward, metrics)
#         return reward
#     except Exception as e:
#         logger.error(f"Error computing reward for task {task_id}: {str(e)}")
#         metrics = {"error": str(e), "status": output.get('status', 500)}
#         rl_context.log_reward(task_id, 0.0, metrics)
#         return 0.0