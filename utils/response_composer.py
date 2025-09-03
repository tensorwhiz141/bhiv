"""
Response Composer with Template-based Generation and Grounding

Composes responses using selected templates with grounding verification
and automatic fallback to more extractive templates when needed.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from utils.logger import get_logger
from utils.grounding_verifier import verify_content_grounding, GroundingMetrics
from reinforcement.template_selector import (
    select_response_template, handle_template_grounding_failure,
    update_template_performance
)
from config.template_config import ResponseTemplate
from reinforcement.rl_context import rl_context

logger = get_logger(__name__)

@dataclass
class ComposerTrace:
    """Detailed trace of response composition process."""
    task_id: str
    template_id: str
    grounded: bool
    grounding_score: float
    fallback_used: bool
    original_template_id: Optional[str] = None
    grounding_attempts: int = 1
    composition_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}

class ResponseComposer:
    """
    Composes responses with template selection and grounding verification.
    
    Features:
    - Template-based response generation
    - Grounding verification with automatic fallback
    - Detailed composition tracing
    - RL integration for template selection learning
    """
    
    def __init__(self):
        self.composition_history: List[ComposerTrace] = []
        logger.info("ResponseComposer initialized with grounding verification")
    
    def compose_response(self,
                        task_id: str,
                        input_data: str,
                        context: Dict[str, Any],
                        source_texts: Optional[List[str]] = None,
                        force_template_id: Optional[str] = None) -> Tuple[Dict[str, Any], ComposerTrace]:
        """
        Compose response with template selection and grounding verification.
        
        Args:
            task_id: Unique task identifier
            input_data: Input text or query
            context: Task context for template selection
            source_texts: Source documents for grounding verification
            force_template_id: Optional specific template to use
            
        Returns:
            Tuple of (enhanced_response, composition_trace)
        """
        start_time = datetime.now()
        logger.info(f"Composing response for task {task_id}")
        
        # Select template
        if force_template_id:
            from config.template_config import get_template_by_id
            selected_template = get_template_by_id(force_template_id)
            if not selected_template:
                logger.error(f"Forced template {force_template_id} not found, using default")
                selected_template = select_response_template(context)
        else:
            selected_template = select_response_template(context)
        
        original_template_id = selected_template.template_id
        logger.info(f"Selected template: {selected_template.template_id} for task {task_id}")
        
        # Generate response with template
        response = self._generate_with_template(input_data, selected_template, context)
        
        # Verify grounding if source texts provided
        grounding_metrics = None
        grounded = True  # Default to true if no sources to check
        fallback_used = False
        grounding_attempts = 1
        
        if source_texts:
            grounding_metrics = verify_content_grounding(
                response.get("result", ""), 
                source_texts, 
                selected_template.grounding_requirements
            )
            grounded = grounding_metrics.is_grounded
            
            # Handle grounding failure with fallback
            if not grounded:
                logger.warning(f"Grounding failed for template {selected_template.template_id} "
                              f"(score: {grounding_metrics.overall_score:.3f})")
                
                # Try fallback template
                fallback_template = handle_template_grounding_failure(
                    task_id, selected_template.template_id, grounding_metrics
                )
                
                if fallback_template.template_id != selected_template.template_id:
                    # Generate with fallback template
                    fallback_response = self._generate_with_template(input_data, fallback_template, context)
                    
                    # Verify grounding with fallback template
                    fallback_grounding = verify_content_grounding(
                        fallback_response.get("result", ""),
                        source_texts,
                        fallback_template.grounding_requirements
                    )
                    
                    grounding_attempts = 2
                    if fallback_grounding.is_grounded or fallback_grounding.overall_score > grounding_metrics.overall_score:
                        # Use fallback response
                        response = fallback_response
                        grounding_metrics = fallback_grounding
                        grounded = fallback_grounding.is_grounded
                        selected_template = fallback_template
                        fallback_used = True
                        logger.info(f"Fallback successful with template {fallback_template.template_id}")
                    else:
                        logger.warning(f"Fallback template also failed grounding, using original response")
        
        # Calculate composition time
        composition_time = (datetime.now() - start_time).total_seconds()
        
        # Create composition trace
        trace = ComposerTrace(
            task_id=task_id,
            template_id=selected_template.template_id,
            grounded=grounded,
            grounding_score=grounding_metrics.overall_score if grounding_metrics else 1.0,
            fallback_used=fallback_used,
            original_template_id=original_template_id if fallback_used else None,
            grounding_attempts=grounding_attempts,
            composition_time=composition_time,
            metadata={
                "input_length": len(input_data),
                "response_length": len(response.get("result", "")),
                "source_count": len(source_texts) if source_texts else 0,
                "template_config": asdict(selected_template),
                "grounding_details": asdict(grounding_metrics) if grounding_metrics else None
            }
        )
        
        # Enhance response with trace information
        enhanced_response = {
            **response,
            "template_id": selected_template.template_id,
            "grounded": grounded,
            "grounding_score": grounding_metrics.overall_score if grounding_metrics else 1.0,
            "composition_trace": asdict(trace)
        }
        
        # Log composition trace
        self.composition_history.append(trace)
        self._log_to_rl_context(trace, enhanced_response)
        
        logger.info(f"Response composition completed for task {task_id}: "
                   f"template={selected_template.template_id}, grounded={grounded}, "
                   f"fallback={fallback_used}, time={composition_time:.3f}s")
        
        return enhanced_response, trace
    
    def _generate_with_template(self, 
                               input_data: str, 
                               template: ResponseTemplate, 
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate response using specified template configuration.
        
        This is a simplified version - in practice, this would integrate
        with the actual language model generation pipeline.
        """
        # Template-specific generation parameters
        max_length = template.max_length
        style = template.style
        extractive_ratio = template.extractive_ratio
        
        # Create template-aware prompt
        style_instructions = {
            "conversational": "Respond in a natural, conversational tone.",
            "academic": "Provide a scholarly, well-structured response.",
            "scholarly": "Give a formal, citation-heavy academic response."
        }
        
        style_instruction = style_instructions.get(style, "Provide a clear, informative response.")
        
        if extractive_ratio > 0.6:
            # High extractive ratio - emphasize citations and direct quotes
            length_instruction = f"Keep response under {max_length} words with at least {template.min_citations} citations."
            approach_instruction = "Focus on direct quotes and specific references from source material."
        else:
            # Lower extractive ratio - more generative
            length_instruction = f"Provide a response of approximately {max_length} words."
            approach_instruction = "Synthesize information to provide comprehensive insights."
        
        # Simulate template-aware generation (replace with actual LLM call)
        enhanced_input = f"{style_instruction} {approach_instruction} {length_instruction}\n\nQuery: {input_data}"
        
        # This would be replaced with actual model generation
        # For now, return a structure that simulates template-aware generation
        response = {
            "result": f"[Template {template.template_id}] Response to: {input_data[:50]}...",
            "model": context.get("model", "template_model"),
            "template_style": style,
            "extractive_ratio": extractive_ratio,
            "max_length": max_length,
            "status": 200,
            "keywords": ["template", "generated", style],
            "processing_time": 0.5  # Simulated processing time
        }
        
        return response
    
    def _log_to_rl_context(self, trace: ComposerTrace, response: Dict[str, Any]):
        """Log template selection as RL action."""
        # Log template selection action
        rl_context.log_action(
            task_id=trace.task_id,
            agent="response_composer",
            model="template_selector",
            action="select_template",
            reward=0.0,  # Will be updated when final reward is computed
            metadata={
                "template_id": trace.template_id,
                "grounded": trace.grounded,
                "grounding_score": trace.grounding_score,
                "fallback_used": trace.fallback_used,
                "original_template_id": trace.original_template_id,
                "composition_time": trace.composition_time
            }
        )
        
        # Log grounding verification if applicable
        if trace.grounding_score < 1.0:  # Only if grounding was actually checked
            rl_context.log_action(
                task_id=trace.task_id,
                agent="grounding_verifier",
                model="grounding_model",
                action="verify_grounding",
                reward=trace.grounding_score,  # Use grounding score as immediate reward
                metadata={
                    "grounded": trace.grounded,
                    "grounding_score": trace.grounding_score,
                    "attempts": trace.grounding_attempts
                }
            )
    
    def update_template_performance_from_reward(self, 
                                              task_id: str, 
                                              final_reward: float):
        """Update template performance when final task reward is available."""
        # Find the trace for this task
        trace = None
        for t in reversed(self.composition_history):
            if t.task_id == task_id:
                trace = t
                break
        
        if trace:
            # Convert grounding metrics back to object if needed
            grounding_metrics = None
            if trace.metadata and "grounding_details" in trace.metadata:
                from utils.grounding_verifier import GroundingMetrics
                details = trace.metadata["grounding_details"]
                if details:
                    grounding_metrics = GroundingMetrics(**details)
            
            # Update template performance
            update_template_performance(
                task_id=task_id,
                template_id=trace.template_id,
                reward=final_reward,
                grounding_metrics=grounding_metrics
            )
            
            logger.info(f"Updated template performance for task {task_id}: "
                       f"template={trace.template_id}, reward={final_reward}")
        else:
            logger.warning(f"No composition trace found for task {task_id}")
    
    def get_composition_summary(self) -> Dict[str, Any]:
        """Get summary of composition performance."""
        if not self.composition_history:
            return {"total_compositions": 0}
        
        total = len(self.composition_history)
        grounded_count = sum(1 for t in self.composition_history if t.grounded)
        fallback_count = sum(1 for t in self.composition_history if t.fallback_used)
        avg_grounding_score = sum(t.grounding_score for t in self.composition_history) / total
        avg_composition_time = sum(t.composition_time for t in self.composition_history) / total
        
        # Template usage distribution
        template_usage = {}
        for trace in self.composition_history:
            template_usage[trace.template_id] = template_usage.get(trace.template_id, 0) + 1
        
        return {
            "total_compositions": total,
            "grounding_success_rate": grounded_count / total,
            "fallback_usage_rate": fallback_count / total,
            "avg_grounding_score": avg_grounding_score,
            "avg_composition_time": avg_composition_time,
            "template_usage": template_usage,
            "recent_traces": [asdict(t) for t in self.composition_history[-5:]]
        }

# Global instance
response_composer = ResponseComposer()

def compose_response_with_grounding(task_id: str,
                                  input_data: str,
                                  context: Dict[str, Any],
                                  source_texts: Optional[List[str]] = None,
                                  force_template_id: Optional[str] = None) -> Tuple[Dict[str, Any], ComposerTrace]:
    """Convenience function for response composition."""
    return response_composer.compose_response(task_id, input_data, context, source_texts, force_template_id)

def update_composition_reward(task_id: str, final_reward: float):
    """Convenience function for updating template performance."""
    return response_composer.update_template_performance_from_reward(task_id, final_reward)