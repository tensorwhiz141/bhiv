"""
Response Composer Module

Handles response composition with template logic, grounding enforcement,
and policy-based template selection with fallback mechanisms.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from utils.logger import get_logger
from config.template_config import ResponseTemplate, get_template_by_id, get_default_template
from utils.grounding_verifier import GroundingVerifier, GroundingMetrics
from reinforcement.template_selector import TemplateSelector
from reinforcement.rl_context import RLContext

logger = get_logger(__name__)

@dataclass
class CompositionResult:
    """Result of response composition."""
    response_text: str
    template_id: str
    grounded: bool
    grounding_score: float
    composition_trace: Dict[str, Any]
    processing_time: float
    fallback_applied: bool = False
    original_template_id: Optional[str] = None

@dataclass
class CompositionContext:
    """Context for response composition."""
    task_id: str
    input_text: str
    input_type: str
    top_chunks: List[Dict[str, Any]]
    agent_type: str
    model_name: str
    user_context: Optional[Dict[str, Any]] = None
    explicit_template_id: Optional[str] = None

class ResponseComposer:
    """
    Main response composer that orchestrates template selection,
    content generation, grounding verification, and fallback handling.
    """
    
    def __init__(self):
        self.grounding_verifier = GroundingVerifier()
        self.template_selector = TemplateSelector()
        self.rl_context = RLContext()
        self.composition_stats = {
            "total_compositions": 0,
            "successful_groundings": 0,
            "fallback_applications": 0,
            "template_usage": {}
        }
        
        logger.info("ResponseComposer initialized")
    
    async def compose_response(self, context: CompositionContext) -> CompositionResult:
        """
        Compose response with template selection, grounding enforcement, and fallback.
        
        Args:
            context: Composition context with input and requirements
            
        Returns:
            CompositionResult with response and metadata
        """
        start_time = time.time()
        self.composition_stats["total_compositions"] += 1
        
        logger.info(f"Starting composition for task {context.task_id}")
        
        try:
            # Step 1: Template Selection
            template = await self._select_template(context)
            original_template_id = template.template_id
            
            # Step 2: Generate response with template
            response_text = await self._generate_with_template(template, context)
            
            # Step 3: Grounding verification
            grounding_metrics = await self._verify_grounding(
                response_text, context.top_chunks, template
            )
            
            # Step 4: Handle grounding failure with fallback
            fallback_applied = False
            if not grounding_metrics.is_grounded:
                logger.warning(f"Grounding failed for task {context.task_id}. "
                             f"Score: {grounding_metrics.overall_score:.3f}")
                
                # Apply fallback template
                fallback_template = self.template_selector.handle_grounding_failure(
                    context.task_id, template.template_id, grounding_metrics
                )
                
                if fallback_template.template_id != template.template_id:
                    # Re-generate with fallback template
                    template = fallback_template
                    response_text = await self._generate_with_template(template, context)
                    
                    # Re-verify grounding
                    grounding_metrics = await self._verify_grounding(
                        response_text, context.top_chunks, template
                    )
                    
                    fallback_applied = True
                    self.composition_stats["fallback_applications"] += 1
                    logger.info(f"Fallback applied for task {context.task_id}. "
                              f"New score: {grounding_metrics.overall_score:.3f}")
            
            # Step 5: Update statistics and RL
            if grounding_metrics.is_grounded:
                self.composition_stats["successful_groundings"] += 1
            
            self._update_template_stats(template.template_id)
            await self._log_rl_action(context, template, grounding_metrics)
            
            # Step 6: Build result
            processing_time = time.time() - start_time
            composition_trace = self._build_composition_trace(
                context, template, grounding_metrics, processing_time, fallback_applied
            )
            
            result = CompositionResult(
                response_text=response_text,
                template_id=template.template_id,
                grounded=grounding_metrics.is_grounded,
                grounding_score=grounding_metrics.overall_score,
                composition_trace=composition_trace,
                processing_time=processing_time,
                fallback_applied=fallback_applied,
                original_template_id=original_template_id if fallback_applied else None
            )
            
            logger.info(f"Composition completed for task {context.task_id}. "
                       f"Template: {template.template_id}, "
                       f"Grounded: {grounding_metrics.is_grounded}, "
                       f"Time: {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Composition failed for task {context.task_id}: {e}")
            # Return fallback response
            processing_time = time.time() - start_time
            return CompositionResult(
                response_text=f"I apologize, but I encountered an error processing your request. Please try again.",
                template_id="error_fallback",
                grounded=False,
                grounding_score=0.0,
                composition_trace={"error": str(e)},
                processing_time=processing_time,
                fallback_applied=True
            )
    
    async def _select_template(self, context: CompositionContext) -> ResponseTemplate:
        """Select appropriate template based on context and policy."""
        # Check for explicit template request
        if context.explicit_template_id:
            template = get_template_by_id(context.explicit_template_id)
            if template:
                logger.info(f"Using explicit template {context.explicit_template_id} for task {context.task_id}")
                return template
            else:
                logger.warning(f"Explicit template {context.explicit_template_id} not found, using policy")
        
        # Use template selector policy
        task_context = {
            "task_id": context.task_id,
            "input_type": context.input_type,
            "agent_type": context.agent_type,
            "chunks_available": len(context.top_chunks),
            "input_length": len(context.input_text.split()) if context.input_text else 0
        }
        
        grounding_context = {
            "source_count": len(context.top_chunks),
            "source_quality": self._assess_source_quality(context.top_chunks)
        }
        
        template = self.template_selector.select_template(task_context, grounding_context)
        logger.info(f"Policy selected template {template.template_id} for task {context.task_id}")
        
        return template
    
    async def _generate_with_template(self, template: ResponseTemplate, context: CompositionContext) -> str:
        """Generate response using specified template configuration."""
        logger.debug(f"Generating response with template {template.template_id}")
        
        # Prepare source content from top chunks
        source_content = []
        for chunk in context.top_chunks:
            chunk_text = chunk.get("text", "")
            if chunk_text:
                source_content.append(chunk_text)
        
        # Apply template-specific generation logic
        if template.extractive_ratio >= 0.7:
            # Heavily extractive: primarily use source content
            response = await self._generate_extractive_response(
                context.input_text, source_content, template
            )
        elif template.extractive_ratio >= 0.4:
            # Balanced: mix extractive and generative
            response = await self._generate_balanced_response(
                context.input_text, source_content, template
            )
        else:
            # Generative: use sources for grounding but generate freely
            response = await self._generate_generative_response(
                context.input_text, source_content, template
            )
        
        # Apply length constraints
        response = self._apply_length_constraints(response, template.max_length)
        
        # Add citations based on template requirements
        response = self._add_citations(response, context.top_chunks, template.min_citations)
        
        return response
    
    async def _generate_extractive_response(self, 
                                          input_text: str, 
                                          source_content: List[str], 
                                          template: ResponseTemplate) -> str:
        """Generate heavily extractive response using source content."""
        if not source_content:
            return "I don't have sufficient information to provide a detailed response."
        
        # Select most relevant sentences from sources
        relevant_sentences = []
        for source in source_content:
            sentences = source.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # Filter short fragments
                    # Simple relevance check based on keyword overlap
                    input_words = set(input_text.lower().split())
                    sentence_words = set(sentence.lower().split())
                    overlap = len(input_words.intersection(sentence_words))
                    if overlap > 0:
                        relevant_sentences.append((sentence, overlap))
        
        # Sort by relevance and take top sentences
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in relevant_sentences[:3]]
        
        if not top_sentences:
            return "Based on the available information, I cannot provide a specific answer."
        
        # Combine with minimal connecting text
        response = "Based on the sources: " + ". ".join(top_sentences) + "."
        return response
    
    async def _generate_balanced_response(self, 
                                        input_text: str, 
                                        source_content: List[str], 
                                        template: ResponseTemplate) -> str:
        """Generate balanced response mixing extractive and generative content."""
        if not source_content:
            return self._generate_fallback_response(input_text)
        
        # Extract key facts from sources
        key_facts = []
        for source in source_content:
            # Simple fact extraction (in production, use more sophisticated NLP)
            sentences = [s.strip() for s in source.split('.') if len(s.strip()) > 15]
            key_facts.extend(sentences[:2])  # Take first 2 sentences from each source
        
        # Generate contextual response
        if "yoga" in input_text.lower():
            intro = "Yoga is a comprehensive practice that encompasses physical, mental, and spiritual dimensions."
        elif "meditation" in input_text.lower():
            intro = "Meditation is a fundamental practice for developing mental clarity and inner peace."
        elif "ayurveda" in input_text.lower():
            intro = "Ayurveda is the traditional Indian system of medicine focusing on holistic health."
        else:
            intro = "According to traditional teachings:"
        
        # Combine intro with extracted facts
        if key_facts:
            facts_text = " ".join(key_facts[:2])  # Use top 2 facts
            response = f"{intro} {facts_text}"
        else:
            response = intro
        
        return response
    
    async def _generate_generative_response(self, 
                                          input_text: str, 
                                          source_content: List[str], 
                                          template: ResponseTemplate) -> str:
        """Generate creative response using sources for grounding."""
        # For generative responses, we provide more interpretive and explanatory content
        # while ensuring it's still grounded in the source material
        
        if "yoga" in input_text.lower():
            base_response = ("Yoga represents a profound journey of self-discovery and transformation. "
                           "Through the practice of asanas (physical postures), pranayama (breath control), "
                           "and dhyana (meditation), practitioners develop not only physical strength and "
                           "flexibility but also mental clarity and spiritual awareness.")
        elif "meditation" in input_text.lower():
            base_response = ("Meditation is a transformative practice that allows us to cultivate inner peace "
                           "and awareness. Through regular practice, we learn to observe our thoughts without "
                           "attachment, developing equanimity and insight that enriches all aspects of life.")
        elif "karma" in input_text.lower():
            base_response = ("The law of karma teaches us that every action has consequences, creating a web "
                           "of cause and effect that shapes our experiences. Understanding karma empowers us "
                           "to make conscious choices that lead to positive outcomes and spiritual growth.")
        elif "ayurveda" in input_text.lower():
            base_response = ("Ayurveda offers a holistic approach to health that recognizes the unique "
                           "constitution of each individual. By balancing the three doshas - Vata, Pitta, "
                           "and Kapha - we can maintain optimal health and prevent disease naturally.")
        else:
            # Generic philosophical response
            base_response = ("The ancient wisdom traditions offer timeless insights into the nature of "
                           "reality and human potential. Through understanding and application of these "
                           "teachings, we can develop greater wisdom and compassion in our daily lives.")
        
        # Enhance with source-specific details if available
        if source_content:
            # Extract key concepts from sources to enhance the response
            source_text = " ".join(source_content)
            if "dharma" in source_text.lower():
                base_response += " This aligns with the concept of dharma - righteous living."
            if "guru" in source_text.lower():
                base_response += " The guidance of a qualified teacher (guru) can illuminate this path."
        
        return base_response
    
    def _generate_fallback_response(self, input_text: str) -> str:
        """Generate basic fallback response when no sources available."""
        return ("I understand you're asking about this topic. While I don't have specific "
                "source material available right now, I'd be happy to help you explore "
                "this subject further with more information.")
    
    async def _verify_grounding(self, 
                               response_text: str, 
                               top_chunks: List[Dict[str, Any]], 
                               template: ResponseTemplate) -> GroundingMetrics:
        """Verify response grounding against source chunks."""
        # Extract source texts from chunks
        source_texts = []
        for chunk in top_chunks:
            if "text" in chunk:
                source_texts.append(chunk["text"])
        
        # Perform grounding verification
        grounding_metrics = self.grounding_verifier.verify_grounding(
            response_text, 
            source_texts, 
            template.grounding_requirements
        )
        
        # Apply token overlap enforcement
        if source_texts:
            token_overlap = self._compute_token_overlap(response_text, source_texts)
            grounding_metrics.details["token_overlap"] = token_overlap
            
            # Adjust grounding score based on token overlap
            if token_overlap < 0.3:  # Low overlap threshold
                grounding_metrics.overall_score *= 0.8  # Penalize low overlap
                grounding_metrics.is_grounded = (
                    grounding_metrics.overall_score >= template.grounding_requirements.get("min_source_overlap", 0.4)
                )
        
        return grounding_metrics
    
    def _compute_token_overlap(self, response_text: str, source_texts: List[str]) -> float:
        """Compute token overlap between response and sources."""
        response_tokens = set(response_text.lower().split())
        source_tokens = set()
        
        for source in source_texts:
            source_tokens.update(source.lower().split())
        
        if not response_tokens:
            return 0.0
        
        overlap_count = len(response_tokens.intersection(source_tokens))
        return overlap_count / len(response_tokens)
    
    def _apply_length_constraints(self, response: str, max_length: int) -> str:
        """Apply template length constraints."""
        words = response.split()
        if len(words) > max_length:
            truncated = " ".join(words[:max_length])
            # Try to end at a sentence boundary
            last_period = truncated.rfind('.')
            if last_period > len(truncated) * 0.8:  # If period is near the end
                truncated = truncated[:last_period + 1]
            else:
                truncated += "..."
            return truncated
        return response
    
    def _add_citations(self, response: str, top_chunks: List[Dict[str, Any]], min_citations: int) -> str:
        """Add citations to response based on template requirements."""
        if min_citations <= 0 or not top_chunks:
            return response
        
        # Simple citation addition (in production, use more sophisticated matching)
        citations_added = 0
        enhanced_response = response
        
        for i, chunk in enumerate(top_chunks[:min_citations]):
            if citations_added >= min_citations:
                break
            
            # Add citation marker
            citation = f" [Source {i+1}]"
            # Insert citation after first sentence that might relate to this chunk
            sentences = enhanced_response.split('.')
            if len(sentences) > citations_added:
                sentences[citations_added] += citation
                enhanced_response = '.'.join(sentences)
                citations_added += 1
        
        return enhanced_response
    
    def _assess_source_quality(self, top_chunks: List[Dict[str, Any]]) -> float:
        """Assess quality of available source chunks."""
        if not top_chunks:
            return 0.0
        
        total_score = 0.0
        for chunk in top_chunks:
            score = chunk.get("score", 0.5)  # Default relevance score
            text_length = len(chunk.get("text", ""))
            
            # Higher score for longer, more relevant chunks
            quality_score = score * min(1.0, text_length / 200)  # Normalize by expected length
            total_score += quality_score
        
        return total_score / len(top_chunks)
    
    def _update_template_stats(self, template_id: str):
        """Update usage statistics for template."""
        if template_id not in self.composition_stats["template_usage"]:
            self.composition_stats["template_usage"][template_id] = 0
        self.composition_stats["template_usage"][template_id] += 1
    
    async def _log_rl_action(self, 
                            context: CompositionContext,
                            template: ResponseTemplate, 
                            grounding_metrics: GroundingMetrics):
        """Log template selection as RL action."""
        try:
            action_data = {
                "action_type": "template_selection",
                "template_id": template.template_id,
                "grounding_score": grounding_metrics.overall_score,
                "is_grounded": grounding_metrics.is_grounded,
                "source_count": len(context.top_chunks),
                "input_length": len(context.input_text.split()) if context.input_text else 0
            }
            
            # Log to RL context for future policy improvements
            self.rl_context.log_action(context.task_id, "template_selector", action_data)
            
        except Exception as e:
            logger.warning(f"Failed to log RL action: {e}")
    
    def _build_composition_trace(self, 
                                context: CompositionContext,
                                template: ResponseTemplate, 
                                grounding_metrics: GroundingMetrics,
                                processing_time: float,
                                fallback_applied: bool) -> Dict[str, Any]:
        """Build detailed composition trace for debugging and analysis."""
        return {
            "task_id": context.task_id,
            "template_id": template.template_id,
            "template_type": template.style,
            "grounded": grounding_metrics.is_grounded,
            "grounding_score": grounding_metrics.overall_score,
            "grounding_details": asdict(grounding_metrics),
            "fallback_applied": fallback_applied,
            "processing_time": processing_time,
            "source_chunks_count": len(context.top_chunks),
            "input_type": context.input_type,
            "timestamp": datetime.now().isoformat(),
            "composition_stats": self.composition_stats.copy()
        }
    
    def get_composition_stats(self) -> Dict[str, Any]:
        """Get current composition statistics."""
        stats = self.composition_stats.copy()
        if stats["total_compositions"] > 0:
            stats["grounding_success_rate"] = stats["successful_groundings"] / stats["total_compositions"]
            stats["fallback_rate"] = stats["fallback_applications"] / stats["total_compositions"]
        else:
            stats["grounding_success_rate"] = 0.0
            stats["fallback_rate"] = 0.0
        
        return stats


# Convenience function for direct composition
async def compose_response_with_grounding(task_id: str,
                                        input_text: str,
                                        top_chunks: List[Dict[str, Any]],
                                        input_type: str = "text",
                                        agent_type: str = "text_agent",
                                        model_name: str = "default",
                                        explicit_template_id: Optional[str] = None) -> CompositionResult:
    """
    Convenience function for response composition with grounding.
    
    Args:
        task_id: Unique task identifier
        input_text: User input text
        top_chunks: Relevant source chunks for grounding
        input_type: Type of input (text, voice, etc.)
        agent_type: Agent handling the request
        model_name: Model being used
        explicit_template_id: Optional explicit template to use
        
    Returns:
        CompositionResult with response and metadata
    """
    composer = ResponseComposer()
    
    context = CompositionContext(
        task_id=task_id,
        input_text=input_text,
        input_type=input_type,
        top_chunks=top_chunks,
        agent_type=agent_type,
        model_name=model_name,
        explicit_template_id=explicit_template_id
    )
    
    return await composer.compose_response(context)


# Template performance monitoring
def update_composition_reward(task_id: str, template_id: str, reward: float, grounding_metrics: Optional[GroundingMetrics] = None):
    """
    Update template performance based on composition results.
    
    Args:
        task_id: Task identifier
        template_id: Template that was used
        reward: Reward/feedback score
        grounding_metrics: Optional grounding verification results
    """
    try:
        # Create template selector instance to update performance
        selector = TemplateSelector()
        selector.update_performance(task_id, template_id, reward, grounding_metrics)
        
        logger.info(f"Updated composition reward for task {task_id}: template={template_id}, reward={reward:.3f}")
        
    except Exception as e:
        logger.error(f"Failed to update composition reward: {e}")