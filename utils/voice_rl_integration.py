"""
Voice-specific Reinforcement Learning Integration for BHIV Core
Implements feedback loops for voice interactions, STT/TTS quality, and user satisfaction
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from utils.logger import get_logger
from reinforcement.replay_buffer import replay_buffer
from reinforcement.reward_functions import get_reward_from_output
from reinforcement.rl_context import RLContext

logger = get_logger(__name__)

class InteractionType(Enum):
    """Types of voice interactions for reward calculation."""
    VOICE_QUERY = "voice_query"
    WAKE_WORD = "wake_word"
    INTERRUPT = "voice_interrupt"
    TTS_PLAYBACK = "tts_playback"
    STT_TRANSCRIPTION = "stt_transcription"

class FeedbackType(Enum):
    """Types of feedback for RL learning."""
    EXPLICIT = "explicit"  # Direct user feedback (thumbs up/down)
    IMPLICIT = "implicit"  # Inferred from behavior
    SYSTEM = "system"      # Automatic system metrics

@dataclass
class VoiceInteractionMetrics:
    """Metrics for voice interaction quality."""
    task_id: str
    interaction_type: InteractionType
    transcription_accuracy: float = 0.0
    response_relevance: float = 0.0
    voice_clarity: float = 0.0
    response_time: float = 0.0
    user_satisfaction: float = 0.0
    interruption_rate: float = 0.0
    wake_word_accuracy: float = 0.0
    persona_appropriateness: float = 0.0
    language_detection_accuracy: float = 0.0
    conversation_flow: float = 0.0

@dataclass
class VoiceFeedback:
    """Voice-specific feedback data."""
    task_id: str
    feedback_type: FeedbackType
    rating: float  # -1.0 to 1.0
    metrics: VoiceInteractionMetrics
    timestamp: datetime
    user_id: str
    persona_used: str
    language_used: str
    processing_time: float
    context: Dict[str, Any]

class VoiceRewardCalculator:
    """Calculate rewards for voice interactions based on multiple factors."""
    
    def __init__(self):
        self.base_weights = {
            'transcription_accuracy': 0.25,
            'response_relevance': 0.20,
            'voice_clarity': 0.15,
            'response_time': 0.15,
            'user_satisfaction': 0.25
        }
        
        self.penalty_weights = {
            'high_latency': -0.2,
            'poor_transcription': -0.3,
            'inappropriate_persona': -0.15,
            'language_mismatch': -0.2
        }
        
        self.bonus_weights = {
            'wake_word_success': 0.1,
            'smooth_interruption': 0.1,
            'persona_match': 0.1,
            'conversation_continuity': 0.1
        }
    
    def calculate_reward(self, feedback: VoiceFeedback) -> float:
        """Calculate reward based on voice interaction feedback."""
        try:
            metrics = feedback.metrics
            base_reward = 0.0
            
            # Base reward from core metrics
            base_reward += metrics.transcription_accuracy * self.base_weights['transcription_accuracy']
            base_reward += metrics.response_relevance * self.base_weights['response_relevance']
            base_reward += metrics.voice_clarity * self.base_weights['voice_clarity']
            base_reward += metrics.user_satisfaction * self.base_weights['user_satisfaction']
            
            # Response time reward (inverse relationship)
            time_reward = max(0, 1.0 - (feedback.processing_time / 10.0))  # Penalty after 10s
            base_reward += time_reward * self.base_weights['response_time']
            
            # Apply penalties
            if feedback.processing_time > 5.0:  # High latency
                base_reward += self.penalty_weights['high_latency']
            
            if metrics.transcription_accuracy < 0.7:  # Poor transcription
                base_reward += self.penalty_weights['poor_transcription']
            
            if metrics.language_detection_accuracy < 0.8:  # Language mismatch
                base_reward += self.penalty_weights['language_mismatch']
            
            if metrics.persona_appropriateness < 0.6:  # Inappropriate persona
                base_reward += self.penalty_weights['inappropriate_persona']
            
            # Apply bonuses
            if metrics.wake_word_accuracy > 0.9:
                base_reward += self.bonus_weights['wake_word_success']
            
            if metrics.interruption_rate < 0.1:  # Low interruption rate
                base_reward += self.bonus_weights['smooth_interruption']
            
            if metrics.persona_appropriateness > 0.8:
                base_reward += self.bonus_weights['persona_match']
            
            if metrics.conversation_flow > 0.8:
                base_reward += self.bonus_weights['conversation_continuity']
            
            # Apply user feedback weight
            if feedback.feedback_type == FeedbackType.EXPLICIT:
                base_reward = base_reward * 0.7 + feedback.rating * 0.3
            
            # Normalize reward to [-1.0, 1.0]
            reward = max(-1.0, min(1.0, base_reward))
            
            logger.info(f"Calculated voice reward: {reward:.3f} for task {feedback.task_id}")
            return reward
            
        except Exception as e:
            logger.error(f"Error calculating voice reward: {e}")
            return 0.0

class VoiceFeedbackCollector:
    """Collect and process voice interaction feedback."""
    
    def __init__(self, reward_calculator: VoiceRewardCalculator):
        self.reward_calculator = reward_calculator
        self.feedback_buffer: List[VoiceFeedback] = []
        self.metrics_cache: Dict[str, VoiceInteractionMetrics] = {}
        self.rl_context = RLContext()
    
    async def collect_stt_feedback(self, task_id: str, transcription: str, 
                                   confidence: float, processing_time: float,
                                   method_used: str) -> None:
        """Collect STT-specific feedback."""
        try:
            # Estimate transcription accuracy based on confidence and other factors
            accuracy_score = self._estimate_stt_accuracy(transcription, confidence, method_used)
            
            metrics = VoiceInteractionMetrics(
                task_id=task_id,
                interaction_type=InteractionType.STT_TRANSCRIPTION,
                transcription_accuracy=accuracy_score,
                response_time=processing_time,
                voice_clarity=confidence
            )
            
            self.metrics_cache[f"{task_id}_stt"] = metrics
            logger.debug(f"Collected STT feedback for task {task_id}: accuracy={accuracy_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error collecting STT feedback: {e}")
    
    async def collect_tts_feedback(self, task_id: str, persona: str, language: str,
                                   generation_time: float, audio_file: Optional[str]) -> None:
        """Collect TTS-specific feedback."""
        try:
            # Estimate TTS quality based on generation success and time
            voice_clarity = 1.0 if audio_file else 0.0
            persona_appropriateness = self._estimate_persona_appropriateness(persona, language)
            
            metrics = VoiceInteractionMetrics(
                task_id=task_id,
                interaction_type=InteractionType.TTS_PLAYBACK,
                voice_clarity=voice_clarity,
                response_time=generation_time,
                persona_appropriateness=persona_appropriateness,
                language_detection_accuracy=1.0  # Assume correct since user selected
            )
            
            self.metrics_cache[f"{task_id}_tts"] = metrics
            logger.debug(f"Collected TTS feedback for task {task_id}: clarity={voice_clarity}")
            
        except Exception as e:
            logger.error(f"Error collecting TTS feedback: {e}")
    
    async def collect_voice_query_feedback(self, task_id: str, user_query: str,
                                           agent_response: str, agent_used: str,
                                           total_processing_time: float) -> None:
        """Collect feedback for complete voice query interaction."""
        try:
            # Estimate response relevance
            relevance_score = self._estimate_response_relevance(user_query, agent_response)
            
            # Get cached STT and TTS metrics
            stt_metrics = self.metrics_cache.get(f"{task_id}_stt")
            tts_metrics = self.metrics_cache.get(f"{task_id}_tts")
            
            # Combine metrics
            combined_metrics = VoiceInteractionMetrics(
                task_id=task_id,
                interaction_type=InteractionType.VOICE_QUERY,
                transcription_accuracy=stt_metrics.transcription_accuracy if stt_metrics else 0.0,
                response_relevance=relevance_score,
                voice_clarity=tts_metrics.voice_clarity if tts_metrics else 0.0,
                response_time=total_processing_time,
                persona_appropriateness=tts_metrics.persona_appropriateness if tts_metrics else 0.0,
                language_detection_accuracy=tts_metrics.language_detection_accuracy if tts_metrics else 0.0
            )
            
            # Create feedback entry
            feedback = VoiceFeedback(
                task_id=task_id,
                feedback_type=FeedbackType.SYSTEM,
                rating=0.0,  # Will be calculated
                metrics=combined_metrics,
                timestamp=datetime.now(),
                user_id="system",
                persona_used=tts_metrics.persona_appropriateness if tts_metrics else "unknown",
                language_used="auto_detected",
                processing_time=total_processing_time,
                context={"agent_used": agent_used, "query_length": len(user_query)}
            )
            
            # Calculate reward
            reward = self.reward_calculator.calculate_reward(feedback)
            feedback.rating = reward
            
            # Store feedback
            self.feedback_buffer.append(feedback)
            
            # Add to RL replay buffer
            await self._add_to_rl_buffer(feedback, reward)
            
            logger.info(f"Collected voice query feedback for task {task_id}: reward={reward:.3f}")
            
        except Exception as e:
            logger.error(f"Error collecting voice query feedback: {e}")
    
    async def collect_user_feedback(self, task_id: str, user_rating: float,
                                    feedback_text: Optional[str] = None) -> None:
        """Collect explicit user feedback."""
        try:
            # Find existing feedback for this task
            existing_feedback = None
            for fb in self.feedback_buffer:
                if fb.task_id == task_id:
                    existing_feedback = fb
                    break
            
            if existing_feedback:
                # Update with user feedback
                existing_feedback.feedback_type = FeedbackType.EXPLICIT
                existing_feedback.rating = user_rating
                existing_feedback.metrics.user_satisfaction = (user_rating + 1.0) / 2.0  # Convert to 0-1
                
                # Recalculate reward with user feedback
                new_reward = self.reward_calculator.calculate_reward(existing_feedback)
                
                # Update RL buffer
                await self._add_to_rl_buffer(existing_feedback, new_reward)
                
                logger.info(f"Updated feedback with user rating for task {task_id}: {user_rating}")
            else:
                logger.warning(f"No existing feedback found for task {task_id}")
                
        except Exception as e:
            logger.error(f"Error collecting user feedback: {e}")
    
    async def collect_wake_word_feedback(self, detected_word: str, confidence: float,
                                         false_positive: bool = False) -> None:
        """Collect wake word detection feedback."""
        try:
            task_id = f"wake_word_{int(time.time())}"
            accuracy = 0.0 if false_positive else confidence
            
            metrics = VoiceInteractionMetrics(
                task_id=task_id,
                interaction_type=InteractionType.WAKE_WORD,
                wake_word_accuracy=accuracy,
                voice_clarity=confidence
            )
            
            feedback = VoiceFeedback(
                task_id=task_id,
                feedback_type=FeedbackType.SYSTEM,
                rating=accuracy,
                metrics=metrics,
                timestamp=datetime.now(),
                user_id="system",
                persona_used="none",
                language_used="auto",
                processing_time=0.1,
                context={"detected_word": detected_word, "false_positive": false_positive}
            )
            
            self.feedback_buffer.append(feedback)
            
            reward = self.reward_calculator.calculate_reward(feedback)
            await self._add_to_rl_buffer(feedback, reward)
            
            logger.debug(f"Collected wake word feedback: {detected_word}, accuracy={accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Error collecting wake word feedback: {e}")
    
    def _estimate_stt_accuracy(self, transcription: str, confidence: float, method: str) -> float:
        """Estimate STT accuracy based on available metrics."""
        # Base score from confidence
        base_score = confidence
        
        # Adjust based on transcription characteristics
        if len(transcription.strip()) == 0:
            return 0.0
        
        # Bonus for reasonable length
        if 5 <= len(transcription.split()) <= 50:
            base_score += 0.1
        
        # Method-based adjustment
        method_bonuses = {
            'whisper': 0.1,
            'wav2vec2': 0.05,
            'speech_recognition': 0.0
        }
        base_score += method_bonuses.get(method, 0.0)
        
        return min(1.0, base_score)
    
    def _estimate_response_relevance(self, query: str, response: str) -> float:
        """Estimate how relevant the response is to the query."""
        # Simple keyword overlap approach (can be enhanced with semantic similarity)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        if len(query_words) == 0:
            return 0.0
        
        overlap = len(query_words.intersection(response_words))
        relevance = overlap / len(query_words)
        
        # Bonus for reasonable response length
        if 10 <= len(response.split()) <= 200:
            relevance += 0.1
        
        return min(1.0, relevance)
    
    def _estimate_persona_appropriateness(self, persona: str, language: str) -> float:
        """Estimate if the persona is appropriate for the content/language."""
        # Simple rule-based approach
        persona_language_fit = {
            ('guru', 'hi'): 1.0,
            ('guru', 'en'): 0.9,
            ('teacher', 'en'): 1.0,
            ('teacher', 'hi'): 0.8,
            ('assistant', 'en'): 1.0,
            ('assistant', 'hi'): 0.9,
            ('friend', 'en'): 0.9,
            ('friend', 'hi'): 0.9
        }
        
        return persona_language_fit.get((persona, language), 0.7)
    
    async def _add_to_rl_buffer(self, feedback: VoiceFeedback, reward: float) -> None:
        """Add feedback to RL replay buffer."""
        try:
            # Create RL entry
            rl_entry = {
                'task_id': feedback.task_id,
                'agent': 'voice_system',
                'input': feedback.context.get('user_query', ''),
                'output': {
                    'type': feedback.metrics.interaction_type.value,
                    'metrics': asdict(feedback.metrics),
                    'feedback_type': feedback.feedback_type.value
                },
                'reward': reward,
                'timestamp': feedback.timestamp.isoformat(),
                'model': feedback.persona_used,
                'processing_time': feedback.processing_time
            }
            
            # Add to replay buffer
            replay_buffer.add_experience(
                task_id=feedback.task_id,
                input_data=rl_entry['input'],
                output=rl_entry['output'],
                agent="voice_system",
                model=feedback.persona_used,
                reward=reward
            )
            
            logger.debug(f"Added voice feedback to RL buffer: {feedback.task_id}")
            
        except Exception as e:
            logger.error(f"Error adding to RL buffer: {e}")
    
    def get_feedback_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of feedback collected in the last N hours."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_feedback = [fb for fb in self.feedback_buffer if fb.timestamp > cutoff_time]
            
            if not recent_feedback:
                return {"message": "No recent feedback available"}
            
            # Calculate averages
            avg_metrics = {
                'transcription_accuracy': 0.0,
                'response_relevance': 0.0,
                'voice_clarity': 0.0,
                'response_time': 0.0,
                'user_satisfaction': 0.0,
                'overall_reward': 0.0
            }
            
            for fb in recent_feedback:
                avg_metrics['transcription_accuracy'] += fb.metrics.transcription_accuracy
                avg_metrics['response_relevance'] += fb.metrics.response_relevance
                avg_metrics['voice_clarity'] += fb.metrics.voice_clarity
                avg_metrics['response_time'] += fb.metrics.response_time
                avg_metrics['user_satisfaction'] += fb.metrics.user_satisfaction
                avg_metrics['overall_reward'] += fb.rating
            
            count = len(recent_feedback)
            for key in avg_metrics:
                avg_metrics[key] /= count
            
            # Count by interaction type
            type_counts = {}
            for fb in recent_feedback:
                interaction_type = fb.metrics.interaction_type.value
                type_counts[interaction_type] = type_counts.get(interaction_type, 0) + 1
            
            return {
                'total_interactions': count,
                'time_period_hours': hours,
                'average_metrics': avg_metrics,
                'interaction_types': type_counts,
                'feedback_types': {
                    'explicit': len([fb for fb in recent_feedback if fb.feedback_type == FeedbackType.EXPLICIT]),
                    'implicit': len([fb for fb in recent_feedback if fb.feedback_type == FeedbackType.IMPLICIT]),
                    'system': len([fb for fb in recent_feedback if fb.feedback_type == FeedbackType.SYSTEM])
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating feedback summary: {e}")
            return {"error": str(e)}

# Global instance for voice feedback collection
voice_reward_calculator = VoiceRewardCalculator()
voice_feedback_collector = VoiceFeedbackCollector(voice_reward_calculator)

async def main():
    """Demo of voice RL integration."""
    print("Voice RL Integration Demo")
    
    # Simulate STT feedback
    await voice_feedback_collector.collect_stt_feedback(
        task_id="demo_task_1",
        transcription="What is the meaning of life?",
        confidence=0.9,
        processing_time=1.2,
        method_used="whisper"
    )
    
    # Simulate TTS feedback
    await voice_feedback_collector.collect_tts_feedback(
        task_id="demo_task_1",
        persona="guru",
        language="en",
        generation_time=2.1,
        audio_file="response.mp3"
    )
    
    # Simulate complete voice query feedback
    await voice_feedback_collector.collect_voice_query_feedback(
        task_id="demo_task_1",
        user_query="What is the meaning of life?",
        agent_response="The meaning of life is a profound philosophical question that has been contemplated for centuries.",
        agent_used="vedas_agent",
        total_processing_time=5.5
    )
    
    # Simulate user feedback
    await voice_feedback_collector.collect_user_feedback(
        task_id="demo_task_1",
        user_rating=0.8,
        feedback_text="Great response!"
    )
    
    # Get feedback summary
    summary = voice_feedback_collector.get_feedback_summary(1)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    asyncio.run(main())