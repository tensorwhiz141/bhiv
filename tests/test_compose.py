"""
Unit Tests for Composer Module

Tests response composition, template logic, grounding enforcement,
and policy hooks with comprehensive coverage.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# Add project root to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from composer.compose import (
    ResponseComposer, 
    CompositionResult,
    CompositionContext,
    compose_response_with_grounding,
    update_composition_reward
)
from composer.gru import GRUResponseModel, GRUComposer, GRUTemplateAdapter
from config.template_config import ResponseTemplate, get_template_by_id
from utils.grounding_verifier import GroundingMetrics

class TestCompositionResult:
    """Test CompositionResult dataclass."""
    
    def test_composition_result_creation(self):
        """Test basic CompositionResult creation."""
        result = CompositionResult(
            response_text="Test response",
            template_id="test_template",
            grounded=True,
            grounding_score=0.85,
            composition_trace={"test": "data"},
            processing_time=1.5
        )
        
        assert result.response_text == "Test response"
        assert result.template_id == "test_template"
        assert result.grounded is True
        assert result.grounding_score == 0.85
        assert result.composition_trace == {"test": "data"}
        assert result.processing_time == 1.5
        assert result.fallback_applied is False
        assert result.original_template_id is None
    
    def test_composition_result_with_fallback(self):
        """Test CompositionResult with fallback applied."""
        result = CompositionResult(
            response_text="Fallback response",
            template_id="extractive_heavy",
            grounded=True,
            grounding_score=0.75,
            composition_trace={"fallback": True},
            processing_time=2.1,
            fallback_applied=True,
            original_template_id="generative_standard"
        )
        
        assert result.fallback_applied is True
        assert result.original_template_id == "generative_standard"

class TestCompositionContext:
    """Test CompositionContext dataclass."""
    
    def test_composition_context_creation(self):
        """Test basic CompositionContext creation."""
        top_chunks = [
            {"text": "Sample text", "score": 0.9},
            {"text": "Another chunk", "score": 0.7}
        ]
        
        context = CompositionContext(
            task_id="test_task_123",
            input_text="What is yoga?",
            input_type="text",
            top_chunks=top_chunks,
            agent_type="text_agent",
            model_name="test_model"
        )
        
        assert context.task_id == "test_task_123"
        assert context.input_text == "What is yoga?"
        assert context.input_type == "text"
        assert len(context.top_chunks) == 2
        assert context.agent_type == "text_agent"
        assert context.model_name == "test_model"
        assert context.user_context is None
        assert context.explicit_template_id is None

class TestResponseComposer:
    """Test ResponseComposer main functionality."""
    
    @pytest.fixture
    def composer(self):
        """Create ResponseComposer instance for testing."""
        return ResponseComposer()
    
    @pytest.fixture
    def sample_context(self):
        """Create sample composition context."""
        return CompositionContext(
            task_id="test_task_001",
            input_text="What is the essence of yoga?",
            input_type="text",
            top_chunks=[
                {
                    "text": "Yoga is a physical, mental and spiritual practice that originated in India. The word yoga comes from the Sanskrit root yuj, meaning to unite or join.",
                    "score": 0.95,
                    "source": "yoga_basics"
                },
                {
                    "text": "The practice of yoga includes physical postures (asanas), breath control (pranayama), and meditation (dhyana).",
                    "score": 0.88,
                    "source": "yoga_components"
                }
            ],
            agent_type="text_agent",
            model_name="test_model"
        )
    
    def test_composer_initialization(self, composer):
        """Test ResponseComposer initialization."""
        assert composer is not None
        assert hasattr(composer, 'grounding_verifier')
        assert hasattr(composer, 'template_selector')
        assert hasattr(composer, 'rl_context')
        assert composer.composition_stats["total_compositions"] == 0
    
    @pytest.mark.asyncio
    async def test_compose_response_success(self, composer, sample_context):
        """Test successful response composition."""
        # Mock template selector
        mock_template = ResponseTemplate(
            template_id="test_template",
            name="Test Template",
            description="Test template",
            max_length=150,
            min_citations=2,
            extractive_ratio=0.5,
            style="balanced",
            grounding_requirements={
                "min_source_overlap": 0.4,
                "citation_density": 0.1,
                "factual_consistency": 0.7
            }
        )
        
        # Mock grounding metrics
        mock_grounding = GroundingMetrics(
            source_overlap=0.8,
            citation_density=0.15,
            factual_consistency=0.85,
            overall_score=0.8,
            is_grounded=True,
            details={"test": "data"}
        )
        
        with patch.object(composer.template_selector, 'select_template', return_value=mock_template), \
             patch.object(composer.grounding_verifier, 'verify_grounding', return_value=mock_grounding), \
             patch.object(composer.rl_context, 'log_action'):
            
            result = await composer.compose_response(sample_context)
            
            assert isinstance(result, CompositionResult)
            assert result.template_id == "test_template"
            assert result.grounded is True
            assert result.grounding_score == 0.8
            assert not result.fallback_applied
            assert len(result.response_text) > 0
            assert "yoga" in result.response_text.lower()
    
    @pytest.mark.asyncio
    async def test_compose_response_with_fallback(self, composer, sample_context):
        """Test response composition with grounding failure and fallback."""
        # Mock initial template (generative)
        initial_template = ResponseTemplate(
            template_id="generative_standard",
            name="Generative Template",
            description="Generative template",
            max_length=200,
            min_citations=1,
            extractive_ratio=0.3,
            style="conversational",
            grounding_requirements={
                "min_source_overlap": 0.5,
                "citation_density": 0.05,
                "factual_consistency": 0.6
            }
        )
        
        # Mock fallback template (extractive)
        fallback_template = ResponseTemplate(
            template_id="extractive_heavy",
            name="Extractive Template",
            description="Extractive template",
            max_length=100,
            min_citations=3,
            extractive_ratio=0.8,
            style="scholarly",
            grounding_requirements={
                "min_source_overlap": 0.3,
                "citation_density": 0.15,
                "factual_consistency": 0.7
            }
        )
        
        # Mock grounding failure, then success
        failed_grounding = GroundingMetrics(
            source_overlap=0.2,
            citation_density=0.02,
            factual_consistency=0.4,
            overall_score=0.3,
            is_grounded=False,
            details={"failed": True}
        )
        
        success_grounding = GroundingMetrics(
            source_overlap=0.7,
            citation_density=0.2,
            factual_consistency=0.8,
            overall_score=0.75,
            is_grounded=True,
            details={"fallback_success": True}
        )
        
        with patch.object(composer.template_selector, 'select_template', return_value=initial_template), \
             patch.object(composer.template_selector, 'handle_grounding_failure', return_value=fallback_template), \
             patch.object(composer.grounding_verifier, 'verify_grounding', side_effect=[failed_grounding, success_grounding]), \
             patch.object(composer.rl_context, 'log_action'):
            
            result = await composer.compose_response(sample_context)
            
            assert isinstance(result, CompositionResult)
            assert result.template_id == "extractive_heavy"
            assert result.fallback_applied is True
            assert result.original_template_id == "generative_standard"
            assert result.grounded is True
            assert result.grounding_score == 0.75
    
    @pytest.mark.asyncio
    async def test_compose_response_explicit_template(self, composer, sample_context):
        """Test composition with explicit template selection."""
        sample_context.explicit_template_id = "balanced_hybrid"
        
        mock_template = get_template_by_id("balanced_hybrid")
        mock_grounding = GroundingMetrics(
            source_overlap=0.6,
            citation_density=0.12,
            factual_consistency=0.75,
            overall_score=0.7,
            is_grounded=True,
            details={"explicit": True}
        )
        
        with patch.object(composer.grounding_verifier, 'verify_grounding', return_value=mock_grounding), \
             patch.object(composer.rl_context, 'log_action'):
            
            result = await composer.compose_response(sample_context)
            
            assert result.template_id == "balanced_hybrid"
            assert not result.fallback_applied
    
    @pytest.mark.asyncio
    async def test_compose_response_error_handling(self, composer, sample_context):
        """Test error handling in composition."""
        with patch.object(composer.template_selector, 'select_template', side_effect=Exception("Test error")):
            
            result = await composer.compose_response(sample_context)
            
            assert isinstance(result, CompositionResult)
            assert result.template_id == "error_fallback"
            assert result.grounded is False
            assert result.fallback_applied is True
            assert "error" in result.response_text.lower()
    
    @pytest.mark.asyncio
    async def test_generate_extractive_response(self, composer, sample_context):
        """Test extractive response generation."""
        template = ResponseTemplate(
            template_id="extractive_heavy",
            name="Extractive",
            description="Extractive template",
            max_length=100,
            min_citations=3,
            extractive_ratio=0.8,
            style="scholarly",
            grounding_requirements={}
        )
        
        source_content = [chunk["text"] for chunk in sample_context.top_chunks]
        
        response = await composer._generate_extractive_response(
            sample_context.input_text, source_content, template
        )
        
        assert len(response) > 0
        assert "based on" in response.lower() or "sources" in response.lower()
        
        # Should contain content from sources
        assert any(word in response.lower() for word in ["yoga", "practice", "meditation"])
    
    @pytest.mark.asyncio
    async def test_generate_balanced_response(self, composer, sample_context):
        """Test balanced response generation."""
        template = ResponseTemplate(
            template_id="balanced_hybrid",
            name="Balanced",
            description="Balanced template",
            max_length=150,
            min_citations=2,
            extractive_ratio=0.5,
            style="academic",
            grounding_requirements={}
        )
        
        source_content = [chunk["text"] for chunk in sample_context.top_chunks]
        
        response = await composer._generate_balanced_response(
            sample_context.input_text, source_content, template
        )
        
        assert len(response) > 0
        assert "yoga" in response.lower()
        # Should be longer than extractive but include source content
        assert len(response.split()) >= 10
    
    @pytest.mark.asyncio
    async def test_generate_generative_response(self, composer, sample_context):
        """Test generative response generation."""
        template = ResponseTemplate(
            template_id="generative_standard",
            name="Generative",
            description="Generative template",
            max_length=200,
            min_citations=1,
            extractive_ratio=0.3,
            style="conversational",
            grounding_requirements={}
        )
        
        source_content = [chunk["text"] for chunk in sample_context.top_chunks]
        
        response = await composer._generate_generative_response(
            sample_context.input_text, source_content, template
        )
        
        assert len(response) > 0
        assert "yoga" in response.lower()
        # Should be more interpretive and explanatory
        assert len(response.split()) >= 15
    
    def test_compute_token_overlap(self, composer):
        """Test token overlap computation."""
        response_text = "Yoga is a practice of physical and mental exercises"
        source_texts = [
            "Yoga includes physical postures and breathing",
            "Mental exercises are part of yoga practice"
        ]
        
        overlap = composer._compute_token_overlap(response_text, source_texts)
        
        assert 0.0 <= overlap <= 1.0
        assert overlap > 0.3  # Should have significant overlap
    
    def test_apply_length_constraints(self, composer):
        """Test length constraint application."""
        long_text = " ".join(["word"] * 150)  # 150 words
        
        # Test truncation
        constrained = composer._apply_length_constraints(long_text, 100)
        assert len(constrained.split()) <= 100
        
        # Test no truncation needed
        short_text = "Short response"
        constrained = composer._apply_length_constraints(short_text, 100)
        assert constrained == short_text
    
    def test_add_citations(self, composer):
        """Test citation addition."""
        response = "Yoga is a comprehensive practice. It includes physical postures."
        top_chunks = [
            {"text": "Source 1 text", "score": 0.9},
            {"text": "Source 2 text", "score": 0.8}
        ]
        
        with_citations = composer._add_citations(response, top_chunks, 2)
        
        assert "[Source 1]" in with_citations
        assert len(with_citations) > len(response)
    
    def test_assess_source_quality(self, composer):
        """Test source quality assessment."""
        high_quality_chunks = [
            {"text": "Long detailed text about yoga practices and philosophy", "score": 0.9},
            {"text": "Another comprehensive source about meditation", "score": 0.85}
        ]
        
        low_quality_chunks = [
            {"text": "Short", "score": 0.3},
            {"text": "Brief", "score": 0.2}
        ]
        
        high_quality = composer._assess_source_quality(high_quality_chunks)
        low_quality = composer._assess_source_quality(low_quality_chunks)
        
        assert high_quality > low_quality
        assert 0.0 <= high_quality <= 1.0
        assert 0.0 <= low_quality <= 1.0
    
    def test_get_composition_stats(self, composer):
        """Test composition statistics retrieval."""
        # Simulate some compositions
        composer.composition_stats["total_compositions"] = 10
        composer.composition_stats["successful_groundings"] = 8
        composer.composition_stats["fallback_applications"] = 2
        
        stats = composer.get_composition_stats()
        
        assert stats["total_compositions"] == 10
        assert stats["grounding_success_rate"] == 0.8
        assert stats["fallback_rate"] == 0.2

class TestConvenienceFunctions:
    """Test module convenience functions."""
    
    @pytest.mark.asyncio
    async def test_compose_response_with_grounding(self):
        """Test convenience function for response composition."""
        top_chunks = [
            {"text": "Yoga is an ancient practice", "score": 0.9}
        ]
        
        with patch('composer.compose.ResponseComposer') as mock_composer_class:
            mock_composer = Mock()
            mock_result = CompositionResult(
                response_text="Test response",
                template_id="test_template",
                grounded=True,
                grounding_score=0.8,
                composition_trace={},
                processing_time=1.0
            )
            mock_composer.compose_response = AsyncMock(return_value=mock_result)
            mock_composer_class.return_value = mock_composer
            
            result = await compose_response_with_grounding(
                task_id="test_task",
                input_text="What is yoga?",
                top_chunks=top_chunks
            )
            
            assert isinstance(result, CompositionResult)
            assert result.response_text == "Test response"
            mock_composer.compose_response.assert_called_once()
    
    def test_update_composition_reward(self):
        """Test composition reward update function."""
        mock_grounding = GroundingMetrics(
            source_overlap=0.7,
            citation_density=0.1,
            factual_consistency=0.8,
            overall_score=0.75,
            is_grounded=True,
            details={}
        )
        
        with patch('composer.compose.TemplateSelector') as mock_selector_class:
            mock_selector = Mock()
            mock_selector_class.return_value = mock_selector
            
            update_composition_reward("test_task", "test_template", 0.8, mock_grounding)
            
            mock_selector.update_performance.assert_called_once_with(
                "test_task", "test_template", 0.8, mock_grounding
            )

class TestGRUStubs:
    """Test GRU stub implementations."""
    
    def test_gru_response_model_init(self):
        """Test GRU model initialization."""
        model = GRUResponseModel(vocab_size=1000, hidden_dim=256)
        
        assert model.vocab_size == 1000
        assert model.hidden_dim == 256
        assert hasattr(model, 'embedding')
        assert hasattr(model, 'gru')
        assert hasattr(model, 'output_projection')
    
    def test_gru_response_model_forward(self):
        """Test GRU model forward pass."""
        import torch
        
        model = GRUResponseModel(vocab_size=100, embedding_dim=32, hidden_dim=64)
        input_ids = torch.randint(0, 100, (2, 10))  # batch_size=2, seq_len=10
        
        output_logits, hidden = model.forward(input_ids)
        
        assert output_logits.shape == (2, 10, 100)  # [batch, seq, vocab]
        assert hidden.shape == (2, 2, 64)  # [num_layers, batch, hidden]
    
    def test_gru_composer_init(self):
        """Test GRU composer initialization."""
        composer = GRUComposer()
        
        assert composer.model is None
        assert composer.tokenizer is None
        assert not composer.is_loaded
    
    @pytest.mark.asyncio
    async def test_gru_composer_compose(self):
        """Test GRU composer composition (stub)."""
        composer = GRUComposer()
        
        result = await composer.compose_with_gru({}, {})
        
        assert "error" in result
        assert result["method"] == "gru_stub"
    
    def test_gru_template_adapter_init(self):
        """Test GRU template adapter initialization."""
        adapter = GRUTemplateAdapter()
        
        assert isinstance(adapter.template_embeddings, dict)
        assert isinstance(adapter.style_controllers, dict)

class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""
    
    @pytest.mark.asyncio
    async def test_full_composition_pipeline(self):
        """Test complete composition pipeline integration."""
        # This test verifies the entire flow works together
        composer = ResponseComposer()
        
        context = CompositionContext(
            task_id="integration_test",
            input_text="Explain the benefits of meditation",
            input_type="text",
            top_chunks=[
                {
                    "text": "Meditation reduces stress and improves mental clarity. Regular practice leads to better emotional regulation.",
                    "score": 0.92,
                    "source": "meditation_benefits"
                },
                {
                    "text": "Scientific studies show meditation increases gray matter in the brain and improves focus and attention.",
                    "score": 0.88,
                    "source": "meditation_science"
                }
            ],
            agent_type="text_agent",
            model_name="integration_test_model"
        )
        
        with patch.object(composer.rl_context, 'log_action'):
            result = await composer.compose_response(context)
            
            # Verify complete result structure
            assert isinstance(result, CompositionResult)
            assert len(result.response_text) > 20
            assert result.template_id in ["generative_standard", "balanced_hybrid", "extractive_heavy"]
            assert isinstance(result.grounded, bool)
            assert 0.0 <= result.grounding_score <= 1.0
            assert isinstance(result.composition_trace, dict)
            assert result.processing_time > 0
            
            # Verify response quality
            response_lower = result.response_text.lower()
            assert "meditation" in response_lower
            
            # Verify trace contains expected fields
            trace = result.composition_trace
            assert "task_id" in trace
            assert "template_id" in trace
            assert "grounded" in trace
            assert "grounding_score" in trace
    
    @pytest.mark.asyncio
    async def test_no_sources_scenario(self):
        """Test composition when no source chunks are available."""
        composer = ResponseComposer()
        
        context = CompositionContext(
            task_id="no_sources_test",
            input_text="What is enlightenment?",
            input_type="text",
            top_chunks=[],  # No source chunks
            agent_type="text_agent",
            model_name="test_model"
        )
        
        with patch.object(composer.rl_context, 'log_action'):
            result = await composer.compose_response(context)
            
            assert isinstance(result, CompositionResult)
            assert len(result.response_text) > 0
            # Should handle gracefully with fallback response
    
    @pytest.mark.asyncio 
    async def test_performance_timing(self):
        """Test that composition completes within reasonable time."""
        composer = ResponseComposer()
        
        context = CompositionContext(
            task_id="timing_test",
            input_text="Quick test query",
            input_type="text",
            top_chunks=[{"text": "Quick source", "score": 0.8}],
            agent_type="text_agent",
            model_name="test_model"
        )
        
        start_time = time.time()
        
        with patch.object(composer.rl_context, 'log_action'):
            result = await composer.compose_response(context)
        
        elapsed_time = time.time() - start_time
        
        # Should complete quickly (under 5 seconds for unit test)
        assert elapsed_time < 5.0
        assert result.processing_time > 0
        assert result.processing_time < 5.0

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])