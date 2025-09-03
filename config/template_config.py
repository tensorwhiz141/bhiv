"""
Template Configuration for BHIV Response Generation

Defines different response templates with varying levels of extractiveness,
citation requirements, and response styles.
"""

from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class ResponseTemplate:
    """Configuration for response generation templates."""
    template_id: str
    name: str
    description: str
    max_length: int
    min_citations: int
    extractive_ratio: float  # 0.0 = fully generative, 1.0 = fully extractive
    style: str
    grounding_requirements: Dict[str, Any]
    reward_weight: float = 1.0

# Template definitions with increasing extractiveness for fallback
RESPONSE_TEMPLATES = {
    "generative_standard": ResponseTemplate(
        template_id="generative_standard",
        name="Standard Generative",
        description="Standard generative response with moderate grounding",
        max_length=200,
        min_citations=2,
        extractive_ratio=0.3,
        style="conversational",
        grounding_requirements={
            "min_source_overlap": 0.3,  # Lowered from 0.4
            "citation_density": 0.05,   # Lowered from 0.1
            "factual_consistency": 0.6  # Lowered from 0.7
        }
    ),
    
    "balanced_hybrid": ResponseTemplate(
        template_id="balanced_hybrid",
        name="Balanced Hybrid",
        description="Balanced mix of generative and extractive content",
        max_length=150,
        min_citations=3,
        extractive_ratio=0.5,
        style="academic",
        grounding_requirements={
            "min_source_overlap": 0.4,  # Lowered from 0.6
            "citation_density": 0.1,    # Lowered from 0.2
            "factual_consistency": 0.7  # Lowered from 0.8
        }
    ),
    
    "extractive_heavy": ResponseTemplate(
        template_id="extractive_heavy",
        name="Extractive Heavy",
        description="Heavily extractive with high citation density (fallback)",
        max_length=100,
        min_citations=4,
        extractive_ratio=0.8,
        style="scholarly",
        grounding_requirements={
            "min_source_overlap": 0.6,  # Lowered from 0.8
            "citation_density": 0.2,    # Lowered from 0.4
            "factual_consistency": 0.8  # Lowered from 0.9
        },
        reward_weight=0.8  # Lower reward weight as fallback
    )
}

# Default template selection order
DEFAULT_TEMPLATE_ORDER = [
    "generative_standard",
    "balanced_hybrid", 
    "extractive_heavy"
]

# Template selection thresholds
GROUNDING_THRESHOLDS = {
    "fail_threshold": 0.4,  # Lowered from 0.5 - Below this, grounding fails
    "warning_threshold": 0.6,  # Below this, consider fallback
    "good_threshold": 0.8   # Above this, grounding is good
}

# Epsilon-greedy policy configuration
TEMPLATE_POLICY_CONFIG = {
    "epsilon": 0.2,  # 20% exploration
    "epsilon_decay": 0.995,
    "min_epsilon": 0.05,
    "fallback_threshold": 0.6,  # Reward threshold to trigger fallback
    "memory_size": 1000
}

def get_template_by_id(template_id: str) -> ResponseTemplate:
    """Get template configuration by ID."""
    return RESPONSE_TEMPLATES.get(template_id)

def get_default_template() -> ResponseTemplate:
    """Get the default template."""
    return RESPONSE_TEMPLATES["generative_standard"]

def get_fallback_template() -> ResponseTemplate:
    """Get the extractive fallback template."""
    return RESPONSE_TEMPLATES["extractive_heavy"]

def get_ordered_templates() -> List[ResponseTemplate]:
    """Get templates in fallback order."""
    return [RESPONSE_TEMPLATES[tid] for tid in DEFAULT_TEMPLATE_ORDER]