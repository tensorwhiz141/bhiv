"""
Composer Module for BHIV Core

This module handles response composition with template logic, grounding enforcement,
and policy-based template selection with fallback mechanisms.

Main Components:
- ResponseComposer: Main composition orchestrator
- GRUComposer: Neural response generation (stub)
- CompositionResult: Response composition results
- CompositionContext: Composition input context

Key Features:
- Template-based response generation
- Grounding verification and enforcement
- Epsilon-greedy template selection
- Automatic fallback on grounding failure
- Token overlap enforcement
- RL integration for policy learning
"""

from .compose import (
    ResponseComposer,
    CompositionResult, 
    CompositionContext,
    compose_response_with_grounding,
    update_composition_reward
)

from .gru import (
    GRUResponseModel,
    GRUComposer,
    GRUTemplateAdapter,
    initialize_gru_system
)

__version__ = "1.0.0"
__author__ = "BHIV Core Team"

__all__ = [
    # Main composition classes
    "ResponseComposer",
    "CompositionResult",
    "CompositionContext",
    
    # Convenience functions
    "compose_response_with_grounding", 
    "update_composition_reward",
    
    # GRU components (stubs)
    "GRUResponseModel",
    "GRUComposer", 
    "GRUTemplateAdapter",
    "initialize_gru_system"
]