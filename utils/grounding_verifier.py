"""
Grounding Verification Module

Verifies if generated content is properly grounded in source material
and computes grounding metrics for template selection decisions.
"""

import re
import spacy
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from utils.logger import get_logger
from config.template_config import GROUNDING_THRESHOLDS

logger = get_logger(__name__)

@dataclass
class GroundingMetrics:
    """Metrics for grounding verification."""
    source_overlap: float
    citation_density: float
    factual_consistency: float
    overall_score: float
    is_grounded: bool
    details: Dict[str, Any]

class GroundingVerifier:
    """Verifies content grounding against source material."""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def verify_grounding(self, 
                        generated_text: str, 
                        source_texts: List[str], 
                        template_requirements: Optional[Dict[str, Any]] = None) -> GroundingMetrics:
        """
        Verify if generated text is properly grounded in source material.
        
        Args:
            generated_text: The generated response text
            source_texts: List of source documents/excerpts
            template_requirements: Template-specific grounding requirements
            
        Returns:
            GroundingMetrics with verification results
        """
        if not source_texts:
            logger.warning("No source texts provided for grounding verification")
            return GroundingMetrics(
                source_overlap=0.0,
                citation_density=0.0,
                factual_consistency=0.0,
                overall_score=0.0,
                is_grounded=False,
                details={"error": "No source texts provided"}
            )
        
        # Compute grounding metrics
        source_overlap = self._compute_source_overlap(generated_text, source_texts)
        citation_density = self._compute_citation_density(generated_text)
        factual_consistency = self._compute_factual_consistency(generated_text, source_texts)
        
        # Weighted overall score
        overall_score = (
            source_overlap * 0.4 + 
            citation_density * 0.3 + 
            factual_consistency * 0.3
        )
        
        # Check against template requirements if provided
        requirements = template_requirements or {}
        min_overlap = requirements.get("min_source_overlap", GROUNDING_THRESHOLDS["fail_threshold"])
        min_citation = requirements.get("citation_density", 0.1)
        min_consistency = requirements.get("factual_consistency", 0.7)
        
        is_grounded = (
            source_overlap >= min_overlap and
            citation_density >= min_citation and
            factual_consistency >= min_consistency and
            overall_score >= GROUNDING_THRESHOLDS["fail_threshold"]
        )
        
        details = {
            "requirements_met": {
                "source_overlap": source_overlap >= min_overlap,
                "citation_density": citation_density >= min_citation,
                "factual_consistency": factual_consistency >= min_consistency
            },
            "thresholds": {
                "required_overlap": min_overlap,
                "required_citation": min_citation,
                "required_consistency": min_consistency
            },
            "source_count": len(source_texts),
            "generated_length": len(generated_text.split())
        }
        
        logger.info(f"Grounding verification: overlap={source_overlap:.3f}, "
                   f"citations={citation_density:.3f}, consistency={factual_consistency:.3f}, "
                   f"overall={overall_score:.3f}, grounded={is_grounded}")
        
        return GroundingMetrics(
            source_overlap=source_overlap,
            citation_density=citation_density,
            factual_consistency=factual_consistency,
            overall_score=overall_score,
            is_grounded=is_grounded,
            details=details
        )
    
    def _compute_source_overlap(self, generated_text: str, source_texts: List[str]) -> float:
        """Compute overlap between generated text and source material."""
        if not self.nlp:
            # Fallback to simple word overlap
            gen_words = set(generated_text.lower().split())
            source_words = set()
            for source in source_texts:
                source_words.update(source.lower().split())
            
            if not gen_words:
                return 0.0
            
            overlap = len(gen_words.intersection(source_words))
            return overlap / len(gen_words)
        
        # Use spaCy for better semantic overlap
        gen_doc = self.nlp(generated_text)
        source_docs = [self.nlp(source) for source in source_texts]
        
        # Extract entities and important tokens
        gen_entities = {ent.text.lower() for ent in gen_doc.ents}
        gen_tokens = {token.lemma_.lower() for token in gen_doc 
                     if not token.is_stop and not token.is_punct and len(token.text) > 2}
        
        source_entities = set()
        source_tokens = set()
        for doc in source_docs:
            source_entities.update({ent.text.lower() for ent in doc.ents})
            source_tokens.update({token.lemma_.lower() for token in doc 
                                if not token.is_stop and not token.is_punct and len(token.text) > 2})
        
        # Compute overlap scores
        entity_overlap = len(gen_entities.intersection(source_entities)) / max(len(gen_entities), 1)
        token_overlap = len(gen_tokens.intersection(source_tokens)) / max(len(gen_tokens), 1)
        
        # Weighted combination
        return entity_overlap * 0.6 + token_overlap * 0.4
    
    def _compute_citation_density(self, generated_text: str) -> float:
        """Compute density of citations in generated text."""
        # Look for citation patterns: [1], (Smith, 2020), etc.
        citation_patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'\([A-Za-z]+,?\s*\d{4}\)',  # (Smith, 2020)
            r'\([A-Za-z]+\s+et\s+al\.?,?\s*\d{4}\)',  # (Smith et al., 2020)
            r'(?i)according\s+to',  # According to
            r'(?i)as\s+stated\s+in',  # As stated in
            r'(?i)as\s+reported\s+by'  # As reported by
        ]
        
        total_citations = 0
        for pattern in citation_patterns:
            total_citations += len(re.findall(pattern, generated_text))
        
        words = len(generated_text.split())
        if words == 0:
            return 0.0
            
        return total_citations / words
    
    def _compute_factual_consistency(self, generated_text: str, source_texts: List[str]) -> float:
        """Compute factual consistency between generated text and sources."""
        if not self.nlp:
            # Fallback: simple heuristic based on shared facts
            gen_sentences = generated_text.split('.')
            source_content = ' '.join(source_texts).lower()
            
            consistent_sentences = 0
            for sentence in gen_sentences:
                if len(sentence.strip()) < 10:  # Skip very short sentences
                    continue
                # Simple check: if key words from sentence appear in source
                words = sentence.lower().split()
                key_words = [w for w in words if len(w) > 4]  # Focus on longer words
                if key_words and any(word in source_content for word in key_words[:3]):
                    consistent_sentences += 1
            
            return consistent_sentences / max(len([s for s in gen_sentences if len(s.strip()) >= 10]), 1)
        
        # Use spaCy for better factual consistency checking
        gen_doc = self.nlp(generated_text)
        source_docs = [self.nlp(source) for source in source_texts]
        
        # Extract claims (sentences with entities or important facts)
        gen_claims = []
        for sent in gen_doc.sents:
            if len(sent.ents) > 0 or any(token.pos_ in ['NOUN', 'VERB'] for token in sent):
                gen_claims.append(sent)
        
        if not gen_claims:
            return 0.5  # Neutral score if no clear claims
        
        # Check each claim against source material
        consistent_claims = 0
        for claim in gen_claims:
            claim_entities = {ent.text.lower() for ent in claim.ents}
            claim_tokens = {token.lemma_.lower() for token in claim 
                          if not token.is_stop and token.pos_ in ['NOUN', 'VERB', 'ADJ']}
            
            # Check if claim elements appear in any source
            for source_doc in source_docs:
                source_entities = {ent.text.lower() for ent in source_doc.ents}
                source_tokens = {token.lemma_.lower() for token in source_doc 
                               if not token.is_stop and token.pos_ in ['NOUN', 'VERB', 'ADJ']}
                
                entity_match = len(claim_entities.intersection(source_entities)) / max(len(claim_entities), 1)
                token_match = len(claim_tokens.intersection(source_tokens)) / max(len(claim_tokens), 1)
                
                if entity_match > 0.3 or token_match > 0.4:  # Reasonable match found
                    consistent_claims += 1
                    break
        
        return consistent_claims / len(gen_claims)

# Global instance
grounding_verifier = GroundingVerifier()

def verify_content_grounding(generated_text: str, 
                           source_texts: List[str], 
                           template_requirements: Optional[Dict[str, Any]] = None) -> GroundingMetrics:
    """Convenience function for grounding verification."""
    return grounding_verifier.verify_grounding(generated_text, source_texts, template_requirements)