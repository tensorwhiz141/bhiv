import pytest
import asyncio
from integration.nipun_adapter import (
    map_output_to_learning,
    map_multi_input_to_learning,
    extract_bloom_taxonomy_level,
    extract_subject_tags,
    aggregate_multi_input_metadata,
    store_nlo_in_mongodb,
    get_nlos_by_subject,
    get_nlos_by_task_id
)
from reinforcement.replay_buffer import ReplayBuffer
from reinforcement.reward_functions import get_reward_from_output
import uuid

@pytest.fixture
def replay_buffer():
    return ReplayBuffer(buffer_file="logs/test_learning_log.json")

def test_map_output_to_nlo_success(replay_buffer):
    task_id = str(uuid.uuid4())
    output = {
        "result": "Sample summary",
        "model": "edumentor_agent",
        "keywords": ["summarize"],
        "input": "Sample input",
        "content_type": "text",
        "confidence": 0.9,
        "difficulty": "easy"
    }
    nlo = map_output_to_learning(output, task_id)
    assert nlo["summary"] == "Sample summary"
    assert nlo["model"] == "edumentor_agent"
    assert nlo["tags"] == ["summarize"]
    assert nlo["content_type"] == "text"
    assert nlo["confidence"] == 0.9
    assert nlo["difficulty"] == "easy"
    assert nlo["subject_tag"] == "summarize"
    assert "reward" in nlo
    assert len(replay_buffer.buffer) > 0

def test_map_output_to_nlo_missing_metadata(replay_buffer):
    task_id = str(uuid.uuid4())
    output = {
        "result": "Sample summary",
        "model": "edumentor_agent",
        "input": "Sample input"
    }
    nlo = map_output_to_learning(output, task_id)
    assert nlo["summary"] == "Sample summary"
    assert nlo["model"] == "edumentor_agent"
    assert nlo["tags"] == []
    assert nlo["content_type"] == "text"
    assert nlo["confidence"] == 0.8
    assert nlo["difficulty"] == "medium"
    assert nlo["subject_tag"] == "general"
    assert "reward" in nlo
    assert len(replay_buffer.buffer) > 0

def test_map_output_to_nlo_error(replay_buffer):
    task_id = str(uuid.uuid4())
    output = {
        "error": "Processing failed",
        "status": 500,
        "input": "Sample input",
        "model": "edumentor_agent"
    }
    nlo = map_output_to_learning(output, task_id)
    assert "error" in nlo
    assert nlo["status"] == 500
    assert len(replay_buffer.buffer) > 0

def test_extract_bloom_taxonomy_level():
    """Test Bloom's taxonomy level extraction."""
    # Test different levels
    assert extract_bloom_taxonomy_level("Please recall the main points") == "remember"
    assert extract_bloom_taxonomy_level("Explain the concept of AI") == "understand"
    assert extract_bloom_taxonomy_level("Apply this formula to solve") == "apply"
    assert extract_bloom_taxonomy_level("Analyze the differences between") == "analyze"
    assert extract_bloom_taxonomy_level("Evaluate the effectiveness of") == "evaluate"
    assert extract_bloom_taxonomy_level("Design a new system") == "create"
    assert extract_bloom_taxonomy_level("") == "remember"

def test_extract_subject_tags():
    """Test subject tag extraction."""
    text = "This is about artificial intelligence and machine learning in education"
    tags = extract_subject_tags(text)
    assert isinstance(tags, list)
    assert len(tags) > 0

def test_aggregate_multi_input_metadata():
    """Test metadata aggregation for multi-input scenarios."""
    outputs = [
        {"result": "PDF analysis", "keywords": ["pdf", "analysis"], "confidence": 0.9, "content_type": "pdf"},
        {"result": "Image description", "keywords": ["image", "visual"], "confidence": 0.8, "content_type": "image"},
        {"result": "Audio transcription", "keywords": ["audio", "speech"], "confidence": 0.7, "content_type": "audio"}
    ]

    aggregated = aggregate_multi_input_metadata(outputs)

    assert aggregated["input_count"] == 3
    assert aggregated["average_confidence"] == 0.8
    assert "pdf analysis image description audio transcription" in aggregated["combined_result"].lower()
    assert set(aggregated["content_types"]) == {"pdf", "image", "audio"}
    assert len(aggregated["combined_keywords"]) >= 6

def test_map_multi_input_to_learning():
    """Test multi-input NLO mapping."""
    task_id = str(uuid.uuid4())
    outputs = [
        {
            "result": "This PDF analyzes artificial intelligence trends in education",
            "model": "archive_agent",
            "keywords": ["AI", "education", "trends"],
            "content_type": "pdf",
            "confidence": 0.9,
            "agent": "archive_agent",
            "processing_time": 2.0
        },
        {
            "result": "Image shows AI neural network diagram",
            "model": "image_agent",
            "keywords": ["AI", "neural", "diagram"],
            "content_type": "image",
            "confidence": 0.8,
            "agent": "image_agent",
            "processing_time": 1.5
        }
    ]

    nlo = map_output_to_learning(outputs, task_id)

    assert nlo["task_id"] == task_id
    assert nlo["is_multi_input"] == True
    assert nlo["input_count"] == 2
    assert "artificial intelligence" in nlo["summary"].lower()
    assert len(nlo["models"]) == 2
    assert "archive_agent" in nlo["agents"]
    assert "image_agent" in nlo["agents"]
    assert nlo["primary_content_type"] == "multi"
    assert nlo["total_processing_time"] == 3.5

def test_map_multi_input_empty_list():
    """Test multi-input mapping with empty list."""
    task_id = str(uuid.uuid4())
    nlo = map_output_to_learning([], task_id)

    assert "error" in nlo
    assert nlo["task_id"] == task_id

@pytest.mark.asyncio
async def test_store_and_retrieve_nlo():
    """Test MongoDB storage and retrieval."""
    task_id = str(uuid.uuid4())
    test_nlo = {
        "summary": "Test NLO for storage",
        "subject_tag": "test_subject",
        "bloom_level": "understand",
        "confidence": 0.9
    }

    # Test storage
    success = await store_nlo_in_mongodb(test_nlo, task_id)
    assert success == True

    # Test retrieval by task_id
    retrieved_nlo = await get_nlos_by_task_id(task_id)
    assert retrieved_nlo is not None
    assert retrieved_nlo.get("task_id") == task_id

    # Test retrieval by subject
    subject_nlos = await get_nlos_by_subject("test_subject", limit=5)
    assert isinstance(subject_nlos, list)