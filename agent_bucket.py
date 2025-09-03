import logging
from utils.logger import get_logger

logger = get_logger(__name__)

def store_agent_data(data: dict):
    """JSON structure for agents."""
    logger.info(f"Storing agent data: {data}")
    return data