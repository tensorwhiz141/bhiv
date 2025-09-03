import json
import os
from typing import Dict, Any, Optional
from utils.logger import get_logger
from reinforcement.agent_selector import AgentSelector
from reinforcement.rl_context import RLContext

logger = get_logger(__name__)

class AgentRegistry:
    """Registry for managing agent configurations and routing with RL support."""
    
    def __init__(self, config_file: str = "config/agent_configs.json"):
        self.config_file = config_file
        self.agents = {}
        self.agent_selector = AgentSelector()
        self.rl_context = RLContext()
        self.load_agents()
    
    def load_agents(self):
        """Load agent configurations from JSON file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.agents = json.load(f)
                logger.info(f"Loaded {len(self.agents)} agents from {self.config_file}")
            else:
                logger.warning(f"Agent config file not found: {self.config_file}")
                self.agents = {
                    "edumentor_agent": {
                        "connection_type": "python_module",
                        "module_path": "agents.stream_transformer_agent",
                        "class_name": "StreamTransformerAgent",
                        "id": "edumentor_agent",
                        "tags": ["summarize", "pdf", "text"],
                        "weight": 0.8
                    },
                    "archive_agent": {
                        "connection_type": "python_module",
                        "module_path": "agents.archive_agent",
                        "class_name": "ArchiveAgent",
                        "id": "archive_agent",
                        "tags": ["archive", "search", "pdf"],
                        "weight": 0.7
                    },
                    "knowledge_agent": {
                        "connection_type": "python_module",
                        "module_path": "agents.knowledge_agent",
                        "class_name": "KnowledgeAgent",
                        "id": "knowledge_agent",
                        "tags": ["semantic_search", "vedabase"],
                        "weight": 0.9
                    }
                }
                self.save_agents()
        except Exception as e:
            logger.error(f"Error loading agent configs: {e}")
            self.agents = {}
    
    def save_agents(self):
        """Save agent configurations to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.agents, f, indent=2)
            logger.info(f"Saved agent configurations to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving agent configs: {e}")
    
    def find_agent(self, task_context: Dict[str, Any]) -> str:
        """Find appropriate agent based on task context with RL support."""
        task_id = task_context.get("task_id", str(uuid.uuid4()))
        task = task_context.get("task", "summarize")
        
        # Use RL-based selection if enabled in settings
        from config.settings import settings
        use_rl = settings.get("use_rl", False)
        
        if use_rl:
            selected_agent = self.agent_selector.select_agent(task, task_context)
            self.rl_context.log_action(
                task_id=task_id,
                agent=selected_agent,
                model="none",
                action="select_agent",
                reward=0.0,  # Initial reward, updated later
                metadata={"task": task, "input_type": task_context.get("input_type", "text")}
            )
            if self.is_agent_available(selected_agent):
                logger.info(f"RL selected agent: {selected_agent} for task: {task}")
                return selected_agent
        
        # Fallback to deterministic routing
        if isinstance(task_context, str):
            # If passed a string, treat it as agent name
            return task_context
        
        # Extract agent from task context
        agent_name = task_context.get('model', task_context.get('agent', 'edumentor_agent'))
        input_type = task_context.get('input_type', 'text')
        tags = task_context.get('tags', [])
        
        # Route based on tags if provided
        if tags:
            for agent_name, config in self.agents.items():
                if any(tag in config.get("tags", []) for tag in tags):
                    self.rl_context.log_action(
                        task_id=task_id,
                        agent=agent_name,
                        model="none",
                        action="select_agent_by_tag",
                        reward=0.0,
                        metadata={"task": task, "tags": tags}
                    )
                    logger.info(f"Selected agent: {agent_name} for tags: {tags}")
                    return agent_name
        
        # Route based on input type if no specific agent
        if agent_name == 'edumentor_agent' and input_type != 'text':
            type_mapping = {
                'pdf': 'archive_agent',
                'image': 'image_agent',
                'audio': 'audio_agent',
                'semantic_search': 'knowledge_agent',
                'vedabase': 'knowledge_agent'
            }
            agent_name = type_mapping.get(input_type, 'edumentor_agent')
        
        self.rl_context.log_action(
            task_id=task_id,
            agent=agent_name,
            model="none",
            action="select_agent_by_type",
            reward=0.0,
            metadata={"task": task, "input_type": input_type}
        )
        logger.info(f"Selected agent: {agent_name} for task: {task}, input_type: {input_type}")
        return agent_name
    
    def get_agent_config(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get agent configuration by name."""
        return self.agents.get(agent_name)
    
    def get_agent(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get agent configuration by name (alias for get_agent_config)."""
        return self.get_agent_config(agent_name)
    
    def list_agents(self) -> Dict[str, Any]:
        """List all available agents."""
        return self.agents
    
    def register_agent(self, agent_name: str, config: Dict[str, Any]):
        """Register a new agent configuration."""
        self.agents[agent_name] = config
        self.save_agents()
        logger.info(f"Registered agent: {agent_name}")
    
    def is_agent_available(self, agent_name: str) -> bool:
        """Check if an agent is available."""
        return agent_name in self.agents

# Global agent registry instance
agent_registry = AgentRegistry()