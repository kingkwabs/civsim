"""
Base Agent Interface for CivSim.

All agents must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Optional

from environment.actions import AnyAction


class BaseAgent(ABC):
    """
    Abstract base class for all CivSim agents.
    
    Agents receive observations and produce actions.
    """
    
    def __init__(self, civ_id: int, name: Optional[str] = None):
        """
        Initialize the agent.
        
        Args:
            civ_id: The civilization ID this agent controls
            name: Optional name for the agent
        """
        self.civ_id = civ_id
        self.name = name or f"{self.__class__.__name__}_{civ_id}"
    
    @abstractmethod
    def select_action(
        self, 
        observation: dict, 
        valid_actions: list[AnyAction]
    ) -> AnyAction:
        """
        Select an action given the current observation.
        
        Args:
            observation: The observation dict for this civilization
            valid_actions: List of valid actions to choose from
            
        Returns:
            The selected action
        """
        pass
    
    def on_game_start(self, observation: dict):
        """Called at the start of a game. Override to initialize state."""
        pass
    
    def on_game_end(self, final_observation: dict, won: bool):
        """Called at the end of a game. Override to handle end-of-game logic."""
        pass
    
    def on_turn_start(self, observation: dict):
        """Called at the start of each turn. Override for turn-based logic."""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(civ_id={self.civ_id}, name='{self.name}')"
