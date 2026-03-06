"""
Random Agent - Baseline agent that selects random valid actions.
"""

import random
from typing import Optional

from .base_agent import BaseAgent
from environment.actions import AnyAction, EndTurnAction


class RandomAgent(BaseAgent):
    """
    An agent that selects actions uniformly at random.
    
    Useful as a baseline and for testing the environment.
    """
    
    def __init__(
        self, 
        civ_id: int, 
        name: Optional[str] = None,
        end_turn_probability: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Initialize the random agent.
        
        Args:
            civ_id: The civilization ID this agent controls
            name: Optional name for the agent
            end_turn_probability: Probability of ending turn vs taking action
            seed: Random seed for reproducibility
        """
        super().__init__(civ_id, name)
        self.end_turn_probability = end_turn_probability
        self.rng = random.Random(seed)
    
    def select_action(
        self, 
        observation: dict, 
        valid_actions: list[AnyAction]
    ) -> AnyAction:
        """Select a random valid action."""
        if not valid_actions:
            return EndTurnAction(civ_id=self.civ_id)
        
        # Sometimes just end turn
        if self.rng.random() < self.end_turn_probability:
            end_actions = [a for a in valid_actions if isinstance(a, EndTurnAction)]
            if end_actions:
                return end_actions[0]
        
        return self.rng.choice(valid_actions)
