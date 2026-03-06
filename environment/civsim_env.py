"""
CivSim Environment - Main game environment with Gymnasium-style interface.

This is the main entry point for interacting with the game.
"""

from __future__ import annotations
from typing import Optional, Any
from dataclasses import dataclass
import numpy as np

from .map import HexMap, ResourceType
from .game_state import GameState, Civilization, DiplomaticStatus
from .actions import Action, AnyAction, get_valid_actions, EndTurnAction, NoOpAction
from .mechanics import GameMechanics


@dataclass
class StepResult:
    """Result of taking a step in the environment."""
    observations: dict[int, dict]  # Per-civilization observations
    rewards: dict[int, float]      # Per-civilization rewards
    terminated: bool               # Game over?
    truncated: bool                # Turn limit reached?
    info: dict                     # Additional information


class CivSimEnv:
    """
    CivSim: A multi-agent civilization strategy environment.
    
    This environment follows a turn-based structure where each civilization
    takes actions in sequence within a turn, then the turn ends and 
    end-of-turn processing occurs.
    
    Usage:
        env = CivSimEnv(num_civilizations=4)
        obs = env.reset()
        
        while not env.is_done():
            for civ_id in env.get_active_civs():
                valid_actions = env.get_valid_actions(civ_id)
                action = agent.select_action(obs[civ_id], valid_actions)
                obs, rewards, done, info = env.step(civ_id, action)
    """
    
    def __init__(
        self,
        map_width: int = 25,
        map_height: int = 25,
        num_civilizations: int = 4,
        max_turns: int = 500,
        visibility_radius: int = 3,
        seed: Optional[int] = None,
    ):
        """
        Initialize the environment.
        
        Args:
            map_width: Width of the hex map
            map_height: Height of the hex map
            num_civilizations: Number of civilizations (players)
            max_turns: Maximum turns before game ends
            visibility_radius: How far units can see (fog of war)
            seed: Random seed for reproducibility
        """
        self.map_width = map_width
        self.map_height = map_height
        self.num_civilizations = num_civilizations
        self.max_turns = max_turns
        self.visibility_radius = visibility_radius
        self.seed = seed
        
        # Will be initialized in reset()
        self.state: Optional[GameState] = None
        self.mechanics: Optional[GameMechanics] = None
        
        # Track which civs have acted this turn
        self._civs_acted_this_turn: set[int] = set()
        
        # Action history for analysis
        self.action_history: list[dict] = []
        
        # Reward tracking
        self._last_scores: dict[int, int] = {}
    
    def reset(self, seed: Optional[int] = None) -> dict[int, dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Optional new seed
            
        Returns:
            Initial observations for all civilizations
        """
        if seed is not None:
            self.seed = seed
        
        self.state = GameState(
            map_width=self.map_width,
            map_height=self.map_height,
            num_civilizations=self.num_civilizations,
            max_turns=self.max_turns,
            seed=self.seed,
        )
        
        self.mechanics = GameMechanics(self.state)
        self._civs_acted_this_turn = set()
        self.action_history = []
        
        # Initialize score tracking
        self._last_scores = {
            civ_id: self.state.get_civ_score(civ_id)
            for civ_id in self.state.civilizations
        }
        
        return self._get_all_observations()
    
    def step(self, civ_id: int, action: Action) -> StepResult:
        """
        Take a single action for a civilization.
        
        Args:
            civ_id: Which civilization is acting
            action: The action to take
            
        Returns:
            StepResult with observations, rewards, termination status, and info
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        if self.state.game_over:
            return StepResult(
                observations=self._get_all_observations(),
                rewards={cid: 0.0 for cid in self.state.civilizations},
                terminated=True,
                truncated=False,
                info={'error': 'Game already over'}
            )
        
        # Validate action
        if action.civ_id != civ_id:
            action.civ_id = civ_id  # Fix mismatched civ_id
        
        # Execute the action
        result = self.mechanics.execute_action(action)
        
        # Log action
        self.action_history.append({
            'turn': self.state.current_turn,
            'civ_id': civ_id,
            'action': action.describe(),
            'result': result,
        })
        
        # Mark civ as having acted if they ended their turn
        if isinstance(action, EndTurnAction):
            self._civs_acted_this_turn.add(civ_id)
        
        # Check if all civs have ended their turn
        alive_civs = {c.id for c in self.state.get_alive_civilizations()}
        if self._civs_acted_this_turn >= alive_civs:
            # Process end of turn
            self.mechanics.process_turn_end()
            self._civs_acted_this_turn = set()
        
        # Calculate rewards
        rewards = self._calculate_rewards()
        
        # Check termination
        terminated = self.state.game_over
        truncated = self.state.current_turn >= self.max_turns
        
        return StepResult(
            observations=self._get_all_observations(),
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info={
                'turn': self.state.current_turn,
                'action_result': result,
                'winner': self.state.winner,
            }
        )
    
    def step_simultaneous(self, actions: dict[int, Action]) -> StepResult:
        """
        Take actions for all civilizations simultaneously.
        
        This is an alternative interface where all civs submit actions at once.
        
        Args:
            actions: Dict mapping civ_id to action
            
        Returns:
            StepResult
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Execute all actions
        results = {}
        for civ_id, action in actions.items():
            action.civ_id = civ_id
            results[civ_id] = self.mechanics.execute_action(action)
            
            self.action_history.append({
                'turn': self.state.current_turn,
                'civ_id': civ_id,
                'action': action.describe(),
                'result': results[civ_id],
            })
        
        # Process end of turn
        self.mechanics.process_turn_end()
        
        # Calculate rewards
        rewards = self._calculate_rewards()
        
        terminated = self.state.game_over
        truncated = self.state.current_turn >= self.max_turns
        
        return StepResult(
            observations=self._get_all_observations(),
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info={
                'turn': self.state.current_turn,
                'action_results': results,
                'winner': self.state.winner,
            }
        )
    
    def get_valid_actions(self, civ_id: int) -> list[AnyAction]:
        """Get all valid actions for a civilization."""
        if self.state is None:
            return []
        return get_valid_actions(self.state, civ_id)
    
    def get_observation(self, civ_id: int) -> dict:
        """Get observation for a specific civilization."""
        if self.state is None:
            return {}
        return self.state.to_observation(civ_id, self.visibility_radius)
    
    def _get_all_observations(self) -> dict[int, dict]:
        """Get observations for all civilizations."""
        if self.state is None:
            return {}
        return {
            civ_id: self.state.to_observation(civ_id, self.visibility_radius)
            for civ_id in self.state.civilizations
        }
    
    def _calculate_rewards(self) -> dict[int, float]:
        """
        Calculate rewards for all civilizations.
        
        Reward is based on score change since last step.
        """
        rewards = {}
        
        for civ_id in self.state.civilizations:
            civ = self.state.civilizations[civ_id]
            
            if not civ.is_alive:
                # Dead civs get large negative reward
                rewards[civ_id] = -100.0
                continue
            
            current_score = self.state.get_civ_score(civ_id)
            last_score = self._last_scores.get(civ_id, 0)
            
            # Reward is score delta
            rewards[civ_id] = float(current_score - last_score)
            
            # Bonus for winning
            if self.state.game_over and self.state.winner == civ_id:
                rewards[civ_id] += 500.0
            
            self._last_scores[civ_id] = current_score
        
        return rewards
    
    def get_active_civs(self) -> list[int]:
        """Get list of civilizations that haven't acted this turn."""
        if self.state is None:
            return []
        alive = {c.id for c in self.state.get_alive_civilizations()}
        return [cid for cid in alive if cid not in self._civs_acted_this_turn]
    
    def is_done(self) -> bool:
        """Check if game is over."""
        if self.state is None:
            return True
        return self.state.game_over
    
    def get_scores(self) -> dict[int, int]:
        """Get current scores for all civilizations."""
        if self.state is None:
            return {}
        return {
            civ_id: self.state.get_civ_score(civ_id)
            for civ_id in self.state.civilizations
        }
    
    def get_winner(self) -> Optional[int]:
        """Get the winning civilization ID, or None if game not over."""
        if self.state is None or not self.state.game_over:
            return None
        return self.state.winner
    
    def render_ascii(self) -> str:
        """
        Render a simple ASCII representation of the map.
        
        Useful for debugging.
        """
        if self.state is None:
            return "No game state"
        
        lines = []
        terrain_chars = {
            'PLAINS': '.',
            'FOREST': 'T',
            'MOUNTAINS': '^',
            'WATER': '~',
            'FERTILE': '*',
            'DESERT': 's',
        }
        
        for r in range(self.state.map_height):
            row = ""
            # Offset for hex grid appearance
            if r % 2 == 1:
                row = " "
            
            for q in range(self.state.map_width):
                tile = self.state.map.get_tile(q, r)
                
                # Check for units
                units = self.state.get_units_at(q, r)
                if units:
                    row += str(units[0].owner)
                # Check for cities
                elif any(c.q == q and c.r == r for c in self.state.cities.values()):
                    city = next(c for c in self.state.cities.values() if c.q == q and c.r == r)
                    row += 'C'
                # Show terrain
                elif tile:
                    row += terrain_chars.get(tile.terrain.name, '?')
                else:
                    row += ' '
                
                row += " "
            
            lines.append(row)
        
        # Add legend
        lines.append("")
        lines.append(f"Turn: {self.state.current_turn}")
        for civ_id, civ in self.state.civilizations.items():
            status = "ALIVE" if civ.is_alive else "DEAD"
            score = self.state.get_civ_score(civ_id)
            lines.append(f"  {civ_id}: {civ.name} - {status} - Score: {score}")
        
        return "\n".join(lines)
    
    def get_state_for_clone(self) -> GameState:
        """Get the game state (for MCTS or other planning algorithms)."""
        return self.state
    
    def set_state(self, state: GameState):
        """Set the game state (for MCTS or loading)."""
        self.state = state
        self.mechanics = GameMechanics(self.state)


# Convenience function for quick testing
def make_env(**kwargs) -> CivSimEnv:
    """Create a CivSim environment with default or custom settings."""
    return CivSimEnv(**kwargs)
