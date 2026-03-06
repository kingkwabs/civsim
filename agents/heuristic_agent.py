"""
Heuristic Agents - Rule-based agents with strategic personalities.

These agents use hand-crafted evaluation functions and decision rules
rather than learned policies.
"""

import random
from typing import Optional
from enum import Enum, auto

from .base_agent import BaseAgent
from environment.actions import (
    AnyAction, ActionType, EndTurnAction, MoveAction, AttackAction,
    GatherAction, FoundCityAction, ProduceAction, DeclareWarAction,
    ProposeAllianceAction, ProposeTradeAction, AcceptTradeAction
)
from environment.map import ResourceType


class AgentPersonality(Enum):
    """Strategic personalities for heuristic agents."""
    MILITARIST = auto()   # Aggressive expansion and combat
    ECONOMIST = auto()    # Resource accumulation and trade
    DIPLOMAT = auto()     # Alliance building and cooperation
    BALANCED = auto()     # Mix of all strategies


class HeuristicAgent(BaseAgent):
    """
    A heuristic-based agent that uses evaluation functions
    and strategic rules to make decisions.
    """
    
    def __init__(
        self,
        civ_id: int,
        personality: AgentPersonality = AgentPersonality.BALANCED,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the heuristic agent.
        
        Args:
            civ_id: The civilization ID this agent controls
            personality: Strategic personality affecting decisions
            name: Optional name for the agent
            seed: Random seed for tie-breaking
        """
        super().__init__(civ_id, name or f"{personality.name.title()}Agent_{civ_id}")
        self.personality = personality
        self.rng = random.Random(seed)
        
        # Weights for different objectives based on personality
        self._set_personality_weights()
    
    def _set_personality_weights(self):
        """Set evaluation weights based on personality."""
        if self.personality == AgentPersonality.MILITARIST:
            self.weights = {
                'military_strength': 2.0,
                'territory': 1.5,
                'resources': 0.5,
                'cities': 1.0,
                'diplomacy': 0.2,
                'aggression': 1.5,
            }
        elif self.personality == AgentPersonality.ECONOMIST:
            self.weights = {
                'military_strength': 0.5,
                'territory': 1.0,
                'resources': 2.0,
                'cities': 1.5,
                'diplomacy': 1.0,
                'aggression': 0.2,
            }
        elif self.personality == AgentPersonality.DIPLOMAT:
            self.weights = {
                'military_strength': 0.5,
                'territory': 0.8,
                'resources': 1.0,
                'cities': 1.0,
                'diplomacy': 2.0,
                'aggression': 0.1,
            }
        else:  # BALANCED
            self.weights = {
                'military_strength': 1.0,
                'territory': 1.0,
                'resources': 1.0,
                'cities': 1.0,
                'diplomacy': 1.0,
                'aggression': 0.5,
            }
    
    def select_action(
        self,
        observation: dict,
        valid_actions: list[AnyAction]
    ) -> AnyAction:
        """Select the best action according to heuristic evaluation."""
        if not valid_actions:
            return EndTurnAction(civ_id=self.civ_id)
        
        # Categorize actions
        categorized = self._categorize_actions(valid_actions)
        
        # Priority-based action selection
        action = self._select_by_priority(observation, categorized)
        
        if action:
            return action
        
        # Default: end turn
        end_actions = categorized.get('end_turn', [])
        if end_actions:
            return end_actions[0]
        
        return self.rng.choice(valid_actions)
    
    def _categorize_actions(self, actions: list[AnyAction]) -> dict[str, list[AnyAction]]:
        """Categorize actions by type for easier processing."""
        categories = {
            'move': [],
            'attack': [],
            'gather': [],
            'found_city': [],
            'produce': [],
            'diplomacy': [],
            'trade': [],
            'end_turn': [],
        }
        
        for action in actions:
            if isinstance(action, MoveAction):
                categories['move'].append(action)
            elif isinstance(action, AttackAction):
                categories['attack'].append(action)
            elif isinstance(action, GatherAction):
                categories['gather'].append(action)
            elif isinstance(action, FoundCityAction):
                categories['found_city'].append(action)
            elif isinstance(action, ProduceAction):
                categories['produce'].append(action)
            elif isinstance(action, (DeclareWarAction, ProposeAllianceAction)):
                categories['diplomacy'].append(action)
            elif isinstance(action, (ProposeTradeAction, AcceptTradeAction)):
                categories['trade'].append(action)
            elif isinstance(action, EndTurnAction):
                categories['end_turn'].append(action)
        
        return categories
    
    def _select_by_priority(
        self,
        observation: dict,
        categorized: dict[str, list[AnyAction]]
    ) -> Optional[AnyAction]:
        """Select action based on strategic priorities."""
        
        # Priority 1: Attack if at war and have opportunity
        if self.weights['aggression'] > 0.3:
            attacks = categorized['attack']
            if attacks:
                return self._best_attack(observation, attacks)
        
        # Priority 2: Accept favorable trades
        trades = [a for a in categorized['trade'] if isinstance(a, AcceptTradeAction)]
        if trades:
            best_trade = self._evaluate_trades(observation, trades)
            if best_trade:
                return best_trade
        
        # Priority 3: Found cities if we have settlers
        if categorized['found_city']:
            return categorized['found_city'][0]
        
        # Priority 4: Set production if cities aren't producing
        production = self._select_production(observation, categorized['produce'])
        if production:
            return production
        
        # Priority 5: Gather resources
        if categorized['gather'] and self.weights['resources'] > 0.5:
            return self.rng.choice(categorized['gather'])
        
        # Priority 6: Diplomatic actions based on personality
        if self.weights['diplomacy'] > 0.5:
            diplo_action = self._select_diplomatic_action(observation, categorized['diplomacy'])
            if diplo_action:
                return diplo_action
        
        # Priority 7: Move units strategically
        if categorized['move']:
            return self._best_move(observation, categorized['move'])
        
        # Priority 8: Declare war if militarist and have advantage
        if self.weights['aggression'] > 1.0:
            war_action = self._consider_war(observation, categorized['diplomacy'])
            if war_action:
                return war_action
        
        return None
    
    def _best_attack(
        self,
        observation: dict,
        attacks: list[AttackAction]
    ) -> Optional[AttackAction]:
        """Select the best attack action."""
        if not attacks:
            return None
        
        # For now, just pick randomly among attacks
        # TODO: Evaluate based on target value and risk
        return self.rng.choice(attacks)
    
    def _best_move(
        self,
        observation: dict,
        moves: list[MoveAction]
    ) -> Optional[MoveAction]:
        """Select the best move action."""
        if not moves:
            return None
        
        # Score each move
        scored_moves = []
        own_units = {(u['q'], u['r']) for u in observation.get('own_units', [])}
        own_cities = {(c['q'], c['r']) for c in observation.get('own_cities', [])}
        
        for move in moves:
            score = 0.0
            target = (move.target_q, move.target_r)
            
            # Find the unit
            unit = None
            for u in observation.get('own_units', []):
                if u['id'] == move.unit_id:
                    unit = u
                    break
            
            if unit is None:
                continue
            
            # Score based on exploration (tiles we don't own)
            visible_tiles = observation.get('visible_tiles', [])
            for tile in visible_tiles:
                if tile['q'] == target[0] and tile['r'] == target[1]:
                    if tile['owner'] != self.civ_id and tile['owner'] is not None:
                        score += 2.0  # Capturing territory is good
                    elif tile['owner'] is None:
                        score += 1.0  # Exploring new territory
                    
                    # Bonus for resource-rich tiles
                    if tile.get('yields'):
                        score += sum(tile['yields'].values()) * 0.1
            
            # Penalty for moving away from cities if militarist defending
            if self.personality == AgentPersonality.MILITARIST:
                for city_pos in own_cities:
                    # Simplified distance check
                    if abs(target[0] - city_pos[0]) + abs(target[1] - city_pos[1]) < 3:
                        score += 0.5  # Stay near cities
            
            # Small random factor for variety
            score += self.rng.random() * 0.5
            
            scored_moves.append((score, move))
        
        if not scored_moves:
            return self.rng.choice(moves) if moves else None
        
        # Return best move
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return scored_moves[0][1]
    
    def _select_production(
        self,
        observation: dict,
        produce_actions: list[ProduceAction]
    ) -> Optional[ProduceAction]:
        """Select what to produce based on personality and current state."""
        if not produce_actions:
            return None
        
        # Count our current units by type
        unit_counts = {}
        for unit in observation.get('own_units', []):
            utype = unit['type']
            unit_counts[utype] = unit_counts.get(utype, 0) + 1
        
        # Determine priority production based on personality
        if self.personality == AgentPersonality.MILITARIST:
            priority = ['warrior', 'archer', 'scout', 'worker', 'settler']
        elif self.personality == AgentPersonality.ECONOMIST:
            priority = ['worker', 'settler', 'scout', 'warrior', 'archer']
        elif self.personality == AgentPersonality.DIPLOMAT:
            priority = ['settler', 'worker', 'scout', 'warrior', 'archer']
        else:  # BALANCED
            priority = ['worker', 'warrior', 'settler', 'scout', 'archer']
        
        # Find available production matching priority
        available_types = {a.production_type for a in produce_actions}
        
        for prod_type in priority:
            if prod_type in available_types:
                # Check if we need more of this type
                current_count = unit_counts.get(prod_type.upper(), 0)
                
                # Limits based on personality
                limits = {
                    'worker': 3 if self.personality == AgentPersonality.ECONOMIST else 2,
                    'warrior': 5 if self.personality == AgentPersonality.MILITARIST else 3,
                    'archer': 4 if self.personality == AgentPersonality.MILITARIST else 2,
                    'settler': 2,
                    'scout': 2,
                }
                
                if current_count < limits.get(prod_type, 3):
                    # Find the action for this production type
                    for action in produce_actions:
                        if action.production_type == prod_type:
                            return action
        
        return None
    
    def _evaluate_trades(
        self,
        observation: dict,
        trade_actions: list[AcceptTradeAction]
    ) -> Optional[AcceptTradeAction]:
        """Evaluate and possibly accept trade offers."""
        if not trade_actions:
            return None
        
        # Look at incoming offers
        incoming = observation.get('incoming_offers', [])
        if not incoming:
            return None
        
        resources = observation.get('resources', {})
        
        for offer in incoming:
            # Simple evaluation: accept if we're getting more value than giving
            offer_value = sum(offer.get('offer', {}).values())
            request_value = sum(offer.get('request', {}).values())
            
            # Check if we can afford the request
            can_afford = all(
                resources.get(rt, 0) >= amt
                for rt, amt in offer.get('request', {}).items()
            )
            
            # Accept if favorable and affordable
            if can_afford and offer_value >= request_value * 0.8:
                for action in trade_actions:
                    if action.trade_id == offer['id']:
                        return action
        
        return None
    
    def _select_diplomatic_action(
        self,
        observation: dict,
        diplo_actions: list[AnyAction]
    ) -> Optional[AnyAction]:
        """Select a diplomatic action based on trust and personality."""
        if not diplo_actions:
            return None
        
        trust_scores = observation.get('trust_scores', {})
        relations = observation.get('relations', {})
        
        for action in diplo_actions:
            if isinstance(action, ProposeAllianceAction):
                target = action.target_civ
                trust = trust_scores.get(str(target), 0.5)
                relation = relations.get(str(target), 'NEUTRAL')
                
                # Diplomats more likely to propose alliances
                threshold = 0.4 if self.personality == AgentPersonality.DIPLOMAT else 0.6
                
                if trust > threshold and relation == 'NEUTRAL':
                    return action
        
        return None
    
    def _consider_war(
        self,
        observation: dict,
        diplo_actions: list[AnyAction]
    ) -> Optional[DeclareWarAction]:
        """Consider declaring war based on relative strength."""
        war_actions = [a for a in diplo_actions if isinstance(a, DeclareWarAction)]
        if not war_actions:
            return None
        
        # Only militarists actively seek war
        if self.personality != AgentPersonality.MILITARIST:
            return None
        
        # Count our military
        our_military = sum(
            1 for u in observation.get('own_units', [])
            if u['type'] in ['WARRIOR', 'ARCHER']
        )
        
        # Need enough military before starting wars
        if our_military < 3:
            return None
        
        # Randomly declare war sometimes
        if self.rng.random() < 0.1:
            return self.rng.choice(war_actions)
        
        return None


# Convenience factory functions
def create_militarist(civ_id: int, seed: Optional[int] = None) -> HeuristicAgent:
    """Create a militarist agent."""
    return HeuristicAgent(civ_id, AgentPersonality.MILITARIST, seed=seed)


def create_economist(civ_id: int, seed: Optional[int] = None) -> HeuristicAgent:
    """Create an economist agent."""
    return HeuristicAgent(civ_id, AgentPersonality.ECONOMIST, seed=seed)


def create_diplomat(civ_id: int, seed: Optional[int] = None) -> HeuristicAgent:
    """Create a diplomat agent."""
    return HeuristicAgent(civ_id, AgentPersonality.DIPLOMAT, seed=seed)


def create_balanced(civ_id: int, seed: Optional[int] = None) -> HeuristicAgent:
    """Create a balanced agent."""
    return HeuristicAgent(civ_id, AgentPersonality.BALANCED, seed=seed)
