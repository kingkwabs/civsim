"""
Action definitions for CivSim.

All actions that agents can take in the game.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Union
from abc import ABC, abstractmethod

from .map import ResourceType


class ActionType(Enum):
    """Types of actions agents can take."""
    # Unit actions
    MOVE = auto()
    ATTACK = auto()
    FORTIFY = auto()
    
    # Worker actions
    GATHER = auto()
    BUILD_IMPROVEMENT = auto()
    
    # Settler actions
    FOUND_CITY = auto()
    
    # City actions
    PRODUCE = auto()
    BUY = auto()
    
    # Diplomatic actions
    PROPOSE_TRADE = auto()
    ACCEPT_TRADE = auto()
    REJECT_TRADE = auto()
    PROPOSE_ALLIANCE = auto()
    ACCEPT_ALLIANCE = auto()
    REJECT_ALLIANCE = auto()
    BREAK_ALLIANCE = auto()
    DECLARE_WAR = auto()
    PROPOSE_PEACE = auto()
    ACCEPT_PEACE = auto()
    
    # Meta actions
    END_TURN = auto()
    NO_OP = auto()


@dataclass
class Action(ABC):
    """Base class for all actions."""
    action_type: ActionType = ActionType.NO_OP
    civ_id: int = 0  # Which civilization is taking this action
    
    @abstractmethod
    def is_valid(self, game_state) -> bool:
        """Check if this action is valid in the current game state."""
        pass
    
    @abstractmethod
    def describe(self) -> str:
        """Human-readable description of the action."""
        pass


@dataclass
class MoveAction(Action):
    """Move a unit to a new position."""
    action_type: ActionType = ActionType.MOVE
    unit_id: int = 0
    target_q: int = 0
    target_r: int = 0
    
    def is_valid(self, game_state) -> bool:
        unit = game_state.units.get(self.unit_id)
        if unit is None or not unit.is_alive():
            return False
        if unit.owner != self.civ_id:
            return False
        if not unit.can_move():
            return False
        
        # Check target is adjacent and passable
        target_tile = game_state.map.get_tile(self.target_q, self.target_r)
        if target_tile is None:
            return False
        
        current_tile = game_state.map.get_tile(unit.q, unit.r)
        if current_tile.distance_to(target_tile) > 1:
            return False
        
        civ = game_state.civilizations.get(self.civ_id)
        has_mountain_tech = 'mountain_crossing' in civ.technologies if civ else False
        
        if not target_tile.is_passable(has_mountain_tech):
            return False
        
        return True
    
    def describe(self) -> str:
        return f"Move unit {self.unit_id} to ({self.target_q}, {self.target_r})"


@dataclass
class AttackAction(Action):
    """Attack an enemy unit or city."""
    action_type: ActionType = ActionType.ATTACK
    unit_id: int = 0
    target_q: int = 0
    target_r: int = 0
    
    def is_valid(self, game_state) -> bool:
        unit = game_state.units.get(self.unit_id)
        if unit is None or not unit.is_alive():
            return False
        if unit.owner != self.civ_id:
            return False
        if unit.has_acted:
            return False
        if unit.attack <= 0:
            return False
        
        # Check target is in range (adjacent for melee)
        target_tile = game_state.map.get_tile(self.target_q, self.target_r)
        if target_tile is None:
            return False
        
        current_tile = game_state.map.get_tile(unit.q, unit.r)
        if current_tile.distance_to(target_tile) > 1:
            return False
        
        # Check there's an enemy target
        enemies_at_target = [
            u for u in game_state.get_units_at(self.target_q, self.target_r)
            if u.owner != self.civ_id
        ]
        enemy_cities = [
            c for c in game_state.cities.values()
            if c.q == self.target_q and c.r == self.target_r and c.owner != self.civ_id
        ]
        
        if not enemies_at_target and not enemy_cities:
            return False
        
        # Check we're at war with the target
        civ = game_state.civilizations.get(self.civ_id)
        if enemies_at_target:
            target_owner = enemies_at_target[0].owner
        else:
            target_owner = enemy_cities[0].owner
        
        from .game_state import DiplomaticStatus
        if civ.relations.get(target_owner) != DiplomaticStatus.AT_WAR:
            return False
        
        return True
    
    def describe(self) -> str:
        return f"Unit {self.unit_id} attacks ({self.target_q}, {self.target_r})"


@dataclass
class GatherAction(Action):
    """Worker gathers resources from current tile."""
    action_type: ActionType = ActionType.GATHER
    unit_id: int = 0
    
    def is_valid(self, game_state) -> bool:
        unit = game_state.units.get(self.unit_id)
        if unit is None or not unit.is_alive():
            return False
        if unit.owner != self.civ_id:
            return False
        if unit.has_acted:
            return False
        
        from .game_state import UnitType
        if unit.unit_type != UnitType.WORKER:
            return False
        
        return True
    
    def describe(self) -> str:
        return f"Worker {self.unit_id} gathers resources"


@dataclass
class FoundCityAction(Action):
    """Settler founds a new city."""
    action_type: ActionType = ActionType.FOUND_CITY
    unit_id: int = 0
    city_name: str = "New City"
    
    def is_valid(self, game_state) -> bool:
        unit = game_state.units.get(self.unit_id)
        if unit is None or not unit.is_alive():
            return False
        if unit.owner != self.civ_id:
            return False
        
        from .game_state import UnitType
        if unit.unit_type != UnitType.SETTLER:
            return False
        
        # Check no city already here
        for city in game_state.cities.values():
            if city.q == unit.q and city.r == unit.r:
                return False
        
        # Check terrain is valid for city
        tile = game_state.map.get_tile(unit.q, unit.r)
        from .map import Terrain
        if tile.terrain in [Terrain.WATER, Terrain.MOUNTAINS]:
            return False
        
        return True
    
    def describe(self) -> str:
        return f"Settler {self.unit_id} founds city '{self.city_name}'"


@dataclass
class ProduceAction(Action):
    """City produces a unit or building."""
    action_type: ActionType = ActionType.PRODUCE
    city_id: int = 0
    production_type: str = ""  # 'warrior', 'worker', 'settler', etc.
    
    def is_valid(self, game_state) -> bool:
        city = game_state.cities.get(self.city_id)
        if city is None:
            return False
        if city.owner != self.civ_id:
            return False
        
        valid_productions = ['warrior', 'archer', 'worker', 'settler', 'scout']
        if self.production_type not in valid_productions:
            return False
        
        return True
    
    def describe(self) -> str:
        return f"City {self.city_id} produces {self.production_type}"


@dataclass
class ProposeTradeAction(Action):
    """Propose a trade to another civilization."""
    action_type: ActionType = ActionType.PROPOSE_TRADE
    target_civ: int = 0
    offer_resources: dict = None  # {ResourceType: amount}
    request_resources: dict = None  # {ResourceType: amount}
    
    def __post_init__(self):
        if self.offer_resources is None:
            self.offer_resources = {}
        if self.request_resources is None:
            self.request_resources = {}
    
    def is_valid(self, game_state) -> bool:
        if self.target_civ == self.civ_id:
            return False
        
        target = game_state.civilizations.get(self.target_civ)
        if target is None or not target.is_alive:
            return False
        
        civ = game_state.civilizations.get(self.civ_id)
        if civ is None:
            return False
        
        # Check we can afford what we're offering
        for rt, amount in self.offer_resources.items():
            if civ.resources.get(rt, 0) < amount:
                return False
        
        # Can't trade while at war
        from .game_state import DiplomaticStatus
        if civ.relations.get(self.target_civ) == DiplomaticStatus.AT_WAR:
            return False
        
        return True
    
    def describe(self) -> str:
        return f"Propose trade to civ {self.target_civ}: offer {self.offer_resources}, request {self.request_resources}"


@dataclass
class AcceptTradeAction(Action):
    """Accept a pending trade offer."""
    action_type: ActionType = ActionType.ACCEPT_TRADE
    trade_id: int = 0
    
    def is_valid(self, game_state) -> bool:
        offer = game_state.trade_offers.get(self.trade_id)
        if offer is None:
            return False
        if offer.to_civ != self.civ_id:
            return False
        if offer.expires_turn < game_state.current_turn:
            return False
        
        # Check both parties can still afford it
        from_civ = game_state.civilizations.get(offer.from_civ)
        to_civ = game_state.civilizations.get(offer.to_civ)
        
        return offer.is_valid(from_civ.resources, to_civ.resources)
    
    def describe(self) -> str:
        return f"Accept trade offer {self.trade_id}"


@dataclass
class RejectTradeAction(Action):
    """Reject a pending trade offer."""
    action_type: ActionType = ActionType.REJECT_TRADE
    trade_id: int = 0
    
    def is_valid(self, game_state) -> bool:
        offer = game_state.trade_offers.get(self.trade_id)
        if offer is None:
            return False
        if offer.to_civ != self.civ_id:
            return False
        return True
    
    def describe(self) -> str:
        return f"Reject trade offer {self.trade_id}"


@dataclass
class DeclareWarAction(Action):
    """Declare war on another civilization."""
    action_type: ActionType = ActionType.DECLARE_WAR
    target_civ: int = 0
    
    def is_valid(self, game_state) -> bool:
        if self.target_civ == self.civ_id:
            return False
        
        target = game_state.civilizations.get(self.target_civ)
        if target is None or not target.is_alive:
            return False
        
        civ = game_state.civilizations.get(self.civ_id)
        from .game_state import DiplomaticStatus
        
        # Can't declare war if already at war
        if civ.relations.get(self.target_civ) == DiplomaticStatus.AT_WAR:
            return False
        
        return True
    
    def describe(self) -> str:
        return f"Declare war on civ {self.target_civ}"


@dataclass
class ProposeAllianceAction(Action):
    """Propose an alliance with another civilization."""
    action_type: ActionType = ActionType.PROPOSE_ALLIANCE
    target_civ: int = 0
    
    def is_valid(self, game_state) -> bool:
        if self.target_civ == self.civ_id:
            return False
        
        target = game_state.civilizations.get(self.target_civ)
        if target is None or not target.is_alive:
            return False
        
        civ = game_state.civilizations.get(self.civ_id)
        from .game_state import DiplomaticStatus
        
        # Can't ally if at war or already allied
        current_status = civ.relations.get(self.target_civ)
        if current_status in [DiplomaticStatus.AT_WAR, DiplomaticStatus.ALLIED]:
            return False
        
        return True
    
    def describe(self) -> str:
        return f"Propose alliance to civ {self.target_civ}"


@dataclass
class EndTurnAction(Action):
    """End this civilization's turn."""
    action_type: ActionType = ActionType.END_TURN
    
    def is_valid(self, game_state) -> bool:
        civ = game_state.civilizations.get(self.civ_id)
        return civ is not None and civ.is_alive
    
    def describe(self) -> str:
        return f"Civ {self.civ_id} ends turn"


@dataclass
class NoOpAction(Action):
    """Do nothing (used for dead civs or when no valid actions)."""
    action_type: ActionType = ActionType.NO_OP
    
    def is_valid(self, game_state) -> bool:
        return True
    
    def describe(self) -> str:
        return f"Civ {self.civ_id} does nothing"


# Type alias for any action
AnyAction = Union[
    MoveAction, AttackAction, GatherAction, FoundCityAction,
    ProduceAction, ProposeTradeAction, AcceptTradeAction, RejectTradeAction,
    DeclareWarAction, ProposeAllianceAction, EndTurnAction, NoOpAction
]


def get_valid_actions(game_state, civ_id: int) -> list[AnyAction]:
    """
    Generate all valid actions for a civilization in the current state.
    
    This is used by agents to know what actions they can take.
    """
    actions = []
    civ = game_state.civilizations.get(civ_id)
    
    if civ is None or not civ.is_alive:
        return [NoOpAction(civ_id=civ_id)]
    
    from .game_state import UnitType, DiplomaticStatus
    
    # Unit actions
    for unit in game_state.get_units_for_civ(civ_id):
        # Movement
        if unit.can_move():
            for neighbor in game_state.map.get_passable_neighbors(unit.q, unit.r):
                action = MoveAction(civ_id=civ_id, unit_id=unit.id, 
                                   target_q=neighbor.q, target_r=neighbor.r)
                if action.is_valid(game_state):
                    actions.append(action)
        
        # Attack (if at war with someone nearby)
        if not unit.has_acted and unit.attack > 0:
            for neighbor in game_state.map.get_neighbors(unit.q, unit.r):
                action = AttackAction(civ_id=civ_id, unit_id=unit.id,
                                     target_q=neighbor.q, target_r=neighbor.r)
                if action.is_valid(game_state):
                    actions.append(action)
        
        # Gather (workers)
        if unit.unit_type == UnitType.WORKER:
            action = GatherAction(civ_id=civ_id, unit_id=unit.id)
            if action.is_valid(game_state):
                actions.append(action)
        
        # Found city (settlers)
        if unit.unit_type == UnitType.SETTLER:
            action = FoundCityAction(civ_id=civ_id, unit_id=unit.id,
                                    city_name=f"{civ.name} City {len(game_state.get_cities_for_civ(civ_id)) + 1}")
            if action.is_valid(game_state):
                actions.append(action)
    
    # City actions
    for city in game_state.get_cities_for_civ(civ_id):
        for prod_type in ['warrior', 'archer', 'worker', 'settler', 'scout']:
            action = ProduceAction(civ_id=civ_id, city_id=city.id, production_type=prod_type)
            if action.is_valid(game_state):
                actions.append(action)
    
    # Diplomatic actions
    for other_civ_id in game_state.civilizations:
        if other_civ_id == civ_id:
            continue
        other_civ = game_state.civilizations[other_civ_id]
        if not other_civ.is_alive:
            continue
        
        # War declaration
        action = DeclareWarAction(civ_id=civ_id, target_civ=other_civ_id)
        if action.is_valid(game_state):
            actions.append(action)
        
        # Alliance proposal
        action = ProposeAllianceAction(civ_id=civ_id, target_civ=other_civ_id)
        if action.is_valid(game_state):
            actions.append(action)
    
    # Trade offer responses
    for offer in game_state.trade_offers.values():
        if offer.to_civ == civ_id:
            accept = AcceptTradeAction(civ_id=civ_id, trade_id=offer.id)
            reject = RejectTradeAction(civ_id=civ_id, trade_id=offer.id)
            if accept.is_valid(game_state):
                actions.append(accept)
            if reject.is_valid(game_state):
                actions.append(reject)
    
    # Always can end turn
    actions.append(EndTurnAction(civ_id=civ_id))
    
    return actions
