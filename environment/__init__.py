"""
CivSim Environment Package

A multi-agent civilization strategy game for AI research.
"""

from .map import HexMap, HexTile, Terrain, ResourceType, HEX_DIRECTIONS
from .game_state import (
    GameState, 
    Civilization, 
    Unit, 
    City, 
    UnitType,
    DiplomaticStatus,
    TradeOffer,
    Message,
)
from .actions import (
    Action,
    ActionType,
    AnyAction,
    MoveAction,
    AttackAction,
    GatherAction,
    FoundCityAction,
    ProduceAction,
    ProposeTradeAction,
    AcceptTradeAction,
    RejectTradeAction,
    DeclareWarAction,
    ProposeAllianceAction,
    EndTurnAction,
    NoOpAction,
    get_valid_actions,
)
from .mechanics import GameMechanics
from .civsim_env import CivSimEnv, StepResult, make_env

__all__ = [
    # Map
    'HexMap',
    'HexTile', 
    'Terrain',
    'ResourceType',
    'HEX_DIRECTIONS',
    
    # Game State
    'GameState',
    'Civilization',
    'Unit',
    'City',
    'UnitType',
    'DiplomaticStatus',
    'TradeOffer',
    'Message',
    
    # Actions
    'Action',
    'ActionType',
    'AnyAction',
    'MoveAction',
    'AttackAction',
    'GatherAction',
    'FoundCityAction',
    'ProduceAction',
    'ProposeTradeAction',
    'AcceptTradeAction',
    'RejectTradeAction',
    'DeclareWarAction',
    'ProposeAllianceAction',
    'EndTurnAction',
    'NoOpAction',
    'get_valid_actions',
    
    # Mechanics
    'GameMechanics',
    
    # Environment
    'CivSimEnv',
    'StepResult',
    'make_env',
]
