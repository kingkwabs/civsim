"""All enums, dataclasses, and constants for CivSim."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ── Enums ────────────────────────────────────────────────────────────────

class ResourceType(Enum):
    WOOD = auto()
    STONE = auto()
    METAL = auto()
    WHEAT = auto()
    WATER = auto()
    COW = auto()


class BuildType(Enum):
    ROAD = auto()
    SETTLEMENT = auto()
    CITY = auto()


class ActionType(Enum):
    TRADE = auto()
    BUILD = auto()
    BUY_DEV = auto()
    PLAY_DEV = auto()
    DIVINE = auto()
    END_TURN = auto()


class DevCardType(Enum):
    EXPANSIONIST = auto()
    ESPIONAGE = auto()
    MAINTENANCE = auto()
    INVENTION = auto()
    PLUNDER = auto()


class PortType(Enum):
    GENERIC = auto()      # 3:1 any resource
    WOOD = auto()         # 2:1 wood
    STONE = auto()        # 2:1 stone
    METAL = auto()        # 2:1 metal
    WHEAT = auto()        # 2:1 wheat
    WATER = auto()        # 2:1 water
    COW = auto()          # 2:1 cow


# ── Type Aliases ─────────────────────────────────────────────────────────

ResourceDict = dict[ResourceType, int]
IntersectionID = int
EdgeID = int
TileID = int


# ── Board Data Structures ───────────────────────────────────────────────

@dataclass
class HexTile:
    tile_id: TileID
    q: int
    r: int
    resource: Optional[ResourceType]  # None for desert
    dice_number: Optional[int]        # None for desert


@dataclass
class Intersection:
    inter_id: IntersectionID
    adjacent_tiles: list[TileID]
    adjacent_edges: list[EdgeID] = field(default_factory=list)
    building: Optional[BuildType] = None
    owner: Optional[int] = None       # player_id or None
    port: Optional[PortType] = None


@dataclass
class Edge:
    edge_id: EdgeID
    intersections: tuple[IntersectionID, IntersectionID]
    road_owner: Optional[int] = None  # player_id or None


@dataclass
class Port:
    port_type: PortType
    intersection_ids: list[IntersectionID]
    ratio: int  # 3 for generic, 2 for specific


# ── Player ───────────────────────────────────────────────────────────────

@dataclass
class Player:
    player_id: int
    resources: ResourceDict = field(default_factory=lambda: {r: 0 for r in ResourceType})
    dev_cards: list[DevCardType] = field(default_factory=list)
    progress_points: int = 0
    is_active: bool = True
    dev_card_played_this_turn: bool = False
    maintenance_paid_this_turn: bool = False

    @property
    def total_resources(self) -> int:
        return sum(self.resources.values())

    def can_afford(self, cost: ResourceDict) -> bool:
        return all(self.resources.get(r, 0) >= amt for r, amt in cost.items())


# ── Actions ──────────────────────────────────────────────────────────────

@dataclass
class Action:
    action_type: ActionType


@dataclass
class Build(Action):
    build_type: BuildType
    position: int  # IntersectionID for settlement/city, EdgeID for road

    def __init__(self, build_type: BuildType, position: int):
        super().__init__(action_type=ActionType.BUILD)
        self.build_type = build_type
        self.position = position


@dataclass
class TradeProposal(Action):
    offering: ResourceDict
    requesting: ResourceDict
    target_player: Optional[int] = None  # None = bank trade
    port_type: Optional[PortType] = None

    def __init__(self, offering: ResourceDict, requesting: ResourceDict,
                 target_player: Optional[int] = None, port_type: Optional[PortType] = None):
        super().__init__(action_type=ActionType.TRADE)
        self.offering = offering
        self.requesting = requesting
        self.target_player = target_player
        self.port_type = port_type


@dataclass
class BuyDevCard(Action):
    def __init__(self):
        super().__init__(action_type=ActionType.BUY_DEV)


@dataclass
class PlayDevCard(Action):
    card_type: DevCardType

    def __init__(self, card_type: DevCardType):
        super().__init__(action_type=ActionType.PLAY_DEV)
        self.card_type = card_type


@dataclass
class DivineIntervention(Action):
    def __init__(self):
        super().__init__(action_type=ActionType.DIVINE)


@dataclass
class EndTurn(Action):
    def __init__(self):
        super().__init__(action_type=ActionType.END_TURN)


# ── Observation ──────────────────────────────────────────────────────────

@dataclass
class OpponentView:
    player_id: int
    total_resources: int
    num_dev_cards: int
    progress_points: int
    is_active: bool


@dataclass
class Observation:
    player_id: int
    my_resources: ResourceDict
    my_dev_cards: list[DevCardType]
    opponents: list[OpponentView]
    board: object  # Board reference
    bank_supply: ResourceDict  # vague representation
    current_dice: Optional[tuple[int, int]]
    turn_number: int


# ── Results ──────────────────────────────────────────────────────────────

@dataclass
class MaintenanceEvent:
    inter_id: IntersectionID
    player_id: int
    building_type: BuildType
    paid: bool
    cost: Optional[ResourceDict] = None


@dataclass
class ActionResult:
    success: bool
    action: Action
    description: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class PlayerStats:
    player_id: int
    final_resources: ResourceDict = field(default_factory=dict)
    buildings_owned: int = 0
    buildings_lost: int = 0
    dev_cards_played: int = 0
    trades_made: int = 0
    divine_interventions: int = 0
    total_resources_earned: int = 0
    total_resources_spent: int = 0


@dataclass
class GameResult:
    winner: Optional[int]
    scores: dict[int, int]
    turns_played: int
    player_stats: dict[int, PlayerStats]


# ── Constants ────────────────────────────────────────────────────────────

BUILDING_COSTS: dict[BuildType, ResourceDict] = {
    BuildType.ROAD: {ResourceType.WOOD: 1, ResourceType.STONE: 1},
    BuildType.SETTLEMENT: {ResourceType.WOOD: 1, ResourceType.STONE: 1,
                           ResourceType.WHEAT: 1, ResourceType.COW: 1},
    BuildType.CITY: {ResourceType.METAL: 3, ResourceType.WHEAT: 2},
}

MAINTENANCE_COSTS: dict[BuildType, ResourceDict] = {
    BuildType.SETTLEMENT: {ResourceType.WATER: 1},
    BuildType.CITY: {ResourceType.WATER: 1, ResourceType.COW: 1},
}

DEV_CARD_COST: ResourceDict = {
    ResourceType.METAL: 1,
    ResourceType.WHEAT: 1,
    ResourceType.WATER: 1,
}

DEV_CARD_COUNTS: dict[DevCardType, int] = {
    DevCardType.EXPANSIONIST: 4,
    DevCardType.ESPIONAGE: 3,
    DevCardType.MAINTENANCE: 3,
    DevCardType.INVENTION: 3,
    DevCardType.PLUNDER: 2,
}

DIVINE_EVENTS: dict[str, dict] = {
    "blessing":   {"prob": 0.30, "effect": "Receive 3 random resources from bank"},
    "famine":     {"prob": 0.20, "effect": "All opponents lose 2 random resources"},
    "earthquake": {"prob": 0.15, "effect": "Destroy one random road (any player)"},
    "prosperity": {"prob": 0.20, "effect": "All players receive 1 resource from each settlement"},
    "plague":     {"prob": 0.10, "effect": "Lose 1 settlement (downgrades to nothing)"},
    "miracle":    {"prob": 0.05, "effect": "Gain 2 free progress points"},
}

DIVINE_COST: ResourceDict = {ResourceType.COW: 2}

WIN_THRESHOLD: int = 10
MAX_TURNS: int = 200
BANK_STARTING_SUPPLY: int = 19  # per resource type

# Re-establishment costs (higher than original)
REESTABLISHMENT_COSTS: dict[BuildType, ResourceDict] = {
    BuildType.SETTLEMENT: {ResourceType.WOOD: 2, ResourceType.STONE: 2,
                           ResourceType.WHEAT: 1, ResourceType.COW: 1},
    BuildType.CITY: {ResourceType.METAL: 4, ResourceType.WHEAT: 3},
}

# Port resource mapping
PORT_RESOURCE_MAP: dict[PortType, Optional[ResourceType]] = {
    PortType.GENERIC: None,
    PortType.WOOD: ResourceType.WOOD,
    PortType.STONE: ResourceType.STONE,
    PortType.METAL: ResourceType.METAL,
    PortType.WHEAT: ResourceType.WHEAT,
    PortType.WATER: ResourceType.WATER,
    PortType.COW: ResourceType.COW,
}
