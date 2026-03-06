"""
Game state representation for CivSim.

Contains all state needed to fully describe a game at any point in time.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import numpy as np

from .map import HexMap, ResourceType, HexTile


class UnitType(Enum):
    """Types of units civilizations can control."""
    SETTLER = auto()     # Can found new cities, weak in combat
    WARRIOR = auto()     # Basic military unit
    ARCHER = auto()      # Ranged military unit
    WORKER = auto()      # Gathers resources, builds improvements
    SCOUT = auto()       # Fast movement, reveals fog of war


@dataclass
class Unit:
    """A single unit on the map."""
    id: int
    unit_type: UnitType
    owner: int  # Civilization ID
    q: int      # Position (axial q)
    r: int      # Position (axial r)
    
    health: int = 100
    max_health: int = 100
    movement_left: int = 2
    max_movement: int = 2
    
    # Combat stats
    attack: int = 10
    defense: int = 10
    
    # Has this unit acted this turn?
    has_acted: bool = False
    
    def __post_init__(self):
        """Set stats based on unit type."""
        stats = {
            UnitType.SETTLER: {'attack': 0, 'defense': 5, 'max_movement': 2, 'max_health': 50},
            UnitType.WARRIOR: {'attack': 15, 'defense': 15, 'max_movement': 2, 'max_health': 100},
            UnitType.ARCHER: {'attack': 20, 'defense': 8, 'max_movement': 2, 'max_health': 75},
            UnitType.WORKER: {'attack': 0, 'defense': 5, 'max_movement': 2, 'max_health': 50},
            UnitType.SCOUT: {'attack': 5, 'defense': 5, 'max_movement': 4, 'max_health': 60},
        }
        if self.unit_type in stats:
            for key, val in stats[self.unit_type].items():
                setattr(self, key, val)
            self.movement_left = self.max_movement
            self.health = self.max_health
    
    @property
    def position(self) -> tuple[int, int]:
        return (self.q, self.r)
    
    @position.setter
    def position(self, coords: tuple[int, int]):
        self.q, self.r = coords
    
    def is_alive(self) -> bool:
        return self.health > 0
    
    def can_move(self) -> bool:
        return self.movement_left > 0 and self.is_alive()
    
    def reset_turn(self):
        """Reset unit state for a new turn."""
        self.movement_left = self.max_movement
        self.has_acted = False


@dataclass 
class City:
    """A city belonging to a civilization."""
    id: int
    name: str
    owner: int  # Civilization ID
    q: int      # Position
    r: int
    
    population: int = 1
    health: int = 200
    max_health: int = 200
    
    # Production queue
    production_queue: list[str] = field(default_factory=list)
    production_progress: int = 0
    
    # Tiles this city is working
    worked_tiles: list[tuple[int, int]] = field(default_factory=list)
    
    @property
    def position(self) -> tuple[int, int]:
        return (self.q, self.r)
    
    def get_yields(self, game_map: HexMap) -> dict[ResourceType, float]:
        """Calculate total resource yields from worked tiles."""
        yields = {rt: 0.0 for rt in ResourceType}
        
        # Base city yield
        yields[ResourceType.FOOD] += 2
        yields[ResourceType.MATERIALS] += 2
        
        # Add yields from worked tiles
        for tq, tr in self.worked_tiles:
            tile = game_map.get_tile(tq, tr)
            if tile:
                for rt, amount in tile.resource_yields.items():
                    yields[rt] += amount
        
        return yields


class DiplomaticStatus(Enum):
    """Relationship status between civilizations."""
    NEUTRAL = auto()
    AT_WAR = auto()
    ALLIED = auto()
    TRADE_AGREEMENT = auto()


@dataclass
class TradeOffer:
    """A pending trade offer between civilizations."""
    id: int
    from_civ: int
    to_civ: int
    
    # What the offering civ gives
    offer_resources: dict[ResourceType, int] = field(default_factory=dict)
    
    # What they want in return
    request_resources: dict[ResourceType, int] = field(default_factory=dict)
    
    # Turn this offer expires
    expires_turn: int = 0
    
    def is_valid(self, from_stockpile: dict, to_stockpile: dict) -> bool:
        """Check if both parties can fulfill the trade."""
        for rt, amount in self.offer_resources.items():
            if from_stockpile.get(rt, 0) < amount:
                return False
        for rt, amount in self.request_resources.items():
            if to_stockpile.get(rt, 0) < amount:
                return False
        return True


@dataclass
class Civilization:
    """A civilization (player) in the game."""
    id: int
    name: str
    
    # Resources stockpile
    resources: dict[ResourceType, float] = field(default_factory=lambda: {
        ResourceType.FOOD: 50.0,
        ResourceType.MATERIALS: 50.0,
        ResourceType.TECHNOLOGY: 0.0,
    })
    
    # Diplomatic relationships with other civs
    relations: dict[int, DiplomaticStatus] = field(default_factory=dict)
    
    # Trust scores for other civs (for AI reasoning)
    trust_scores: dict[int, float] = field(default_factory=dict)
    
    # Technologies unlocked
    technologies: set[str] = field(default_factory=set)
    
    # Is this civ still in the game?
    is_alive: bool = True
    
    # Score for evaluation
    score: int = 0
    
    def can_afford(self, costs: dict[ResourceType, float]) -> bool:
        """Check if civ can afford a cost."""
        for rt, amount in costs.items():
            if self.resources.get(rt, 0) < amount:
                return False
        return True
    
    def pay_cost(self, costs: dict[ResourceType, float]) -> bool:
        """Deduct costs from resources. Returns False if can't afford."""
        if not self.can_afford(costs):
            return False
        for rt, amount in costs.items():
            self.resources[rt] -= amount
        return True
    
    def add_resources(self, amounts: dict[ResourceType, float]):
        """Add resources to stockpile."""
        for rt, amount in amounts.items():
            self.resources[rt] = self.resources.get(rt, 0) + amount
    
    def update_trust(self, other_civ: int, delta: float):
        """Update trust score for another civ."""
        current = self.trust_scores.get(other_civ, 0.5)
        self.trust_scores[other_civ] = max(0.0, min(1.0, current + delta))


@dataclass
class Message:
    """A diplomatic message between civilizations."""
    id: int
    from_civ: int
    to_civ: int
    message_type: str  # 'trade_offer', 'alliance_proposal', 'war_declaration', 'peace_offer'
    content: dict = field(default_factory=dict)
    turn_sent: int = 0


class GameState:
    """
    Complete state of a CivSim game.
    
    This is the authoritative game state that the environment manages.
    """
    
    def __init__(
        self,
        map_width: int = 25,
        map_height: int = 25,
        num_civilizations: int = 4,
        max_turns: int = 500,
        seed: Optional[int] = None,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Game configuration
        self.map_width = map_width
        self.map_height = map_height
        self.num_civilizations = num_civilizations
        self.max_turns = max_turns
        
        # Current turn (0-indexed)
        self.current_turn: int = 0
        
        # The map
        self.map: HexMap = HexMap(map_width, map_height, seed)
        
        # Civilizations
        self.civilizations: dict[int, Civilization] = {}
        
        # All units on the map
        self.units: dict[int, Unit] = {}
        self._next_unit_id: int = 0
        
        # All cities
        self.cities: dict[int, City] = {}
        self._next_city_id: int = 0
        
        # Pending trade offers
        self.trade_offers: dict[int, TradeOffer] = {}
        self._next_trade_id: int = 0
        
        # Message history (for analysis)
        self.messages: list[Message] = []
        self._next_message_id: int = 0
        
        # Game over flag
        self.game_over: bool = False
        self.winner: Optional[int] = None
        
        # Initialize civilizations
        self._initialize_civilizations()
    
    def _initialize_civilizations(self):
        """Set up starting civilizations with units and cities."""
        civ_names = ['Rome', 'Egypt', 'China', 'Greece', 'Persia', 'India', 'Japan', 'Aztec']
        spawn_positions = self.map.get_spawn_positions(self.num_civilizations)
        
        for i in range(self.num_civilizations):
            civ = Civilization(
                id=i,
                name=civ_names[i % len(civ_names)],
            )
            self.civilizations[i] = civ
            
            # Initialize diplomacy (neutral with everyone)
            for j in range(self.num_civilizations):
                if i != j:
                    civ.relations[j] = DiplomaticStatus.NEUTRAL
                    civ.trust_scores[j] = 0.5
            
            # Spawn starting units
            sq, sr = spawn_positions[i]
            
            # Starting city
            city = self._create_city(i, sq, sr, f"{civ.name} Capital")
            
            # Starting units
            self._create_unit(i, UnitType.WARRIOR, sq, sr)
            self._create_unit(i, UnitType.WORKER, sq, sr)
            self._create_unit(i, UnitType.SCOUT, sq, sr)
            
            # Claim nearby tiles
            for tile in self.map.get_tiles_in_radius(sq, sr, 2):
                tile.owner = i
    
    def _create_unit(self, owner: int, unit_type: UnitType, q: int, r: int) -> Unit:
        """Create a new unit and add it to the game."""
        unit = Unit(
            id=self._next_unit_id,
            unit_type=unit_type,
            owner=owner,
            q=q,
            r=r,
        )
        self.units[unit.id] = unit
        self._next_unit_id += 1
        return unit
    
    def _create_city(self, owner: int, q: int, r: int, name: str) -> City:
        """Create a new city and add it to the game."""
        city = City(
            id=self._next_city_id,
            name=name,
            owner=owner,
            q=q,
            r=r,
        )
        self.cities[city.id] = city
        self._next_city_id += 1
        
        # City claims its tile
        tile = self.map.get_tile(q, r)
        if tile:
            tile.owner = owner
            tile.structure = 'city'
        
        return city
    
    def get_units_at(self, q: int, r: int) -> list[Unit]:
        """Get all units at a position."""
        return [u for u in self.units.values() if u.q == q and u.r == r and u.is_alive()]
    
    def get_units_for_civ(self, civ_id: int) -> list[Unit]:
        """Get all units belonging to a civilization."""
        return [u for u in self.units.values() if u.owner == civ_id and u.is_alive()]
    
    def get_cities_for_civ(self, civ_id: int) -> list[City]:
        """Get all cities belonging to a civilization."""
        return [c for c in self.cities.values() if c.owner == civ_id]
    
    def get_civ_score(self, civ_id: int) -> int:
        """Calculate current score for a civilization."""
        civ = self.civilizations.get(civ_id)
        if civ is None or not civ.is_alive:
            return 0
        
        score = 0
        
        # Points for cities
        cities = self.get_cities_for_civ(civ_id)
        score += len(cities) * 100
        score += sum(c.population for c in cities) * 20
        
        # Points for units
        units = self.get_units_for_civ(civ_id)
        score += len(units) * 10
        
        # Points for territory
        owned_tiles = sum(1 for t in self.map if t.owner == civ_id)
        score += owned_tiles * 5
        
        # Points for resources
        score += int(civ.resources.get(ResourceType.FOOD, 0) * 0.5)
        score += int(civ.resources.get(ResourceType.MATERIALS, 0) * 0.5)
        score += int(civ.resources.get(ResourceType.TECHNOLOGY, 0) * 2)
        
        # Points for tech
        score += len(civ.technologies) * 50
        
        civ.score = score
        return score
    
    def get_alive_civilizations(self) -> list[Civilization]:
        """Get all civilizations still in the game."""
        return [c for c in self.civilizations.values() if c.is_alive]
    
    def check_elimination(self, civ_id: int):
        """Check if a civilization has been eliminated."""
        civ = self.civilizations.get(civ_id)
        if civ is None:
            return
        
        cities = self.get_cities_for_civ(civ_id)
        units = self.get_units_for_civ(civ_id)
        
        if len(cities) == 0 and len(units) == 0:
            civ.is_alive = False
    
    def check_game_over(self) -> bool:
        """Check if the game has ended."""
        alive = self.get_alive_civilizations()
        
        # One civ left standing
        if len(alive) == 1:
            self.game_over = True
            self.winner = alive[0].id
            return True
        
        # No civs left (shouldn't happen)
        if len(alive) == 0:
            self.game_over = True
            return True
        
        # Turn limit reached
        if self.current_turn >= self.max_turns:
            self.game_over = True
            # Winner is highest score
            scores = [(c.id, self.get_civ_score(c.id)) for c in alive]
            self.winner = max(scores, key=lambda x: x[1])[0]
            return True
        
        return False
    
    def to_observation(self, civ_id: int, visibility_radius: int = 3) -> dict:
        """
        Generate the observation for a specific civilization.
        
        This implements fog of war - civs only see tiles near their units/cities.
        """
        civ = self.civilizations.get(civ_id)
        if civ is None:
            return {}
        
        # Determine visible tiles
        visible_coords: set[tuple[int, int]] = set()
        
        # Vision from units
        for unit in self.get_units_for_civ(civ_id):
            for tile in self.map.get_tiles_in_radius(unit.q, unit.r, visibility_radius):
                visible_coords.add((tile.q, tile.r))
        
        # Vision from cities
        for city in self.get_cities_for_civ(civ_id):
            for tile in self.map.get_tiles_in_radius(city.q, city.r, visibility_radius + 1):
                visible_coords.add((tile.q, tile.r))
        
        # Build observation dict
        obs = {
            'turn': self.current_turn,
            'civ_id': civ_id,
            'resources': dict(civ.resources),
            'technologies': list(civ.technologies),
            
            # Visible map state
            'visible_tiles': [],
            'visible_units': [],
            'visible_cities': [],
            
            # Own state (always visible)
            'own_units': [],
            'own_cities': [],
            
            # Diplomatic state
            'relations': dict(civ.relations),
            'trust_scores': dict(civ.trust_scores),
            
            # Pending offers
            'incoming_offers': [],
            'outgoing_offers': [],
        }
        
        # Add visible tile data
        for q, r in visible_coords:
            tile = self.map.get_tile(q, r)
            if tile:
                obs['visible_tiles'].append({
                    'q': q, 'r': r,
                    'terrain': tile.terrain.name,
                    'owner': tile.owner,
                    'structure': tile.structure,
                    'yields': {rt.name: v for rt, v in tile.resource_yields.items()},
                })
        
        # Add visible units
        for unit in self.units.values():
            if not unit.is_alive():
                continue
            if (unit.q, unit.r) in visible_coords:
                unit_data = {
                    'id': unit.id,
                    'type': unit.unit_type.name,
                    'owner': unit.owner,
                    'q': unit.q, 'r': unit.r,
                    'health': unit.health,
                }
                if unit.owner == civ_id:
                    unit_data['movement_left'] = unit.movement_left
                    unit_data['has_acted'] = unit.has_acted
                    obs['own_units'].append(unit_data)
                obs['visible_units'].append(unit_data)
        
        # Add visible cities
        for city in self.cities.values():
            if (city.q, city.r) in visible_coords:
                city_data = {
                    'id': city.id,
                    'name': city.name,
                    'owner': city.owner,
                    'q': city.q, 'r': city.r,
                    'population': city.population,
                }
                if city.owner == civ_id:
                    city_data['health'] = city.health
                    city_data['production_queue'] = city.production_queue
                    obs['own_cities'].append(city_data)
                obs['visible_cities'].append(city_data)
        
        # Add trade offers
        for offer in self.trade_offers.values():
            offer_data = {
                'id': offer.id,
                'from_civ': offer.from_civ,
                'to_civ': offer.to_civ,
                'offer': {rt.name: v for rt, v in offer.offer_resources.items()},
                'request': {rt.name: v for rt, v in offer.request_resources.items()},
                'expires': offer.expires_turn,
            }
            if offer.to_civ == civ_id:
                obs['incoming_offers'].append(offer_data)
            elif offer.from_civ == civ_id:
                obs['outgoing_offers'].append(offer_data)
        
        return obs
    
    def clone(self) -> GameState:
        """Create a deep copy of the game state (for MCTS simulations)."""
        import copy
        return copy.deepcopy(self)
