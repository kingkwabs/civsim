"""
Hex grid map implementation for CivSim.

Uses axial coordinates (q, r) for hex positioning.
Reference: https://www.redblobgames.com/grids/hexagons/
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterator, Optional
import random


class Terrain(Enum):
    """Terrain types that affect resource generation and movement."""
    PLAINS = auto()      # Balanced - moderate food/materials
    FOREST = auto()      # High materials, low food
    MOUNTAINS = auto()   # High materials, impassable without tech
    WATER = auto()       # Impassable, but enables trade routes
    FERTILE = auto()     # High food production
    DESERT = auto()      # Low resources, but may have rare materials


class ResourceType(Enum):
    """The three core resources in the game."""
    FOOD = auto()        # Required for unit upkeep and population
    MATERIALS = auto()   # Required for building and military
    TECHNOLOGY = auto()  # Required for upgrades and special abilities


@dataclass
class HexTile:
    """A single hex tile on the map."""
    q: int  # Axial coordinate q
    r: int  # Axial coordinate r
    terrain: Terrain = Terrain.PLAINS
    
    # Resource yields per turn when worked
    resource_yields: dict[ResourceType, float] = field(default_factory=dict)
    
    # Current owner (civilization id), None if unclaimed
    owner: Optional[int] = None
    
    # Any structure built here
    structure: Optional[str] = None
    
    def __post_init__(self):
        if not self.resource_yields:
            self.resource_yields = self._default_yields()
    
    def _default_yields(self) -> dict[ResourceType, float]:
        """Default resource yields based on terrain type."""
        yields = {
            Terrain.PLAINS: {ResourceType.FOOD: 2, ResourceType.MATERIALS: 1, ResourceType.TECHNOLOGY: 0},
            Terrain.FOREST: {ResourceType.FOOD: 1, ResourceType.MATERIALS: 3, ResourceType.TECHNOLOGY: 0},
            Terrain.MOUNTAINS: {ResourceType.FOOD: 0, ResourceType.MATERIALS: 4, ResourceType.TECHNOLOGY: 1},
            Terrain.WATER: {ResourceType.FOOD: 2, ResourceType.MATERIALS: 0, ResourceType.TECHNOLOGY: 0},
            Terrain.FERTILE: {ResourceType.FOOD: 4, ResourceType.MATERIALS: 1, ResourceType.TECHNOLOGY: 0},
            Terrain.DESERT: {ResourceType.FOOD: 0, ResourceType.MATERIALS: 1, ResourceType.TECHNOLOGY: 2},
        }
        return yields.get(self.terrain, {ResourceType.FOOD: 1, ResourceType.MATERIALS: 1, ResourceType.TECHNOLOGY: 0})
    
    @property
    def s(self) -> int:
        """Third cube coordinate (derived from axial)."""
        return -self.q - self.r
    
    def distance_to(self, other: HexTile) -> int:
        """Calculate hex distance to another tile."""
        return (abs(self.q - other.q) + abs(self.r - other.r) + abs(self.s - other.s)) // 2
    
    def is_passable(self, has_mountain_tech: bool = False) -> bool:
        """Check if a unit can move through this tile."""
        if self.terrain == Terrain.WATER:
            return False
        if self.terrain == Terrain.MOUNTAINS and not has_mountain_tech:
            return False
        return True
    
    def __hash__(self):
        return hash((self.q, self.r))
    
    def __eq__(self, other):
        if not isinstance(other, HexTile):
            return False
        return self.q == other.q and self.r == other.r


# Axial direction vectors for the 6 hex neighbors
HEX_DIRECTIONS = [
    (1, 0), (1, -1), (0, -1),
    (-1, 0), (-1, 1), (0, 1)
]


class HexMap:
    """
    The game map consisting of hexagonal tiles.
    
    Uses a rectangular bounding box in axial coordinates.
    """
    
    def __init__(self, width: int = 25, height: int = 25, seed: Optional[int] = None):
        """
        Initialize a hex map.
        
        Args:
            width: Number of columns (q direction)
            height: Number of rows (r direction)
            seed: Random seed for procedural generation
        """
        self.width = width
        self.height = height
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Store tiles in a dict for sparse access
        self.tiles: dict[tuple[int, int], HexTile] = {}
        
        self._generate_map()
    
    def _generate_map(self) -> None:
        """Procedurally generate the map terrain."""
        # Simple noise-based generation
        # In a full implementation, you'd use Perlin noise or similar
        
        for q in range(self.width):
            for r in range(self.height):
                terrain = self._generate_terrain(q, r)
                self.tiles[(q, r)] = HexTile(q=q, r=r, terrain=terrain)
    
    def _generate_terrain(self, q: int, r: int) -> Terrain:
        """Generate terrain for a single tile based on position and noise."""
        # Simple procedural generation - can be improved with Perlin noise
        val = self.rng.random()
        
        # Edge tiles more likely to be water
        edge_dist = min(q, r, self.width - 1 - q, self.height - 1 - r)
        if edge_dist < 2 and val < 0.4:
            return Terrain.WATER
        
        # Interior terrain distribution
        if val < 0.35:
            return Terrain.PLAINS
        elif val < 0.55:
            return Terrain.FOREST
        elif val < 0.70:
            return Terrain.FERTILE
        elif val < 0.85:
            return Terrain.MOUNTAINS
        elif val < 0.92:
            return Terrain.DESERT
        else:
            return Terrain.WATER
    
    def get_tile(self, q: int, r: int) -> Optional[HexTile]:
        """Get tile at coordinates, or None if out of bounds."""
        return self.tiles.get((q, r))
    
    def __getitem__(self, coords: tuple[int, int]) -> Optional[HexTile]:
        """Allow map[q, r] access."""
        return self.get_tile(coords[0], coords[1])
    
    def get_neighbors(self, q: int, r: int) -> list[HexTile]:
        """Get all valid neighboring tiles."""
        neighbors = []
        for dq, dr in HEX_DIRECTIONS:
            tile = self.get_tile(q + dq, r + dr)
            if tile is not None:
                neighbors.append(tile)
        return neighbors
    
    def get_tiles_in_radius(self, q: int, r: int, radius: int) -> list[HexTile]:
        """Get all tiles within a certain hex distance."""
        center = self.get_tile(q, r)
        if center is None:
            return []
        
        tiles = []
        for tile in self.tiles.values():
            if center.distance_to(tile) <= radius:
                tiles.append(tile)
        return tiles
    
    def get_passable_neighbors(self, q: int, r: int, has_mountain_tech: bool = False) -> list[HexTile]:
        """Get neighboring tiles that can be moved into."""
        return [t for t in self.get_neighbors(q, r) if t.is_passable(has_mountain_tech)]
    
    def find_path(self, start: tuple[int, int], end: tuple[int, int], 
                  has_mountain_tech: bool = False) -> Optional[list[tuple[int, int]]]:
        """
        A* pathfinding between two points.
        
        Returns list of coordinates from start to end, or None if no path.
        """
        start_tile = self.get_tile(*start)
        end_tile = self.get_tile(*end)
        
        if start_tile is None or end_tile is None:
            return None
        if not end_tile.is_passable(has_mountain_tech):
            return None
        
        # A* implementation
        open_set = {start}
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        
        g_score = {start: 0}
        f_score = {start: start_tile.distance_to(end_tile)}
        
        while open_set:
            # Get node with lowest f_score
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            if current == end:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))
            
            open_set.remove(current)
            current_tile = self.get_tile(*current)
            
            for neighbor in self.get_passable_neighbors(*current, has_mountain_tech):
                neighbor_coords = (neighbor.q, neighbor.r)
                tentative_g = g_score[current] + 1  # All moves cost 1 for now
                
                if tentative_g < g_score.get(neighbor_coords, float('inf')):
                    came_from[neighbor_coords] = current
                    g_score[neighbor_coords] = tentative_g
                    f_score[neighbor_coords] = tentative_g + neighbor.distance_to(end_tile)
                    open_set.add(neighbor_coords)
        
        return None  # No path found
    
    def __iter__(self) -> Iterator[HexTile]:
        """Iterate over all tiles."""
        return iter(self.tiles.values())
    
    def __len__(self) -> int:
        """Number of tiles in the map."""
        return len(self.tiles)
    
    def to_numpy(self, attribute: str = 'terrain') -> np.ndarray:
        """
        Convert map to numpy array for neural network input.
        
        Args:
            attribute: Which attribute to extract ('terrain', 'owner', etc.)
        
        Returns:
            2D numpy array of shape (height, width)
        """
        arr = np.zeros((self.height, self.width), dtype=np.int32)
        
        for (q, r), tile in self.tiles.items():
            if attribute == 'terrain':
                arr[r, q] = tile.terrain.value
            elif attribute == 'owner':
                arr[r, q] = tile.owner if tile.owner is not None else -1
        
        return arr
    
    def get_spawn_positions(self, num_players: int) -> list[tuple[int, int]]:
        """
        Find good spawn positions for civilizations.
        
        Tries to space players evenly and place them on passable terrain.
        """
        # Find all passable, non-edge tiles
        candidates = [
            (t.q, t.r) for t in self.tiles.values()
            if t.is_passable() and 
            min(t.q, t.r, self.width - 1 - t.q, self.height - 1 - t.r) > 3
        ]
        
        if len(candidates) < num_players:
            raise ValueError(f"Map too small for {num_players} players")
        
        # Greedy placement: maximize minimum distance between spawns
        spawns = [candidates[self.rng.randint(0, len(candidates) - 1)]]
        
        while len(spawns) < num_players:
            best_candidate = None
            best_min_dist = -1
            
            for c in candidates:
                if c in spawns:
                    continue
                min_dist = min(
                    self.tiles[c].distance_to(self.tiles[s]) 
                    for s in spawns
                )
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_candidate = c
            
            if best_candidate:
                spawns.append(best_candidate)
            else:
                break
        
        return spawns
