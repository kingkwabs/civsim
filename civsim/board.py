"""Hex board construction and spatial queries for CivSim."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

from .data_types import (
    BuildType, Edge, EdgeID, HexTile, Intersection, IntersectionID,
    Port, PortType, ResourceType, TileID,
)

# Axial hex direction vectors
AXIAL_DIRECTIONS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

# Standard Catan resource distribution (19 tiles)
STANDARD_RESOURCES = (
    [ResourceType.WOOD] * 4 +
    [ResourceType.STONE] * 3 +
    [ResourceType.METAL] * 3 +
    [ResourceType.WHEAT] * 4 +
    [ResourceType.WATER] * 3 +
    [ResourceType.COW] * 1 +
    [None]  # desert
)

# Standard dice number placement (excludes 7, no desert tile)
STANDARD_DICE_NUMBERS = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]

# Standard Catan hex ring layout (axial coords for radius-2 board)
STANDARD_HEX_COORDS = [
    # Center
    (0, 0),
    # Ring 1
    (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
    # Ring 2
    (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 2), (-2, 1),
    (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -2), (2, -1),
]


@dataclass
class Board:
    tiles: dict[TileID, HexTile] = field(default_factory=dict)
    intersections: dict[IntersectionID, Intersection] = field(default_factory=dict)
    edges: dict[EdgeID, Edge] = field(default_factory=dict)
    ports: list[Port] = field(default_factory=list)
    _coord_to_tile: dict[tuple[int, int], TileID] = field(default_factory=dict)

    @staticmethod
    def hex_neighbors(q: int, r: int) -> list[tuple[int, int]]:
        return [(q + dq, r + dr) for dq, dr in AXIAL_DIRECTIONS]

    @classmethod
    def from_layout(
        cls,
        tile_layout: list[tuple[int, int, Optional[ResourceType]]],
        dice_numbers: Optional[list[int]] = None,
    ) -> Board:
        board = cls()

        # Build tiles
        non_desert_idx = 0
        dice_nums = list(dice_numbers) if dice_numbers else list(STANDARD_DICE_NUMBERS)
        for tile_id, (q, r, resource) in enumerate(tile_layout):
            if resource is not None and non_desert_idx < len(dice_nums):
                dn = dice_nums[non_desert_idx]
                non_desert_idx += 1
            else:
                dn = None
            tile = HexTile(tile_id=tile_id, q=q, r=r, resource=resource, dice_number=dn)
            board.tiles[tile_id] = tile
            board._coord_to_tile[(q, r)] = tile_id

        # Derive intersections: a vertex is shared by exactly 3 mutually adjacent tiles
        board._derive_intersections()
        # Derive edges: two intersections that share exactly 2 tiles
        board._derive_edges()
        # Wire adjacency
        board._wire_edge_adjacency()

        return board

    @classmethod
    def standard_layout(cls, seed: Optional[int] = None) -> Board:
        rng = random.Random(seed)
        resources = list(STANDARD_RESOURCES)
        rng.shuffle(resources)
        dice_nums = list(STANDARD_DICE_NUMBERS)
        rng.shuffle(dice_nums)

        layout = [(q, r, resources[i]) for i, (q, r) in enumerate(STANDARD_HEX_COORDS)]
        board = cls.from_layout(layout, dice_nums)
        board._place_standard_ports(rng)
        return board

    def _derive_intersections(self) -> None:
        tile_ids = list(self.tiles.keys())
        inter_id = 0
        seen: set[frozenset[TileID]] = set()

        # Find all sets of 3 mutually adjacent tiles
        for t1, t2, t3 in combinations(tile_ids, 3):
            h1, h2, h3 = self.tiles[t1], self.tiles[t2], self.tiles[t3]
            if (self._are_adjacent(h1, h2) and
                self._are_adjacent(h2, h3) and
                self._are_adjacent(h1, h3)):
                key = frozenset([t1, t2, t3])
                if key not in seen:
                    seen.add(key)
                    self.intersections[inter_id] = Intersection(
                        inter_id=inter_id,
                        adjacent_tiles=[t1, t2, t3],
                    )
                    inter_id += 1

        # Also create intersections for border vertices (shared by 2 adjacent tiles)
        # These are vertices on the edge of the board
        for t1, t2 in combinations(tile_ids, 2):
            h1, h2 = self.tiles[t1], self.tiles[t2]
            if self._are_adjacent(h1, h2):
                # Find common neighbor coords
                n1 = set(self.hex_neighbors(h1.q, h1.r))
                n2 = set(self.hex_neighbors(h2.q, h2.r))
                common = n1 & n2
                for cn in common:
                    if cn not in self._coord_to_tile:
                        # This is a border vertex
                        key = frozenset([t1, t2, cn])
                        if key not in seen:
                            seen.add(key)
                            self.intersections[inter_id] = Intersection(
                                inter_id=inter_id,
                                adjacent_tiles=[t1, t2],
                            )
                            inter_id += 1

    def _derive_edges(self) -> None:
        edge_id = 0
        seen: set[frozenset[IntersectionID]] = set()

        for i1 in self.intersections.values():
            for i2 in self.intersections.values():
                if i1.inter_id >= i2.inter_id:
                    continue
                shared = set(i1.adjacent_tiles) & set(i2.adjacent_tiles)
                if len(shared) >= 2:
                    key = frozenset([i1.inter_id, i2.inter_id])
                    if key not in seen:
                        seen.add(key)
                        self.edges[edge_id] = Edge(
                            edge_id=edge_id,
                            intersections=(i1.inter_id, i2.inter_id),
                        )
                        edge_id += 1

    def _wire_edge_adjacency(self) -> None:
        for edge in self.edges.values():
            i1, i2 = edge.intersections
            self.intersections[i1].adjacent_edges.append(edge.edge_id)
            self.intersections[i2].adjacent_edges.append(edge.edge_id)

    def _are_adjacent(self, h1: HexTile, h2: HexTile) -> bool:
        return (h2.q, h2.r) in self.hex_neighbors(h1.q, h1.r)

    def _place_standard_ports(self, rng: random.Random) -> None:
        # Find border intersections (adjacent to < 3 tiles on the board)
        border_inters = [
            i for i in self.intersections.values()
            if len(i.adjacent_tiles) < 3
        ]
        if not border_inters:
            return

        port_types = (
            [PortType.GENERIC] * 4 +
            [PortType.WOOD, PortType.STONE, PortType.METAL,
             PortType.WHEAT, PortType.WATER]
        )
        rng.shuffle(port_types)

        # Group border intersections into pairs that share an edge
        border_ids = {i.inter_id for i in border_inters}
        border_edges = [
            e for e in self.edges.values()
            if e.intersections[0] in border_ids and e.intersections[1] in border_ids
        ]

        for i, edge in enumerate(border_edges):
            if i >= len(port_types):
                break
            pt = port_types[i]
            ratio = 3 if pt == PortType.GENERIC else 2
            port = Port(port_type=pt, intersection_ids=list(edge.intersections), ratio=ratio)
            self.ports.append(port)
            for iid in edge.intersections:
                self.intersections[iid].port = pt

    # ── Spatial Queries ──────────────────────────────────────────────────

    def get_tiles_for_intersection(self, inter_id: IntersectionID) -> list[HexTile]:
        inter = self.intersections[inter_id]
        return [self.tiles[tid] for tid in inter.adjacent_tiles]

    def get_adjacent_intersections(self, inter_id: IntersectionID) -> list[IntersectionID]:
        inter = self.intersections[inter_id]
        neighbors = set()
        for eid in inter.adjacent_edges:
            edge = self.edges[eid]
            other = edge.intersections[0] if edge.intersections[1] == inter_id else edge.intersections[1]
            neighbors.add(other)
        return list(neighbors)

    def get_edges_for_intersection(self, inter_id: IntersectionID) -> list[EdgeID]:
        return list(self.intersections[inter_id].adjacent_edges)

    # ── Valid Position Queries ───────────────────────────────────────────

    def get_valid_draft_settlements(self, player_id: int) -> list[IntersectionID]:
        valid = []
        for inter in self.intersections.values():
            if inter.building is not None:
                continue
            # Distance rule: no adjacent buildings
            if self._has_adjacent_building(inter.inter_id):
                continue
            valid.append(inter.inter_id)
        return valid

    def get_valid_draft_roads(self, player_id: int, settlement_id: IntersectionID) -> list[EdgeID]:
        valid = []
        for eid in self.intersections[settlement_id].adjacent_edges:
            edge = self.edges[eid]
            if edge.road_owner is None:
                valid.append(eid)
        return valid

    def get_valid_settlement_positions(self, player_id: int) -> list[IntersectionID]:
        valid = []
        for inter in self.intersections.values():
            if inter.building is not None:
                continue
            if self._has_adjacent_building(inter.inter_id):
                continue
            # Must be adjacent to player's road
            if not self._has_adjacent_road(inter.inter_id, player_id):
                continue
            valid.append(inter.inter_id)
        return valid

    def get_valid_road_positions(self, player_id: int) -> list[EdgeID]:
        valid = []
        for edge in self.edges.values():
            if edge.road_owner is not None:
                continue
            # Must be adjacent to player's existing road or building
            i1, i2 = edge.intersections
            if (self._player_has_presence(i1, player_id) or
                self._player_has_presence(i2, player_id)):
                valid.append(edge.edge_id)
        return valid

    def get_valid_city_positions(self, player_id: int) -> list[IntersectionID]:
        return [
            inter.inter_id for inter in self.intersections.values()
            if inter.building == BuildType.SETTLEMENT and inter.owner == player_id
        ]

    def get_abandoned_buildings(self) -> list[IntersectionID]:
        return [
            inter.inter_id for inter in self.intersections.values()
            if inter.building is not None and inter.owner is None
        ]

    def player_has_port_access(self, player_id: int, port_type: PortType) -> bool:
        for port in self.ports:
            if port.port_type == port_type:
                for iid in port.intersection_ids:
                    inter = self.intersections[iid]
                    if inter.owner == player_id:
                        return True
        return False

    def get_player_ports(self, player_id: int) -> list[Port]:
        result = []
        seen = set()
        for port in self.ports:
            for iid in port.intersection_ids:
                if self.intersections[iid].owner == player_id:
                    key = id(port)
                    if key not in seen:
                        seen.add(key)
                        result.append(port)
                    break
        return result

    # ── Mutation ─────────────────────────────────────────────────────────

    def place_road(self, player_id: int, edge_id: EdgeID) -> None:
        self.edges[edge_id].road_owner = player_id

    def place_settlement(self, player_id: int, inter_id: IntersectionID) -> None:
        inter = self.intersections[inter_id]
        inter.building = BuildType.SETTLEMENT
        inter.owner = player_id

    def upgrade_to_city(self, player_id: int, inter_id: IntersectionID) -> None:
        inter = self.intersections[inter_id]
        inter.building = BuildType.CITY

    def abandon_building(self, inter_id: IntersectionID) -> None:
        self.intersections[inter_id].owner = None

    def claim_building(self, player_id: int, inter_id: IntersectionID) -> None:
        self.intersections[inter_id].owner = player_id

    # ── Private Helpers ──────────────────────────────────────────────────

    def _has_adjacent_building(self, inter_id: IntersectionID) -> bool:
        for neighbor_id in self.get_adjacent_intersections(inter_id):
            if self.intersections[neighbor_id].building is not None:
                return True
        return False

    def _has_adjacent_road(self, inter_id: IntersectionID, player_id: int) -> bool:
        for eid in self.intersections[inter_id].adjacent_edges:
            if self.edges[eid].road_owner == player_id:
                return True
        return False

    def _player_has_presence(self, inter_id: IntersectionID, player_id: int) -> bool:
        inter = self.intersections[inter_id]
        if inter.owner == player_id:
            return True
        for eid in inter.adjacent_edges:
            if self.edges[eid].road_owner == player_id:
                return True
        return False

    def clone(self) -> Board:
        return copy.deepcopy(self)
