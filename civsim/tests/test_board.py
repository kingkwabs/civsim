"""Tests for board construction and spatial queries."""
import pytest
from civsim.board import Board, STANDARD_HEX_COORDS
from civsim.data_types import BuildType, PortType, ResourceType


class TestBoardConstruction:
    def test_standard_layout_creates_19_tiles(self):
        board = Board.standard_layout(seed=42)
        assert len(board.tiles) == 19

    def test_standard_layout_has_intersections(self):
        board = Board.standard_layout(seed=42)
        # Standard Catan has 54 intersections
        assert len(board.intersections) > 0

    def test_standard_layout_has_edges(self):
        board = Board.standard_layout(seed=42)
        # Standard Catan has 72 edges
        assert len(board.edges) > 0

    def test_intersections_reference_valid_tiles(self):
        board = Board.standard_layout(seed=42)
        for inter in board.intersections.values():
            for tid in inter.adjacent_tiles:
                assert tid in board.tiles

    def test_edges_reference_valid_intersections(self):
        board = Board.standard_layout(seed=42)
        for edge in board.edges.values():
            assert edge.intersections[0] in board.intersections
            assert edge.intersections[1] in board.intersections

    def test_hex_neighbors_returns_6(self):
        neighbors = Board.hex_neighbors(0, 0)
        assert len(neighbors) == 6

    def test_deterministic_with_seed(self):
        b1 = Board.standard_layout(seed=123)
        b2 = Board.standard_layout(seed=123)
        assert len(b1.tiles) == len(b2.tiles)
        for tid in b1.tiles:
            assert b1.tiles[tid].resource == b2.tiles[tid].resource
            assert b1.tiles[tid].dice_number == b2.tiles[tid].dice_number


class TestSpatialQueries:
    @pytest.fixture
    def board(self):
        return Board.standard_layout(seed=42)

    def test_get_tiles_for_intersection(self, board):
        for inter in board.intersections.values():
            tiles = board.get_tiles_for_intersection(inter.inter_id)
            assert len(tiles) >= 2  # border intersections have 2, inner have 3

    def test_get_adjacent_intersections(self, board):
        for inter in board.intersections.values():
            neighbors = board.get_adjacent_intersections(inter.inter_id)
            # Each intersection has 2-3 neighbors
            assert len(neighbors) >= 1

    def test_valid_draft_settlements_initially_all(self, board):
        valid = board.get_valid_draft_settlements(0)
        assert len(valid) == len(board.intersections)

    def test_distance_rule_enforced(self, board):
        valid = board.get_valid_draft_settlements(0)
        # Place a settlement
        pos = valid[0]
        board.place_settlement(0, pos)
        # Adjacent positions should now be invalid
        new_valid = board.get_valid_draft_settlements(0)
        assert pos not in new_valid
        neighbors = board.get_adjacent_intersections(pos)
        for n in neighbors:
            assert n not in new_valid

    def test_valid_road_positions_empty_initially(self, board):
        # No buildings placed yet, so no valid road positions
        valid = board.get_valid_road_positions(0)
        assert len(valid) == 0

    def test_valid_roads_after_settlement(self, board):
        pos = board.get_valid_draft_settlements(0)[0]
        board.place_settlement(0, pos)
        valid = board.get_valid_road_positions(0)
        assert len(valid) > 0

    def test_valid_city_positions(self, board):
        pos = board.get_valid_draft_settlements(0)[0]
        board.place_settlement(0, pos)
        valid = board.get_valid_city_positions(0)
        assert pos in valid

    def test_abandoned_buildings(self, board):
        pos = board.get_valid_draft_settlements(0)[0]
        board.place_settlement(0, pos)
        assert len(board.get_abandoned_buildings()) == 0
        board.abandon_building(pos)
        assert pos in board.get_abandoned_buildings()


class TestBoardMutation:
    @pytest.fixture
    def board(self):
        return Board.standard_layout(seed=42)

    def test_place_road(self, board):
        pos = board.get_valid_draft_settlements(0)[0]
        board.place_settlement(0, pos)
        roads = board.get_valid_draft_roads(0, pos)
        board.place_road(0, roads[0])
        assert board.edges[roads[0]].road_owner == 0

    def test_place_settlement(self, board):
        pos = board.get_valid_draft_settlements(0)[0]
        board.place_settlement(0, pos)
        inter = board.intersections[pos]
        assert inter.building == BuildType.SETTLEMENT
        assert inter.owner == 0

    def test_upgrade_to_city(self, board):
        pos = board.get_valid_draft_settlements(0)[0]
        board.place_settlement(0, pos)
        board.upgrade_to_city(0, pos)
        assert board.intersections[pos].building == BuildType.CITY

    def test_clone_independence(self, board):
        pos = board.get_valid_draft_settlements(0)[0]
        board.place_settlement(0, pos)
        clone = board.clone()
        clone.intersections[pos].owner = 1
        assert board.intersections[pos].owner == 0
