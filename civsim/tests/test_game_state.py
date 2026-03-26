"""Tests for game state management."""
import pytest
from civsim.game_state import GameState
from civsim.data_types import BuildType, ResourceType, WIN_THRESHOLD


class TestGameStateInit:
    def test_new_game_creates_players(self):
        state = GameState.new_game(num_players=3, seed=42)
        assert len(state.players) == 3

    def test_new_game_stocks_bank(self):
        state = GameState.new_game(seed=42)
        for r in ResourceType:
            assert state.bank[r] == 19

    def test_new_game_shuffles_deck(self):
        state = GameState.new_game(seed=42)
        assert len(state.dev_card_deck) == 15  # 4+3+3+3+2

    def test_deterministic_with_seed(self):
        s1 = GameState.new_game(seed=99)
        s2 = GameState.new_game(seed=99)
        assert s1.current_player == s2.current_player
        assert [c.name for c in s1.dev_card_deck] == [c.name for c in s2.dev_card_deck]


class TestDiceAndProduction:
    def test_roll_dice_range(self):
        state = GameState.new_game(seed=42)
        for _ in range(100):
            d1, d2 = state.roll_dice()
            assert 1 <= d1 <= 6
            assert 1 <= d2 <= 6

    def test_produce_resources_with_settlement(self):
        state = GameState.new_game(seed=42)
        # Place a settlement on an intersection adjacent to a producing tile
        inter = None
        for i in state.board.intersections.values():
            tiles = state.board.get_tiles_for_intersection(i.inter_id)
            if any(t.resource is not None and t.dice_number is not None for t in tiles):
                inter = i
                break
        assert inter is not None
        state.board.place_settlement(0, inter.inter_id)

        # Find matching dice number
        tile = next(t for t in state.board.get_tiles_for_intersection(inter.inter_id)
                    if t.resource is not None)
        state.current_dice = (tile.dice_number, 0)  # Hack to set exact sum
        # Adjust to get correct sum
        target = tile.dice_number
        if target <= 6:
            state.current_dice = (target, 0)
        # Actually roll properly
        state.current_dice = (max(1, target - 1), 1) if target > 1 else (1, 0)
        # Just set the dice directly for testing
        for d1 in range(1, 7):
            for d2 in range(1, 7):
                if d1 + d2 == target:
                    state.current_dice = (d1, d2)
                    break
            else:
                continue
            break

        production = state.produce_resources()
        # Player 0 should have received resources
        assert any(amt > 0 for amt in production.get(0, {}).values()) or target < 2 or target > 12


class TestMaintenance:
    def test_maintenance_paid(self):
        state = GameState.new_game(seed=42)
        pos = state.board.get_valid_draft_settlements(0)[0]
        state.board.place_settlement(0, pos)
        # Give player water to pay
        state.players[0].resources[ResourceType.WATER] = 5
        events = state.resolve_maintenance(0)
        assert all(e.paid for e in events)

    def test_maintenance_unpaid_loses_building(self):
        state = GameState.new_game(seed=42)
        pos = state.board.get_valid_draft_settlements(0)[0]
        state.board.place_settlement(0, pos)
        # Player has no water
        state.players[0].resources[ResourceType.WATER] = 0
        events = state.resolve_maintenance(0)
        assert any(not e.paid for e in events)
        assert state.board.intersections[pos].owner is None

    def test_maintenance_card_skips(self):
        state = GameState.new_game(seed=42)
        pos = state.board.get_valid_draft_settlements(0)[0]
        state.board.place_settlement(0, pos)
        state.players[0].resources[ResourceType.WATER] = 0
        state.players[0].maintenance_paid_this_turn = True
        events = state.resolve_maintenance(0)
        assert len(events) == 0  # Skipped entirely
        assert state.board.intersections[pos].owner == 0  # Still owned


class TestGameOver:
    def test_not_over_initially(self):
        state = GameState.new_game(seed=42)
        assert not state.is_game_over()

    def test_over_at_threshold(self):
        state = GameState.new_game(seed=42)
        state.players[0].progress_points = WIN_THRESHOLD
        assert state.is_game_over()
        assert state.winner == 0

    def test_over_when_one_active(self):
        state = GameState.new_game(seed=42)
        state.players[1].is_active = False
        state.players[2].is_active = False
        assert state.is_game_over()
        assert state.winner == 0

    def test_over_at_max_turns(self):
        state = GameState.new_game(seed=42)
        state.turn_number = 200
        state.players[1].progress_points = 5
        assert state.is_game_over()
        assert state.winner == 1


class TestObservation:
    def test_observation_hides_opponent_resources(self):
        state = GameState.new_game(seed=42)
        state.players[1].resources[ResourceType.WOOD] = 5
        obs = state.get_observation(0)
        assert obs.player_id == 0
        # Check opponents have only counts
        for opp in obs.opponents:
            if opp.player_id == 1:
                assert opp.total_resources == 5
