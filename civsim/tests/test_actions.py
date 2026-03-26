"""Tests for valid action enumeration."""
import pytest
from civsim.actions import get_valid_actions, _player_can_afford
from civsim.data_types import (
    ActionType, BuildType, BUILDING_COSTS, DEV_CARD_COST, DevCardType,
    EndTurn, ResourceType,
)
from civsim.game_state import GameState


class TestValidActions:
    def test_always_includes_end_turn(self):
        state = GameState.new_game(seed=42)
        actions = get_valid_actions(state, 0)
        assert any(isinstance(a, EndTurn) for a in actions)

    def test_no_builds_without_resources(self):
        state = GameState.new_game(seed=42)
        actions = get_valid_actions(state, 0)
        build_actions = [a for a in actions if a.action_type == ActionType.BUILD]
        assert len(build_actions) == 0

    def test_build_road_when_affordable(self):
        state = GameState.new_game(seed=42)
        # Place settlement first
        pos = state.board.get_valid_draft_settlements(0)[0]
        state.board.place_settlement(0, pos)
        # Give resources
        state.players[0].resources[ResourceType.WOOD] = 5
        state.players[0].resources[ResourceType.STONE] = 5
        actions = get_valid_actions(state, 0)
        road_actions = [a for a in actions if a.action_type == ActionType.BUILD
                        and hasattr(a, 'build_type') and a.build_type == BuildType.ROAD]
        assert len(road_actions) > 0

    def test_divine_intervention_needs_cows(self):
        state = GameState.new_game(seed=42)
        # No cows
        actions = get_valid_actions(state, 0)
        divine = [a for a in actions if a.action_type == ActionType.DIVINE]
        assert len(divine) == 0
        # Give cows
        state.players[0].resources[ResourceType.COW] = 2
        actions = get_valid_actions(state, 0)
        divine = [a for a in actions if a.action_type == ActionType.DIVINE]
        assert len(divine) == 1

    def test_buy_dev_card_when_affordable(self):
        state = GameState.new_game(seed=42)
        state.players[0].resources[ResourceType.METAL] = 1
        state.players[0].resources[ResourceType.WHEAT] = 1
        state.players[0].resources[ResourceType.WATER] = 1
        actions = get_valid_actions(state, 0)
        buy_dev = [a for a in actions if a.action_type == ActionType.BUY_DEV]
        assert len(buy_dev) == 1

    def test_play_dev_card_from_hand(self):
        state = GameState.new_game(seed=42)
        state.players[0].dev_cards.append(DevCardType.ESPIONAGE)
        actions = get_valid_actions(state, 0)
        play_dev = [a for a in actions if a.action_type == ActionType.PLAY_DEV]
        assert len(play_dev) == 1

    def test_one_dev_card_per_turn(self):
        state = GameState.new_game(seed=42)
        state.players[0].dev_cards.append(DevCardType.ESPIONAGE)
        state.players[0].dev_card_played_this_turn = True
        actions = get_valid_actions(state, 0)
        play_dev = [a for a in actions if a.action_type == ActionType.PLAY_DEV]
        assert len(play_dev) == 0


class TestAffordability:
    def test_can_afford_exact(self):
        resources = {ResourceType.WOOD: 1, ResourceType.STONE: 1}
        cost = {ResourceType.WOOD: 1, ResourceType.STONE: 1}
        assert _player_can_afford(resources, cost)

    def test_cannot_afford(self):
        resources = {ResourceType.WOOD: 0}
        cost = {ResourceType.WOOD: 1}
        assert not _player_can_afford(resources, cost)
