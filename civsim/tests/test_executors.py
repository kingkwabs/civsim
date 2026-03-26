"""Tests for action execution logic."""
import pytest
from civsim.action_executors import (
    execute_build, execute_buy_dev_card, execute_divine_intervention,
    execute_trade, execute_play_dev_card, _deduct_resources,
)
from civsim.data_types import (
    Build, BuildType, DevCardType, PlayDevCard, ResourceType, TradeProposal,
)
from civsim.game_state import GameState
from civsim.agents import RandomAgent


@pytest.fixture
def state_with_settlement():
    """Game state with player 0 having a settlement and resources."""
    state = GameState.new_game(seed=42)
    pos = state.board.get_valid_draft_settlements(0)[0]
    state.board.place_settlement(0, pos)
    state.players[0].progress_points = 1
    # Give plenty of resources
    for r in ResourceType:
        state.players[0].resources[r] = 10
    return state


class TestExecuteBuild:
    def test_build_road(self, state_with_settlement):
        state = state_with_settlement
        roads = state.board.get_valid_road_positions(0)
        assert len(roads) > 0
        action = Build(BuildType.ROAD, roads[0])
        result = execute_build(state, 0, action)
        assert result.success
        assert state.board.edges[roads[0]].road_owner == 0

    def test_build_settlement_adds_point(self, state_with_settlement):
        state = state_with_settlement
        # Place a road first to enable settlement
        roads = state.board.get_valid_road_positions(0)
        state.board.place_road(0, roads[0])
        settlements = state.board.get_valid_settlement_positions(0)
        if settlements:
            old_points = state.players[0].progress_points
            action = Build(BuildType.SETTLEMENT, settlements[0])
            result = execute_build(state, 0, action)
            assert result.success
            assert state.players[0].progress_points == old_points + 1

    def test_upgrade_to_city(self, state_with_settlement):
        state = state_with_settlement
        cities = state.board.get_valid_city_positions(0)
        assert len(cities) > 0
        old_points = state.players[0].progress_points
        action = Build(BuildType.CITY, cities[0])
        result = execute_build(state, 0, action)
        assert result.success
        assert state.players[0].progress_points == old_points + 1


class TestExecuteTrade:
    def test_bank_trade(self, state_with_settlement):
        state = state_with_settlement
        action = TradeProposal(
            offering={ResourceType.WOOD: 4},
            requesting={ResourceType.METAL: 1},
        )
        old_wood = state.players[0].resources[ResourceType.WOOD]
        result = execute_trade(state, 0, action, None)
        assert result.success
        assert state.players[0].resources[ResourceType.WOOD] == old_wood - 4
        assert state.players[0].resources[ResourceType.METAL] >= 1


class TestExecuteBuyDevCard:
    def test_buy_dev_card(self, state_with_settlement):
        state = state_with_settlement
        deck_size = len(state.dev_card_deck)
        result = execute_buy_dev_card(state, 0)
        assert result.success
        assert len(state.dev_card_deck) == deck_size - 1
        assert len(state.players[0].dev_cards) == 1


class TestExecutePlayDevCard:
    def test_espionage(self, state_with_settlement):
        state = state_with_settlement
        state.players[1].resources[ResourceType.WOOD] = 5
        state.players[0].dev_cards.append(DevCardType.ESPIONAGE)
        agents = {i: RandomAgent(i, seed=42) for i in range(3)}
        action = PlayDevCard(DevCardType.ESPIONAGE)
        result = execute_play_dev_card(state, 0, action, agents)
        assert result.success

    def test_invention(self, state_with_settlement):
        state = state_with_settlement
        state.players[0].dev_cards.append(DevCardType.INVENTION)
        agents = {i: RandomAgent(i, seed=42) for i in range(3)}
        action = PlayDevCard(DevCardType.INVENTION)
        result = execute_play_dev_card(state, 0, action, agents)
        assert result.success

    def test_maintenance_card(self, state_with_settlement):
        state = state_with_settlement
        state.players[0].dev_cards.append(DevCardType.MAINTENANCE)
        agents = {i: RandomAgent(i, seed=42) for i in range(3)}
        action = PlayDevCard(DevCardType.MAINTENANCE)
        result = execute_play_dev_card(state, 0, action, agents)
        assert result.success
        assert state.players[0].maintenance_paid_this_turn


class TestExecuteDivineIntervention:
    def test_divine_intervention(self, state_with_settlement):
        state = state_with_settlement
        result = execute_divine_intervention(state, 0)
        assert result.success
        assert state.players[0].resources[ResourceType.COW] == 8  # paid 2


class TestHelpers:
    def test_deduct_resources(self):
        state = GameState.new_game(seed=42)
        state.players[0].resources[ResourceType.WOOD] = 5
        _deduct_resources(state, 0, {ResourceType.WOOD: 3})
        assert state.players[0].resources[ResourceType.WOOD] == 2
