"""Tests for live PlayerStats accumulation."""
from __future__ import annotations

import random

from civsim.action_executors import execute_action
from civsim.agents import RandomAgent
from civsim.data_types import (
    Build, BuildType, BuyDevCard, DevCardType, DivineIntervention,
    DEV_CARD_COST, DIVINE_COST, PlayDevCard, ResourceType, TradeProposal,
)
from civsim.environment import Environment
from civsim.game_state import GameState


def _fresh_state(seed: int = 0) -> GameState:
    return GameState.new_game(num_players=3, seed=seed)


def _stuff(state: GameState, pid: int, **resources: int) -> None:
    for r_name, amt in resources.items():
        r = ResourceType[r_name.upper()]
        state.players[pid].resources[r] = amt


def test_player_stats_initialized():
    state = _fresh_state()
    for pid, p in state.players.items():
        assert p.stats is not None
        assert p.stats.player_id == pid
        assert p.stats.total_resources_earned == 0
        assert p.stats.total_resources_spent == 0
        assert p.stats.buildings_built == {}


def test_build_increments_built_and_spent():
    state = _fresh_state(seed=1)
    pid = 0
    _stuff(state, pid, wood=2, stone=2, wheat=1, cow=1)

    # Plant a road first so we have a valid settlement spot
    inter = next(iter(state.board.intersections.values()))
    edge_id = inter.adjacent_edges[0]
    state.board.place_road(pid, edge_id)
    state.players[pid].stats.record_build(BuildType.ROAD)  # simulate manual draft

    valid_settlements = state.board.get_valid_settlement_positions(pid)
    assert valid_settlements
    execute_action(state, pid, Build(BuildType.SETTLEMENT, valid_settlements[0]))

    stats = state.players[pid].stats
    assert stats.buildings_built[BuildType.SETTLEMENT] == 1
    # Settlement costs 1W + 1S + 1H + 1C = 4 resources
    assert stats.total_resources_spent == 4


def test_bank_trade_counts():
    state = _fresh_state()
    pid = 0
    _stuff(state, pid, wood=4)

    execute_action(state, pid, TradeProposal(
        offering={ResourceType.WOOD: 4},
        requesting={ResourceType.METAL: 1},
    ))

    stats = state.players[pid].stats
    assert stats.trades_made == 1
    assert stats.bank_trades == 1
    assert stats.player_trades == 0
    assert stats.total_resources_spent == 4
    assert stats.total_resources_earned == 1


def test_dev_card_counters():
    state = _fresh_state(seed=2)
    pid = 0
    _stuff(state, pid, metal=1, wheat=1, water=1)

    execute_action(state, pid, BuyDevCard())
    stats = state.players[pid].stats
    assert stats.dev_cards_bought == 1
    assert stats.total_resources_spent == 3  # DEV_CARD_COST = 3 resources

    # Now play it
    card = state.players[pid].dev_cards[0]
    execute_action(state, pid, PlayDevCard(card))
    assert state.players[pid].stats.dev_cards_played == 1


def test_divine_intervention_counter():
    state = _fresh_state(seed=3)
    pid = 0
    _stuff(state, pid, cow=2)

    execute_action(state, pid, DivineIntervention())
    stats = state.players[pid].stats
    assert stats.divine_interventions == 1
    assert sum(stats.divine_outcomes.values()) == 1
    assert stats.total_resources_spent >= 2  # divine cost


def test_production_bumps_earned():
    """When dice produce resources, total_resources_earned should reflect that."""
    state = _fresh_state(seed=4)
    pid = 0
    # Place a settlement on an intersection
    inter = next(iter(state.board.intersections.values()))
    state.board.place_settlement(pid, inter.inter_id)

    # Find a dice number that produces for this settlement
    tiles = state.board.get_tiles_for_intersection(inter.inter_id)
    producing_tile = next((t for t in tiles if t.dice_number is not None), None)
    assert producing_tile is not None

    # Force dice to that value
    state.current_dice = (producing_tile.dice_number - 1, 1) \
        if producing_tile.dice_number > 1 else (1, 1)
    # Ensure sum matches
    target = producing_tile.dice_number
    state.current_dice = (target // 2, target - target // 2) if target >= 2 else (1, 1)

    before = state.players[pid].stats.total_resources_earned
    state.produce_resources()
    after = state.players[pid].stats.total_resources_earned
    assert after > before


def test_maintenance_failure_increments_lost():
    state = _fresh_state(seed=5)
    pid = 0
    inter = next(iter(state.board.intersections.values()))
    state.board.place_settlement(pid, inter.inter_id)
    # Empty player water → can't pay upkeep
    state.players[pid].resources[ResourceType.WATER] = 0

    state.resolve_maintenance(pid)
    stats = state.players[pid].stats
    assert stats.buildings_lost == 1
    assert stats.maintenance_failures == 1


def test_full_game_stats_consistency():
    """In a real game, stats should reflect actual gameplay events."""
    agents = [RandomAgent(player_id=i, seed=100 + i) for i in range(3)]
    env = Environment(agents, num_players=3)
    result = env.run_game()

    # Snake draft places 2 settlements + 2 roads per player
    for pid, stats in result.player_stats.items():
        built = stats.buildings_built
        # At least the draft buildings
        assert built.get(BuildType.SETTLEMENT, 0) >= 2, (
            f"Player {pid} should have at least 2 draft settlements: {built}"
        )
        assert built.get(BuildType.ROAD, 0) >= 2, (
            f"Player {pid} should have at least 2 draft roads: {built}"
        )
        # Each player should have earned starting resources during draft
        assert stats.total_resources_earned > 0


def test_get_final_results_uses_live_stats():
    """get_final_results should return the same PlayerStats object that was
    being accumulated, not construct a fresh one."""
    state = _fresh_state(seed=6)
    pid = 0
    _stuff(state, pid, wood=4)
    state.players[pid].stats.trades_made = 99  # synthetic marker

    result = state.get_final_results()
    assert result.player_stats[pid].trades_made == 99
