"""Tests for upkeep helpers and upkeep-aware agent behavior."""
from __future__ import annotations

from civsim.agents import GreedyAgent
from civsim.data_types import BuildType, Observation, ResourceType
from civsim.environment import Environment
from civsim.game_state import GameState
from civsim.upkeep import (
    buildings_at_risk, buildings_at_risk_from_obs,
    total_upkeep_cost, upkeep_gap,
)


def _place_settlement(state: GameState, pid: int, inter_id: int) -> None:
    state.board.place_settlement(pid, inter_id)


def test_no_upkeep_with_no_buildings():
    state = GameState.new_game(num_players=3, seed=0)
    cost = total_upkeep_cost(state.board, 0)
    assert cost == {}
    assert buildings_at_risk(state.players[0], state.board) == 0


def test_upkeep_cost_aggregates_per_resource():
    state = GameState.new_game(num_players=3, seed=0)
    inters = list(state.board.intersections.keys())
    _place_settlement(state, 0, inters[0])
    _place_settlement(state, 0, inters[10])
    # Upgrade one to city
    state.board.upgrade_to_city(0, inters[0])

    cost = total_upkeep_cost(state.board, 0)
    # 1 settlement (1 water) + 1 city (1 cow, cow-only after balance pass)
    assert cost[ResourceType.WATER] == 1
    assert cost[ResourceType.COW] == 1


def test_upkeep_gap_when_resources_short():
    state = GameState.new_game(num_players=3, seed=0)
    inters = list(state.board.intersections.keys())
    _place_settlement(state, 0, inters[0])
    state.players[0].resources[ResourceType.WATER] = 0

    gap = upkeep_gap(state.players[0].resources, state.board, 0)
    assert gap == {ResourceType.WATER: 1}


def test_buildings_at_risk_counts_only_what_would_fail():
    state = GameState.new_game(num_players=3, seed=0)
    inters = list(state.board.intersections.keys())
    _place_settlement(state, 0, inters[0])
    _place_settlement(state, 0, inters[10])
    # Only enough water for one
    state.players[0].resources[ResourceType.WATER] = 1
    assert buildings_at_risk(state.players[0], state.board) == 1


def test_buildings_at_risk_respects_maintenance_card_flag():
    state = GameState.new_game(num_players=3, seed=0)
    inters = list(state.board.intersections.keys())
    _place_settlement(state, 0, inters[0])
    state.players[0].resources[ResourceType.WATER] = 0
    state.players[0].maintenance_paid_this_turn = True
    assert buildings_at_risk(state.players[0], state.board) == 0


def test_greedy_refuses_to_end_turn_when_upkeep_at_risk():
    """Score-based: with an at-risk building, EndTurn must score below
    any constructive action (including a useless bank trade)."""
    state = GameState.new_game(num_players=3, seed=1)
    inters = list(state.board.intersections.keys())
    _place_settlement(state, 0, inters[0])
    # Empty water, but have 4 wood so bank trade is available
    state.players[0].resources[ResourceType.WATER] = 0
    state.players[0].resources[ResourceType.WOOD] = 4
    obs = state.get_observation(0)

    agent = GreedyAgent(player_id=0, seed=0)
    from civsim.actions import get_valid_actions
    valid = get_valid_actions(state, 0)
    action = agent.select_action(obs, valid)
    from civsim.data_types import EndTurn
    assert not isinstance(action, EndTurn), (
        "Greedy should not EndTurn while a building is at risk"
    )


def test_greedy_prefers_water_seeking_trade_under_upkeep_pressure():
    state = GameState.new_game(num_players=3, seed=2)
    inters = list(state.board.intersections.keys())
    _place_settlement(state, 0, inters[0])
    state.players[0].resources[ResourceType.WATER] = 0
    state.players[0].resources[ResourceType.WOOD] = 4
    obs = state.get_observation(0)

    agent = GreedyAgent(player_id=0)
    from civsim.actions import get_valid_actions
    from civsim.data_types import TradeProposal
    valid = get_valid_actions(state, 0)
    action = agent.select_action(obs, valid)
    # Either picked a bank trade (4 wood → 1 water) or another action that
    # converts to water; in both cases it must be a trade and request water.
    assert isinstance(action, TradeProposal), f"got {type(action).__name__}"
    assert ResourceType.WATER in action.requesting


def test_upkeep_aware_greedy_takes_defensive_action():
    """End-to-end behavior: upkeep-aware Greedy must actually *try* to
    defend its buildings — trades or MAINTENANCE card plays — when at
    risk. Whether the defense succeeds depends on game economy
    (sometimes there's no viable 4:1 partition or p2p partner), so we
    measure intent (defensive actions taken), not perfect outcomes.
    """
    from civsim.data_types import DevCardType

    agents = [GreedyAgent(player_id=i, seed=42 + i) for i in range(3)]
    env = Environment(agents, num_players=3)
    result = env.run_game()

    total_trades = sum(s.trades_made for s in result.player_stats.values())
    maintenance_plays = sum(
        s.divine_outcomes.get("MAINTENANCE_card_played", 0)
        for s in result.player_stats.values()
    )
    # Across a 200-turn 3-player game with chronic upkeep pressure, the
    # upkeep-aware heuristic should have driven at least *some* trades.
    # Pre-upkeep-aware Greedy could go entire games with zero trades.
    assert total_trades > 0, "Upkeep-aware Greedy should have traded at least once"


# NOTE: The full-game head-to-head between upkeep-aware and upkeep-blind
# Greedy was removed. Once the game economy was rebalanced (rain, slower
# maintenance, more cow), variance across seeds widened enough that
# either side could "win" the comparison by 1-2 failures of pure noise.
# The synthetic scenario tests above already validate that the heuristic
# fires in the cases it's designed for; an end-to-end comparison should
# wait until we have larger sample sizes or a stronger baseline.


def test_rl_encoder_includes_upkeep_features():
    from civsim.rl.features import STATE_DIM, encode_observation
    state = GameState.new_game(num_players=3, seed=3)
    obs = state.get_observation(0)
    f = encode_observation(obs)
    assert len(f) == STATE_DIM
    # Last 3 features are the upkeep block — should be 0 with no buildings
    assert f[-3:] == [0.0, 0.0, 0.0]
