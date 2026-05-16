"""Tests for target-aware GreedyAgent scoring."""
from __future__ import annotations

from civsim.actions import get_valid_actions
from civsim.agents import GreedyAgent
from civsim.data_types import (
    Build, BuildType, BuyDevCard, DivineIntervention, EndTurn,
    PlayDevCard, DevCardType, ResourceType, WIN_THRESHOLD,
)
from civsim.environment import Environment
from civsim.game_state import GameState


def _agent_and_obs(seed: int = 0) -> tuple[GreedyAgent, GameState]:
    state = GameState.new_game(num_players=3, seed=seed)
    agent = GreedyAgent(player_id=0, seed=seed)
    return agent, state


def test_urgency_zero_with_no_progress():
    agent, state = _agent_and_obs()
    obs = state.get_observation(0)
    assert agent._urgency(obs) == 0.0


def test_urgency_rises_with_progress_points():
    from civsim.data_types import SETTLEMENT_PP
    agent, state = _agent_and_obs(seed=2)
    inters = list(state.board.intersections.keys())
    # Plant a handful of settlements
    for iid in inters[:3]:
        state.board.place_settlement(0, iid)
    obs = state.get_observation(0)
    expected = (3 * SETTLEMENT_PP) / WIN_THRESHOLD
    assert abs(agent._urgency(obs) - expected) < 1e-9


def test_urgency_clamped_at_one_when_at_threshold():
    agent, state = _agent_and_obs(seed=3)
    inters = list(state.board.intersections.keys())
    for iid in inters[:5]:
        state.board.place_settlement(0, iid)
        state.board.upgrade_to_city(0, iid)  # each city = 2 PP → 10 total
    obs = state.get_observation(0)
    assert agent._urgency(obs) == 1.0


def test_city_outranks_settlement_when_urgent_with_sustain_buffer():
    """At urgency=1.0 city scoring should beat settlement scoring when the
    player has enough cow/water reserves to actually sustain the new
    building (without the buffer, the sustainability guard fires instead)."""
    agent = GreedyAgent(player_id=0)
    state = GameState.new_game(num_players=3, seed=4)
    # Give the player enough cow and water to pass the sustain checks
    state.players[0].resources[ResourceType.COW] = 3
    state.players[0].resources[ResourceType.WATER] = 3
    obs = state.get_observation(0)

    city = Build(BuildType.CITY, 0)
    settlement = Build(BuildType.SETTLEMENT, 0)
    city_low = agent._score_action(city, obs, {}, {}, 0, urgency=0.0)
    sett_low = agent._score_action(settlement, obs, {}, {}, 0, urgency=0.0)
    city_high = agent._score_action(city, obs, {}, {}, 0, urgency=1.0)
    sett_high = agent._score_action(settlement, obs, {}, {}, 0, urgency=1.0)

    assert city_low > sett_low
    assert city_high > sett_high
    # The gap should widen with urgency (cities scale up faster)
    assert (city_high - sett_high) > (city_low - sett_low)


def test_city_score_drops_when_cow_buffer_unavailable():
    """Sustainability guard: refuse to be excited about cities with no cow."""
    agent = GreedyAgent(player_id=0)
    state = GameState.new_game(num_players=3, seed=5)
    state.players[0].resources[ResourceType.COW] = 0
    obs = state.get_observation(0)
    city = Build(BuildType.CITY, 0)
    score = agent._score_action(city, obs, {}, {}, 0, urgency=1.0)
    # Without cow buffer, city should score below normal-urgency baseline
    assert score < 100.0


def test_road_scoring_drops_at_high_urgency():
    """Roads should become much less attractive when close to winning."""
    agent = GreedyAgent(player_id=0)
    state = GameState.new_game(num_players=3, seed=5)
    obs = state.get_observation(0)
    road = Build(BuildType.ROAD, 0)
    low = agent._score_action(road, obs, {}, {}, 0, urgency=0.0)
    high = agent._score_action(road, obs, {}, {}, 0, urgency=1.0)
    assert low > high
    # At urgency=1.0 roads should fall below at least settlement (80+40=120)
    settlement = Build(BuildType.SETTLEMENT, 0)
    assert high < agent._score_action(settlement, obs, {}, {}, 0, urgency=1.0)


def test_target_aware_greedy_builds_cities():
    """Target-aware Greedy should build cities more often than non-target
    behavior. (Threshold wins remain out of reach without lookahead — see
    notes — but city construction is the key target-aware signal.)"""
    total_cities = 0
    games = 15
    for seed in range(games):
        agents = [GreedyAgent(player_id=i, seed=seed * 13 + i) for i in range(3)]
        env = Environment(agents, num_players=3)
        r = env.run_game()
        total_cities += sum(
            s.buildings_built.get(BuildType.CITY, 0) for s in r.player_stats.values()
        )
    avg = total_cities / games
    # Pre-target-aware Greedy averaged ~1.9 city builds/game.
    # With target-awareness this jumps to ~4. Verify the city-building
    # behavior is on rather than testing a noisy win-rate target.
    assert avg >= 2.5, (
        f"Expected target-aware Greedy to build >=2.5 cities/game; got {avg:.1f}"
    )
