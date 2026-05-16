"""Tests for the barn-day mechanic and cow-only city upkeep."""
from __future__ import annotations

from civsim.agents import GreedyAgent
from civsim.data_types import (
    BARN_COW_PER_PLAYER, BARN_PROBABILITY, BuildType, MAINTENANCE_COSTS,
    ResourceType,
)
from civsim.environment import Environment
from civsim.game_state import GameState


def test_city_upkeep_is_cow_only():
    """Cities should no longer cost water for upkeep — just cow."""
    cost = MAINTENANCE_COSTS[BuildType.CITY]
    assert ResourceType.WATER not in cost
    assert cost.get(ResourceType.COW) == 1


def test_settlement_upkeep_still_water_only():
    """Settlement upkeep is unchanged — water only."""
    cost = MAINTENANCE_COSTS[BuildType.SETTLEMENT]
    assert cost == {ResourceType.WATER: 1}


def test_roll_barn_day_is_reproducible():
    state1 = GameState.new_game(num_players=3, seed=99)
    state2 = GameState.new_game(num_players=3, seed=99)
    p1 = [state1.roll_barn_day() for _ in range(50)]
    p2 = [state2.roll_barn_day() for _ in range(50)]
    assert p1 == p2


def test_roll_barn_day_frequency():
    state = GameState.new_game(num_players=3, seed=1)
    trials = 5000
    hits = sum(1 for _ in range(trials) if state.roll_barn_day())
    actual = hits / trials
    assert abs(actual - BARN_PROBABILITY) < 0.03


def test_apply_barn_day_grants_cow_additively():
    state = GameState.new_game(num_players=3, seed=2)
    bank_before = state.bank[ResourceType.COW]
    before = {pid: p.resources[ResourceType.COW] for pid, p in state.players.items()}

    recipients = state.apply_barn_day()

    assert set(recipients) == set(state.players.keys())
    for pid, p in state.players.items():
        assert p.resources[ResourceType.COW] == before[pid] + BARN_COW_PER_PLAYER
        assert p.stats.barn_received == BARN_COW_PER_PLAYER
    # Bank not deducted
    assert state.bank[ResourceType.COW] == bank_before


def test_apply_barn_day_skips_inactive_players():
    state = GameState.new_game(num_players=3, seed=3)
    state.players[2].is_active = False
    recipients = state.apply_barn_day()
    assert 2 not in recipients
    assert state.players[2].stats.barn_received == 0


def test_full_game_with_barn_day_and_cow_only_cities_progresses():
    """End-to-end: agents now build cities (was ~0/game pre-cow-fix)."""
    games = 10
    total_cities = 0
    for seed in range(games):
        agents = [GreedyAgent(player_id=i, seed=seed * 23 + i) for i in range(3)]
        env = Environment(agents, num_players=3)
        r = env.run_game()
        for s in r.player_stats.values():
            total_cities += s.buildings_built.get(BuildType.CITY, 0)
    avg_cities = total_cities / games
    # Pre-cow-fix baseline: ~0 cities/game (city upkeep was dual-resource).
    # Post-fix: ~4 cities/game.
    assert avg_cities >= 2.0, f"Cities/game too low: {avg_cities:.2f}"
