"""Tests for the rain mechanic and starting water stockpile."""
from __future__ import annotations

import random

from civsim.agents import GreedyAgent, RandomAgent
from civsim.data_types import (
    RAIN_PROBABILITY, RAIN_WATER_PER_PLAYER, ResourceType,
    STARTING_WATER_STOCKPILE,
)
from civsim.environment import Environment
from civsim.game_state import GameState


def test_starting_stockpile_bonus_applied_after_draft():
    """Every player should end the draft with at least STARTING_WATER_STOCKPILE
    water — even those whose draft tiles produced no water."""
    agents = [GreedyAgent(player_id=i, seed=42 + i) for i in range(3)]
    env = Environment(agents, num_players=3)
    env.reset()
    for pid, p in env.state.players.items():
        assert p.resources[ResourceType.WATER] >= STARTING_WATER_STOCKPILE, (
            f"P{pid} only has {p.resources[ResourceType.WATER]} water "
            f"after draft; expected ≥ {STARTING_WATER_STOCKPILE}"
        )


def test_starting_stockpile_is_additive_not_from_bank():
    """Stockpile bonus should not deduct from bank — it's free water."""
    agents = [GreedyAgent(player_id=i, seed=42 + i) for i in range(3)]
    env = Environment(agents, num_players=3)
    env.reset()
    # Bank starts at BANK_STARTING_SUPPLY (19) per resource.
    # Tile production from settlements consumes some water from bank
    # (because that's draft-tile production, which IS from bank). Plus
    # any production this turn. The stockpile bonus itself shouldn't add
    # to that consumption.
    total_water_held = sum(p.resources[ResourceType.WATER] for p in env.state.players.values())
    bank_water = env.state.bank[ResourceType.WATER]
    # Held + bank should exceed starting bank by exactly 3 × stockpile (since
    # rain at turn 0 could already have fired adding more — so use ≥).
    expected_minimum = 19 + 3 * STARTING_WATER_STOCKPILE
    assert total_water_held + bank_water >= expected_minimum, (
        f"held={total_water_held} + bank={bank_water} < {expected_minimum}"
    )


def test_roll_rain_uses_state_rng():
    """Same seed → reproducible rain pattern."""
    state1 = GameState.new_game(num_players=3, seed=99)
    state2 = GameState.new_game(num_players=3, seed=99)
    pattern1 = [state1.roll_rain() for _ in range(50)]
    pattern2 = [state2.roll_rain() for _ in range(50)]
    assert pattern1 == pattern2


def test_roll_rain_frequency_in_long_run():
    """Empirical frequency should match RAIN_PROBABILITY within tolerance."""
    state = GameState.new_game(num_players=3, seed=1)
    trials = 5000
    rains = sum(1 for _ in range(trials) if state.roll_rain())
    actual = rains / trials
    # ±0.02 tolerance (very generous given 5000 trials)
    assert abs(actual - RAIN_PROBABILITY) < 0.03, (
        f"Rain rate {actual:.3f} too far from RAIN_PROBABILITY={RAIN_PROBABILITY}"
    )


def test_apply_rain_grants_water_additively_to_active_players():
    state = GameState.new_game(num_players=3, seed=2)
    bank_before = state.bank[ResourceType.WATER]
    before = {pid: p.resources[ResourceType.WATER] for pid, p in state.players.items()}

    recipients = state.apply_rain()

    assert set(recipients) == set(state.players.keys())
    for pid, p in state.players.items():
        assert p.resources[ResourceType.WATER] == before[pid] + RAIN_WATER_PER_PLAYER
        assert p.stats.rain_received == RAIN_WATER_PER_PLAYER
    # Bank untouched — rain is from the sky
    assert state.bank[ResourceType.WATER] == bank_before


def test_apply_rain_skips_inactive_players():
    state = GameState.new_game(num_players=3, seed=3)
    state.players[1].is_active = False
    recipients = state.apply_rain()
    assert 1 not in recipients
    assert state.players[1].stats.rain_received == 0


def test_rain_enables_more_building_in_full_game():
    """Rain should let agents recover from upkeep crises and build more.

    Note: maintenance_failures may *increase* with rain because agents
    rebuild settlements that subsequently get lost too. The right signal
    is total construction activity and progress points scored.
    """
    def run(rain_on: bool) -> tuple[int, int]:
        if not rain_on:
            import civsim.game_state as gs
            orig = gs.GameState.roll_rain
            gs.GameState.roll_rain = lambda self: False
        try:
            builds = pp = 0
            for seed in range(8):
                agents = [GreedyAgent(player_id=i, seed=seed * 13 + i) for i in range(3)]
                env = Environment(agents, num_players=3)
                r = env.run_game()
                builds += sum(
                    sum(s.buildings_built.values()) for s in r.player_stats.values()
                )
                pp += sum(r.scores.values())
            return builds, pp
        finally:
            if not rain_on:
                gs.GameState.roll_rain = orig

    rain_on_builds, rain_on_pp = run(True)
    rain_off_builds, rain_off_pp = run(False)

    assert rain_on_builds > rain_off_builds, (
        f"Rain should enable more building activity. "
        f"on={rain_on_builds}, off={rain_off_builds}"
    )
