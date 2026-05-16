"""Tests for the post-rain balance pass: maintenance cadence + cow tile count."""
from __future__ import annotations

from civsim.agents import GreedyAgent
from civsim.board import STANDARD_RESOURCES, Board
from civsim.data_types import (
    BuildType, MAINTENANCE_TURN_INTERVAL, ResourceType,
)
from civsim.environment import Environment
from civsim.game_state import GameState


def test_standard_layout_has_two_cow_tiles():
    """Cow tiles bumped from 1 to 2 to relieve city-upkeep bottleneck."""
    cow_count = sum(1 for r in STANDARD_RESOURCES if r == ResourceType.COW)
    assert cow_count == 2

    # Same change should be reflected in actual board generation
    board = Board.standard_layout(seed=0)
    cow_tiles = [t for t in board.tiles.values() if t.resource == ResourceType.COW]
    assert len(cow_tiles) == 2


def test_total_tiles_still_19():
    """Rebalance must not change the 19-tile board size."""
    assert len(STANDARD_RESOURCES) == 19


def test_maintenance_only_fires_every_nth_own_turn():
    """A player whose own turns_since_maintenance hasn't reached the
    interval should skip the upkeep check entirely on _end_turn."""
    agents = [GreedyAgent(player_id=i, seed=10 + i) for i in range(3)]
    env = Environment(agents, num_players=3)
    env.reset()
    pid = env.state.current_player
    p = env.state.players[pid]
    # Force a known no-upkeep state but plant a settlement so resolve
    # would *otherwise* destroy something.
    inter = next(iter(env.state.board.intersections.values()))
    env.state.board.place_settlement(pid, inter.inter_id)
    p.resources[ResourceType.WATER] = 0
    p.turns_since_maintenance = 0  # fresh — should skip this turn

    pre = p.stats.maintenance_failures
    env._end_turn()
    # If interval > 1, the immediate end_turn should not have fired
    # maintenance for this player.
    if MAINTENANCE_TURN_INTERVAL > 1:
        assert p.stats.maintenance_failures == pre, (
            "Maintenance fired despite turns_since_maintenance < interval"
        )


def test_maintenance_fires_when_interval_reached():
    agents = [GreedyAgent(player_id=i, seed=11 + i) for i in range(3)]
    env = Environment(agents, num_players=3)
    env.reset()
    pid = env.state.current_player
    p = env.state.players[pid]
    # env.reset() already placed 2 draft settlements; just zero out water
    # so we can observe maintenance firing.
    p.resources[ResourceType.WATER] = 0
    p.turns_since_maintenance = MAINTENANCE_TURN_INTERVAL - 1

    pre = p.stats.maintenance_failures
    env._end_turn()
    assert p.stats.maintenance_failures > pre, "Maintenance should have fired"
    # Counter should reset after a maintenance round
    assert p.turns_since_maintenance == 0


def test_games_actually_finish_with_progress_under_new_balance():
    """The point of the balance pass: agents should accumulate net progress
    points across a game instead of staying stuck at zero.
    """
    total_pp_across_games = 0
    games = 10  # larger sample for stability
    for seed in range(games):
        agents = [GreedyAgent(player_id=i, seed=seed * 17 + i) for i in range(3)]
        env = Environment(agents, num_players=3)
        r = env.run_game()
        total_pp_across_games += sum(r.scores.values())
    avg = total_pp_across_games / games
    # Post-rebalance (s=2 c=4 pp + maintenance interval=3) regression bar —
    # 60-game aggregate showed Greedy total_pp/game averages around ~6.
    # 2.0 is well above noise.
    assert avg > 2.0, (
        f"Expected >2.0 total progress points/game across all players; got {avg:.2f}"
    )


def test_greedy_wins_some_games_at_threshold_10():
    """With the final balance (s=2/c=4 PP, maintenance interval=3, rain +
    barn day at 33%), 3-Greedy games should produce threshold wins at a
    measurable rate. 60-game aggregate sits around 13%; 1 win in 20 is a
    permissive bar that still distinguishes 'engine produces winnable
    games' from 'no agent ever wins'."""
    from civsim.data_types import WIN_THRESHOLD
    wins = 0
    games = 20
    for seed in range(games):
        agents = [GreedyAgent(player_id=i, seed=seed * 13 + i) for i in range(3)]
        env = Environment(agents, num_players=3)
        r = env.run_game()
        if r.turns_played < 200 and max(r.scores.values()) >= WIN_THRESHOLD:
            wins += 1
    assert wins >= 1, (
        f"Expected at least 1 win in {games} Greedy games; got {wins}"
    )
