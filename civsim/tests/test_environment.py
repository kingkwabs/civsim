"""Tests for environment orchestration."""
import pytest
from civsim.agents import RandomAgent, GreedyAgent
from civsim.data_types import BuildType, ResourceType
from civsim.environment import Environment


class TestEnvironmentReset:
    def test_reset_returns_observation(self):
        agents = [RandomAgent(i, seed=42 + i) for i in range(3)]
        env = Environment(agents, num_players=3)
        obs = env.reset(seed=42)
        assert obs is not None
        assert obs.player_id == env.state.current_player

    def test_snake_draft_places_settlements(self):
        agents = [RandomAgent(i, seed=42 + i) for i in range(3)]
        env = Environment(agents, num_players=3)
        env.reset(seed=42)
        # Each player should have 2 settlements after draft
        for pid in range(3):
            settlements = [
                i for i in env.state.board.intersections.values()
                if i.owner == pid and i.building == BuildType.SETTLEMENT
            ]
            assert len(settlements) == 2, f"Player {pid} has {len(settlements)} settlements"

    def test_snake_draft_places_roads(self):
        agents = [RandomAgent(i, seed=42 + i) for i in range(3)]
        env = Environment(agents, num_players=3)
        env.reset(seed=42)
        for pid in range(3):
            roads = [e for e in env.state.board.edges.values() if e.road_owner == pid]
            assert len(roads) == 2, f"Player {pid} has {len(roads)} roads"

    def test_starting_resources_distributed(self):
        agents = [RandomAgent(i, seed=42 + i) for i in range(3)]
        env = Environment(agents, num_players=3)
        env.reset(seed=42)
        # Players should have some starting resources
        for pid in range(3):
            total = sum(env.state.players[pid].resources.values())
            assert total > 0, f"Player {pid} has no starting resources"


class TestEnvironmentStep:
    def test_step_end_turn(self):
        agents = [RandomAgent(i, seed=42 + i) for i in range(3)]
        env = Environment(agents, num_players=3)
        env.reset(seed=42)
        from civsim.data_types import EndTurn
        obs, reward, done, info = env.step(EndTurn())
        assert not done or env.state.game_over


class TestFullGameLoop:
    def test_random_agents_complete_game(self):
        agents = [RandomAgent(i, seed=42 + i) for i in range(3)]
        env = Environment(agents, num_players=3)
        result = env.run_game()
        assert result is not None
        assert result.winner is not None
        assert result.turns_played > 0

    def test_greedy_agents_complete_game(self):
        agents = [GreedyAgent(i, seed=42 + i) for i in range(3)]
        env = Environment(agents, num_players=3)
        result = env.run_game()
        assert result is not None
        assert result.winner is not None

    def test_mixed_agents_complete_game(self):
        agents = [
            RandomAgent(0, seed=42),
            GreedyAgent(1, seed=43),
            RandomAgent(2, seed=44),
        ]
        env = Environment(agents, num_players=3)
        result = env.run_game()
        assert result is not None
        assert result.winner is not None

    def test_deterministic_games(self):
        agents1 = [RandomAgent(i, seed=100 + i) for i in range(3)]
        env1 = Environment(agents1, num_players=3)
        r1 = env1.run_game()

        agents2 = [RandomAgent(i, seed=100 + i) for i in range(3)]
        env2 = Environment(agents2, num_players=3)
        r2 = env2.run_game()

        # Same seeds should produce same results
        # (Note: may differ due to dict ordering, but winner should match)
        assert r1.winner == r2.winner
        assert r1.turns_played == r2.turns_played
