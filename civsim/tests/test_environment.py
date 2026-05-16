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
        env1.reset(seed=42)
        while not env1.state.game_over:
            pid = env1.state.current_player
            from civsim.actions import get_valid_actions
            from civsim.action_executors import execute_action
            from civsim.data_types import EndTurn
            obs = env1.state.get_observation(pid)
            action = env1.agents[pid].select_action(obs, get_valid_actions(env1.state, pid))
            if isinstance(action, EndTurn):
                env1._end_turn()
            else:
                execute_action(env1.state, pid, action, env1.agents)
        r1 = env1.state.get_final_results()

        agents2 = [RandomAgent(i, seed=100 + i) for i in range(3)]
        env2 = Environment(agents2, num_players=3)
        env2.reset(seed=42)
        while not env2.state.game_over:
            pid = env2.state.current_player
            from civsim.actions import get_valid_actions
            from civsim.action_executors import execute_action
            from civsim.data_types import EndTurn
            obs = env2.state.get_observation(pid)
            action = env2.agents[pid].select_action(obs, get_valid_actions(env2.state, pid))
            if isinstance(action, EndTurn):
                env2._end_turn()
            else:
                execute_action(env2.state, pid, action, env2.agents)
        r2 = env2.state.get_final_results()

        # Same env seed + same agent seeds → identical outcomes
        assert r1.winner == r2.winner
        assert r1.turns_played == r2.turns_played
