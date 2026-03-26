"""End-to-end tests: full game + tournament."""
import pytest
from civsim.agents import RandomAgent, GreedyAgent
from civsim.environment import Environment
from civsim.evaluation import GameRunner


class TestEndToEnd:
    def test_3_random_agents_game(self):
        agents = [RandomAgent(i, seed=i * 10) for i in range(3)]
        env = Environment(agents, num_players=3)
        result = env.run_game()
        assert result.winner is not None
        assert result.turns_played > 0
        assert all(pid in result.scores for pid in range(3))
        print(f"Winner: Player {result.winner}, Turns: {result.turns_played}")
        print(f"Scores: {result.scores}")

    def test_2_player_game(self):
        agents = [RandomAgent(i, seed=i * 10) for i in range(2)]
        env = Environment(agents, num_players=2)
        result = env.run_game()
        assert result.winner is not None

    def test_tournament_runs(self):
        runner = GameRunner([RandomAgent, GreedyAgent], num_players=3)
        tournament = runner.run_tournament(n_games=5, seed=42)
        assert len(tournament.results) == 5
        assert tournament.avg_game_length > 0
        print(f"Tournament: {tournament.win_counts}")
        print(f"Avg length: {tournament.avg_game_length:.1f} turns")
        print(f"Time: {tournament.total_time:.2f}s")
        print(runner.metrics.summary())
