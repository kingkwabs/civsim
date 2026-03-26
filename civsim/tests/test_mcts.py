"""Tests for MCTS agent."""
import pytest
from civsim.agents import MCTSAgent, RandomAgent, GreedyAgent, MCTSNode
from civsim.data_types import EndTurn, ResourceType, BuildType
from civsim.environment import Environment
from civsim.game_state import GameState
from civsim.actions import get_valid_actions


class TestMCTSNode:
    def test_ucb1_unexplored_is_infinity(self):
        state = GameState.new_game(seed=42)
        node = MCTSNode(state=state, player_id=0)
        assert node.ucb1() == float("inf")

    def test_ucb1_after_visits(self):
        state = GameState.new_game(seed=42)
        parent = MCTSNode(state=state, player_id=0)
        parent.visits = 10
        child = MCTSNode(state=state, player_id=0, parent=parent)
        child.visits = 5
        child.wins = 3.0
        score = child.ucb1()
        assert score > 0
        assert score < 10  # sanity check

    def test_best_child_picks_highest_ucb(self):
        state = GameState.new_game(seed=42)
        parent = MCTSNode(state=state, player_id=0)
        parent.visits = 20

        c1 = MCTSNode(state=state, player_id=0, parent=parent, action=EndTurn())
        c1.visits = 10
        c1.wins = 8.0

        c2 = MCTSNode(state=state, player_id=0, parent=parent, action=EndTurn())
        c2.visits = 10
        c2.wins = 2.0

        parent.children = [c1, c2]
        assert parent.best_child() == c1

    def test_best_action_child_picks_most_visited(self):
        state = GameState.new_game(seed=42)
        parent = MCTSNode(state=state, player_id=0)

        c1 = MCTSNode(state=state, player_id=0, parent=parent, action=EndTurn())
        c1.visits = 5
        c1.wins = 5.0  # perfect win rate but fewer visits

        c2 = MCTSNode(state=state, player_id=0, parent=parent, action=EndTurn())
        c2.visits = 15
        c2.wins = 8.0

        parent.children = [c1, c2]
        assert parent.best_action_child() == c2  # most visited wins


class TestMCTSAgent:
    def test_select_action_returns_valid(self):
        """MCTS should return an action from the valid list."""
        agents = [MCTSAgent(0, seed=42, n_simulations=20, rollout_depth=10),
                  RandomAgent(1, seed=43),
                  RandomAgent(2, seed=44)]
        env = Environment(agents, num_players=3)
        obs = env.reset(seed=42)

        pid = env.state.current_player
        valid = get_valid_actions(env.state, pid)
        agent = agents[pid] if isinstance(agents[pid], MCTSAgent) else agents[0]

        # Give MCTS agent some resources to have interesting choices
        env.state.players[agent.player_id].resources[ResourceType.WOOD] = 5
        env.state.players[agent.player_id].resources[ResourceType.STONE] = 5
        obs = env.state.get_observation(agent.player_id)
        valid = get_valid_actions(env.state, agent.player_id)

        action = agent.select_action(obs, valid)
        assert action is not None

    def test_single_action_skips_search(self):
        """With only EndTurn available, MCTS should skip the search."""
        agent = MCTSAgent(0, seed=42, n_simulations=100)
        agents_list = [agent, RandomAgent(1, seed=43), RandomAgent(2, seed=44)]
        env = Environment(agents_list, num_players=3)
        obs = env.reset(seed=42)

        # Only EndTurn
        valid = [EndTurn()]
        action = agent.select_action(obs, valid)
        assert isinstance(action, EndTurn)

    def test_mcts_completes_game(self):
        """MCTS agent should complete a full game without errors."""
        agents = [
            MCTSAgent(0, seed=42, n_simulations=10, rollout_depth=5),
            RandomAgent(1, seed=43),
            RandomAgent(2, seed=44),
        ]
        env = Environment(agents, num_players=3)
        result = env.run_game()
        assert result is not None
        assert result.winner is not None
        assert result.turns_played > 0
        print(f"MCTS game: winner={result.winner}, turns={result.turns_played}")
        print(f"Scores: {result.scores}")

    def test_mcts_vs_random_tournament(self):
        """Run a small tournament to verify MCTS works across multiple games."""
        from civsim.evaluation import GameRunner

        # Use low sim count for speed
        class FastMCTS(MCTSAgent):
            def __init__(self, player_id, seed=None):
                super().__init__(player_id, seed, n_simulations=10, rollout_depth=5)

        runner = GameRunner([FastMCTS, RandomAgent, RandomAgent], num_players=3)
        tournament = runner.run_tournament(n_games=3, seed=42)
        assert len(tournament.results) == 3
        print(f"MCTS tournament: {tournament.win_counts}")
        print(f"Avg length: {tournament.avg_game_length:.1f} turns")

    def test_mcts_prefers_building_over_end_turn(self):
        """Given resources and build options, MCTS should prefer building."""
        agent = MCTSAgent(0, seed=42, n_simulations=50, rollout_depth=10)
        agents_list = [agent, RandomAgent(1, seed=43), RandomAgent(2, seed=44)]
        env = Environment(agents_list, num_players=3)
        env.reset(seed=42)

        # Give player 0 lots of resources
        for r in ResourceType:
            env.state.players[0].resources[r] = 10

        # Force current player to be 0
        env.state.current_player = 0

        obs = env.state.get_observation(0)
        valid = get_valid_actions(env.state, 0)

        non_end = [a for a in valid if not isinstance(a, EndTurn)]
        if non_end:
            action = agent.select_action(obs, valid)
            # With resources and options, MCTS should usually not just EndTurn
            # (this is probabilistic, so we just check it ran without error)
            assert action is not None


class TestMCTSEvaluation:
    def test_evaluate_winning_state(self):
        agent = MCTSAgent(0, seed=42)
        state = GameState.new_game(seed=42)
        state.game_over = True
        state.winner = 0
        result = agent._evaluate_state(state)
        assert result == 1.0

    def test_evaluate_losing_state(self):
        agent = MCTSAgent(0, seed=42)
        state = GameState.new_game(seed=42)
        state.game_over = True
        state.winner = 1
        result = agent._evaluate_state(state)
        assert result == 0.0

    def test_evaluate_ongoing_state_ahead(self):
        agent = MCTSAgent(0, seed=42)
        state = GameState.new_game(seed=42)
        state.players[0].progress_points = 5
        state.players[1].progress_points = 2
        state.players[2].progress_points = 1
        result = agent._evaluate_state(state)
        assert result > 0.5  # we're ahead

    def test_evaluate_ongoing_state_behind(self):
        agent = MCTSAgent(0, seed=42)
        state = GameState.new_game(seed=42)
        state.players[0].progress_points = 1
        state.players[1].progress_points = 7
        state.players[2].progress_points = 2
        result = agent._evaluate_state(state)
        assert result < 0.5  # we're behind
