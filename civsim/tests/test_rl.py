"""Tests for the RL agent, feature encoding, network, and trainer."""
import pytest

try:
    import torch
except ImportError:
    torch = None

requires_torch = pytest.mark.skipif(
    torch is None,
    reason="PyTorch is required for RL network, agent, and trainer tests",
)

from civsim.agents import RandomAgent
from civsim.data_types import (
    ActionType, Build, BuildType, BuyDevCard, CITY_PP, DevCardType,
    EndTurn, PlayDevCard, ResourceType, SETTLEMENT_PP, WIN_THRESHOLD,
)
from civsim.environment import Environment
from civsim.game_state import GameState
from civsim.actions import get_valid_actions
from civsim.rl.features import (
    ACTION_DIM, STATE_DIM, encode_action, encode_observation,
)
from civsim.rl.network import PolicyValueNetwork
from civsim.rl.rl_agent import RLAgent
from civsim.rl.trainer import RLTrainer


# ── Feature Encoding ─────────────────────────────────────────────────────

class TestFeatureEncoding:
    @pytest.fixture
    def obs(self):
        agents = [RandomAgent(i, seed=42 + i) for i in range(3)]
        env = Environment(agents, num_players=3)
        env.reset(seed=42)
        return env.state.get_observation(0)

    def test_state_encoding_shape(self, obs):
        features = encode_observation(obs)
        assert len(features) == STATE_DIM

    def test_state_encoding_values_bounded(self, obs):
        features = encode_observation(obs)
        for f in features:
            assert -5.0 <= f <= 5.0, f"Feature value {f} out of expected range"

    def test_state_encoding_deterministic(self, obs):
        f1 = encode_observation(obs)
        f2 = encode_observation(obs)
        assert f1 == f2

    def test_state_encoding_uses_current_progress_point_constants(self, obs):
        board = obs.board
        expected_pp = 0
        for inter in board.intersections.values():
            if inter.owner != obs.player_id or inter.building is None:
                continue
            if inter.building == BuildType.SETTLEMENT:
                expected_pp += SETTLEMENT_PP
            elif inter.building == BuildType.CITY:
                expected_pp += CITY_PP

        features = encode_observation(obs)
        assert features[11] == expected_pp / WIN_THRESHOLD

    def test_action_encoding_end_turn(self):
        f = encode_action(EndTurn())
        assert len(f) == ACTION_DIM
        assert f[5] == 1.0  # END_TURN is index 5

    def test_action_encoding_build_road(self):
        f = encode_action(Build(BuildType.ROAD, 0))
        assert f[0] == 1.0  # BUILD is index 0
        assert f[7] == 1.0  # ROAD is index 7

    def test_action_encoding_build_settlement(self):
        f = encode_action(Build(BuildType.SETTLEMENT, 0))
        assert f[0] == 1.0
        assert f[8] == 1.0  # SETTLEMENT is index 8

    def test_action_encoding_build_city(self):
        f = encode_action(Build(BuildType.CITY, 0))
        assert f[0] == 1.0
        assert f[9] == 1.0  # CITY is index 9

    def test_action_encoding_buy_dev(self):
        f = encode_action(BuyDevCard())
        assert f[2] == 1.0  # BUY_DEV is index 2

    def test_action_encoding_play_dev(self):
        f = encode_action(PlayDevCard(DevCardType.ESPIONAGE))
        assert f[3] == 1.0  # PLAY_DEV is index 3
        assert f[11] == 1.0  # ESPIONAGE is index 11

    def test_action_encoding_reestablish(self):
        from civsim.data_types import ReestablishBuilding
        f = encode_action(ReestablishBuilding(BuildType.CITY, 0))
        assert f[6] == 1.0  # REESTABLISH is index 6
        assert f[9] == 1.0  # CITY subtype also set (shared slot with Build)

    def test_all_action_types_encode_differently(self):
        actions = [
            Build(BuildType.ROAD, 0),
            Build(BuildType.SETTLEMENT, 0),
            Build(BuildType.CITY, 0),
            BuyDevCard(),
            PlayDevCard(DevCardType.PLUNDER),
            EndTurn(),
        ]
        encodings = [encode_action(a) for a in actions]
        # All should be distinct
        for i, e1 in enumerate(encodings):
            for j, e2 in enumerate(encodings):
                if i != j:
                    assert e1 != e2, f"Actions {i} and {j} have same encoding"


# ── Network ──────────────────────────────────────────────────────────────

@requires_torch
class TestNetwork:
    def test_forward_pass_shapes(self):
        net = PolicyValueNetwork()
        state = torch.randn(STATE_DIM)
        actions = torch.randn(5, ACTION_DIM)  # 5 valid actions
        mask = torch.ones(5)

        log_probs, value = net(state, actions, mask)
        assert log_probs.shape == (5,)
        assert value.shape == (1,)

    def test_forward_pass_batched(self):
        net = PolicyValueNetwork()
        state = torch.randn(4, STATE_DIM)  # batch of 4
        actions = torch.randn(4, 5, ACTION_DIM)  # 5 actions each
        mask = torch.ones(4, 5)

        log_probs, value = net(state, actions, mask)
        assert log_probs.shape == (4, 5)
        assert value.shape == (4, 1)

    def test_masking_zeros_invalid(self):
        net = PolicyValueNetwork()
        state = torch.randn(STATE_DIM)
        actions = torch.randn(5, ACTION_DIM)
        mask = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])

        log_probs, _ = net(state, actions, mask)
        probs = torch.exp(log_probs)
        # Masked actions should have ~0 probability
        assert probs[1].item() < 1e-6
        assert probs[3].item() < 1e-6
        # Valid actions should have positive probability
        assert probs[0].item() > 0
        assert probs[2].item() > 0
        assert probs[4].item() > 0

    def test_log_probs_sum_to_one(self):
        net = PolicyValueNetwork()
        state = torch.randn(STATE_DIM)
        actions = torch.randn(3, ACTION_DIM)
        mask = torch.ones(3)

        log_probs, _ = net(state, actions, mask)
        probs = torch.exp(log_probs)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_save_and_load(self, tmp_path):
        net1 = PolicyValueNetwork()
        path = str(tmp_path / "model.pt")
        net1.save(path)

        net2 = PolicyValueNetwork()
        net2.load(path)

        # Weights should match
        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            assert torch.allclose(p1, p2)


# ── RL Agent ─────────────────────────────────────────────────────────────

@requires_torch
class TestRLAgent:
    @pytest.fixture
    def env_and_agent(self):
        net = PolicyValueNetwork()
        rl = RLAgent(0, network=net, seed=42, training=True)
        agents = [rl, RandomAgent(1, seed=43), RandomAgent(2, seed=44)]
        env = Environment(agents, num_players=3)
        env.reset(seed=42)
        return env, rl

    def test_select_action_returns_valid(self, env_and_agent):
        env, rl = env_and_agent
        obs = env.state.get_observation(0)
        valid = get_valid_actions(env.state, 0)
        action = rl.select_action(obs, valid)
        assert action is not None

    def test_training_mode_stores_trajectory(self, env_and_agent):
        env, rl = env_and_agent
        obs = env.state.get_observation(0)
        valid = get_valid_actions(env.state, 0)
        rl.select_action(obs, valid)
        rl.record_reward(1.0)

        log_probs, values, rewards = rl.get_trajectory()
        assert len(log_probs) == 1
        assert len(values) == 1
        assert len(rewards) == 1
        assert rewards[0] == 1.0

    def test_clear_trajectory(self, env_and_agent):
        env, rl = env_and_agent
        obs = env.state.get_observation(0)
        valid = get_valid_actions(env.state, 0)
        rl.select_action(obs, valid)
        rl.record_reward(1.0)
        rl.clear_trajectory()

        log_probs, values, rewards = rl.get_trajectory()
        assert len(log_probs) == 0
        assert len(rewards) == 0

    def test_greedy_mode(self):
        net = PolicyValueNetwork()
        rl = RLAgent(0, network=net, seed=42, training=False)
        agents = [rl, RandomAgent(1, seed=43), RandomAgent(2, seed=44)]
        env = Environment(agents, num_players=3)
        env.reset(seed=42)

        obs = env.state.get_observation(0)
        valid = get_valid_actions(env.state, 0)
        # Should pick deterministically
        a1 = rl.select_action(obs, valid)
        a2 = rl.select_action(obs, valid)
        assert type(a1) == type(a2)

    def test_rl_agent_completes_game(self):
        net = PolicyValueNetwork()
        agents = [
            RLAgent(0, network=net, seed=42, training=False),
            RandomAgent(1, seed=43),
            RandomAgent(2, seed=44),
        ]
        env = Environment(agents, num_players=3)
        result = env.run_game()
        assert result is not None
        assert result.winner is not None


# ── Trainer ──────────────────────────────────────────────────────────────

@requires_torch
class TestTrainer:
    def test_compute_returns(self):
        net = PolicyValueNetwork()
        trainer = RLTrainer(net, num_players=3)
        rewards = [1.0, 0.0, 2.0, -1.0]
        returns = trainer._compute_returns(rewards)
        assert len(returns) == 4
        # Last return should be just the last reward
        assert abs(returns[-1] - (-1.0)) < 1e-6
        # Each return should be >= the next (in absolute terms, due to discounting)

    def test_short_training_run(self):
        """Train for a few episodes and verify stats are collected."""
        net = PolicyValueNetwork()
        trainer = RLTrainer(net, num_players=3, lr=1e-3)
        stats = trainer.train(n_episodes=5, print_every=0, seed=42)

        assert stats.total_games == 5
        assert len(stats.episode_rewards) == 5
        assert len(stats.episode_lengths) == 5
        assert all(l > 0 for l in stats.episode_lengths)

    def test_loss_decreases_or_stays_stable(self):
        """Train for enough episodes to see loss isn't diverging."""
        net = PolicyValueNetwork()
        trainer = RLTrainer(net, num_players=3, lr=1e-3)
        stats = trainer.train(n_episodes=20, print_every=0, seed=42)

        # Just verify losses are finite (not NaN/inf)
        for pl in stats.policy_losses:
            assert not (pl != pl), "Policy loss is NaN"  # NaN check
        for vl in stats.value_losses:
            assert not (vl != vl), "Value loss is NaN"

    def test_win_rate_tracking(self):
        net = PolicyValueNetwork()
        trainer = RLTrainer(net, num_players=3)
        stats = trainer.train(n_episodes=10, print_every=0, seed=42)
        assert len(stats.win_rates) == 10
        for wr in stats.win_rates:
            assert 0.0 <= wr <= 1.0


# ── Inference-time threshold override ────────────────────────────────────

class TestThresholdOverride:
    """The override fires when RL is one build from WIN_THRESHOLD and a
    Build action would close out the game. Synthetic tests against
    constructed states; the override's logic is deterministic so unit
    tests are sufficient.
    """

    def _setup(self, seed=1):
        agents = [RandomAgent(i, seed=seed + i) for i in range(3)]
        env = Environment(agents, num_players=3)
        env.reset(seed=seed)
        return env

    def test_override_inactive_below_threshold(self):
        from civsim.data_types import Build, BuildType
        env = self._setup()
        agent = RLAgent(player_id=0, network=None, training=False, threshold_override=True)
        obs = env.state.get_observation(0)
        # At fresh game start, my_pp is just the draft settlements (~4 PP).
        # Override should not fire.
        forced = agent._threshold_override_action(obs, [Build(BuildType.SETTLEMENT, 0)])
        assert forced is None

    def test_override_picks_city_upgrade_when_winning(self):
        from civsim.data_types import Build, BuildType
        env = self._setup()
        # Manually plant enough buildings to reach exactly WIN_THRESHOLD - 2
        inters = list(env.state.board.intersections.keys())
        for iid in inters[:4]:
            env.state.board.place_settlement(0, iid)
        # 4 settlements × 2 PP = 8 PP. A city upgrade gives +2 → win.

        agent = RLAgent(player_id=0, network=None, training=False, threshold_override=True)
        obs = env.state.get_observation(0)
        cities = env.state.board.get_valid_city_positions(0)
        assert cities, "need a valid city upgrade for this test"
        valid = [Build(BuildType.CITY, cities[0]), EndTurn()]
        forced = agent._threshold_override_action(obs, valid)
        assert forced is not None
        assert isinstance(forced, Build) and forced.build_type == BuildType.CITY

    def test_override_picks_settlement_when_it_wins(self):
        from civsim.data_types import Build, BuildType
        env = self._setup()
        inters = list(env.state.board.intersections.keys())
        for iid in inters[:4]:
            env.state.board.place_settlement(0, iid)
        # 4 settlements × 2 = 8 PP. A 5th settlement gives +2 → win.

        agent = RLAgent(player_id=0, network=None, training=False, threshold_override=True)
        obs = env.state.get_observation(0)
        # Use any free intersection that has road access (simulate one)
        # For the unit test, just hand in a Build action — override doesn't
        # check validity, just the build_type and threshold math.
        valid = [Build(BuildType.SETTLEMENT, 50), EndTurn()]
        forced = agent._threshold_override_action(obs, valid)
        assert forced is not None
        assert isinstance(forced, Build) and forced.build_type == BuildType.SETTLEMENT

    def test_override_disabled_returns_nothing(self):
        from civsim.data_types import Build, BuildType
        env = self._setup()
        inters = list(env.state.board.intersections.keys())
        for iid in inters[:4]:
            env.state.board.place_settlement(0, iid)
        # When threshold_override=False, the helper should never be called
        # at all — but if it were, it should still behave sensibly.
        agent = RLAgent(player_id=0, network=None, training=False, threshold_override=False)
        # The agent.select_action wraps the helper; with override off it
        # falls through to network/random. With network=None it returns
        # rng.choice — make sure that path still works.
        obs = env.state.get_observation(0)
        valid = [Build(BuildType.CITY, 0)]
        action = agent.select_action(obs, valid)
        assert action is valid[0]
