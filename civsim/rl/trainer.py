"""REINFORCE trainer with baseline (Actor-Critic).

Training loop:
    1. Play a full game with the RL agent(s)
    2. Compute discounted returns for each step
    3. Compute advantages: return - value_prediction
    4. Policy loss:  -log_prob * advantage  (push good actions up)
    5. Value loss:   (return - value)²      (better predictions)
    6. Entropy bonus: encourage exploration early on
    7. Backprop and update weights

Usage:
    from civsim.rl import RLTrainer, PolicyValueNetwork, RLAgent

    network = PolicyValueNetwork()
    trainer = RLTrainer(network, num_players=3)
    trainer.train(n_episodes=1000)
    network.save("model.pt")
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ..agents import RandomAgent
from ..data_types import EndTurn, GameResult
from ..environment import Environment
from ..actions import get_valid_actions
from ..action_executors import execute_action

try:
    import torch
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class TrainingStats:
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    policy_losses: list[float] = field(default_factory=list)
    value_losses: list[float] = field(default_factory=list)
    win_rates: list[float] = field(default_factory=list)  # rolling window
    wins: int = 0
    total_games: int = 0


class RLTrainer:
    """Trains an RL agent via REINFORCE with baseline.

    Args:
        network:        PolicyValueNetwork to train
        num_players:    number of players per game
        lr:             learning rate
        gamma:          discount factor for returns
        entropy_coeff:  weight for entropy bonus (encourages exploration)
        value_coeff:    weight for value loss
        max_actions_per_turn: safety cap on actions per turn during training
        opponent_type:  agent class for opponents (default: RandomAgent)
    """

    def __init__(
        self,
        network,
        num_players: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        max_actions_per_turn: int = 50,
        opponent_type=None,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for training. Install with: pip install torch")

        self.network = network
        self.num_players = num_players
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_actions_per_turn = max_actions_per_turn
        self.opponent_type = opponent_type or RandomAgent

        self.optimizer = optim.Adam(network.parameters(), lr=lr)
        self.stats = TrainingStats()

    def train(
        self,
        n_episodes: int = 1000,
        print_every: int = 50,
        seed: Optional[int] = None,
    ) -> TrainingStats:
        """Train for n_episodes games."""
        from .rl_agent import RLAgent

        start_time = time.time()
        recent_wins: list[bool] = []

        for episode in range(n_episodes):
            ep_seed = (seed + episode * 1000) if seed is not None else None

            # Create agents: player 0 = RL, rest = opponents
            rl_agent = RLAgent(
                player_id=0,
                network=self.network,
                seed=ep_seed,
                training=True,
                temperature=max(0.5, 1.0 - episode / (n_episodes * 0.8)),
            )
            opponents = [
                self.opponent_type(player_id=i, seed=ep_seed + i if ep_seed else None)
                for i in range(1, self.num_players)
            ]
            all_agents = [rl_agent] + opponents

            # Play a game, collecting trajectory for RL agent
            result = self._play_training_game(all_agents, ep_seed)

            # Compute losses and update
            loss_info = self._update(rl_agent)

            # Track stats
            won = result.winner == 0
            recent_wins.append(won)
            if len(recent_wins) > 100:
                recent_wins.pop(0)

            self.stats.total_games += 1
            if won:
                self.stats.wins += 1
            self.stats.episode_lengths.append(result.turns_played)
            self.stats.episode_rewards.append(sum(rl_agent.rewards))
            if loss_info:
                self.stats.policy_losses.append(loss_info["policy_loss"])
                self.stats.value_losses.append(loss_info["value_loss"])
            self.stats.win_rates.append(sum(recent_wins) / len(recent_wins))

            if print_every and (episode + 1) % print_every == 0:
                elapsed = time.time() - start_time
                wr = self.stats.win_rates[-1]
                avg_len = sum(self.stats.episode_lengths[-print_every:]) / print_every
                avg_reward = sum(self.stats.episode_rewards[-print_every:]) / print_every
                print(
                    f"Episode {episode+1:5d} | "
                    f"Win rate: {wr:.1%} | "
                    f"Avg reward: {avg_reward:+.1f} | "
                    f"Avg length: {avg_len:.0f} | "
                    f"Time: {elapsed:.1f}s"
                )

        return self.stats

    def _play_training_game(
        self,
        agents: list,
        seed: Optional[int],
    ) -> GameResult:
        """Play one game, recording rewards for the RL agent (player 0)."""
        env = Environment(agents, num_players=self.num_players)
        env.reset(seed=seed)
        rl_agent = agents[0]  # RLAgent at index 0

        while not env.state.game_over:
            pid = env.state.current_player

            if pid == rl_agent.player_id:
                # RL agent's turn — step through actions individually
                self._run_rl_turn(env, rl_agent)
            else:
                # Opponent's turn — run normally
                agent = env.agents[pid]
                env._run_action_loop(pid, agent)

            if env.state.game_over:
                break

        # Final reward: add win/loss bonus to the last recorded reward
        result = env.state.get_final_results()
        bonus = 10.0 if result.winner == rl_agent.player_id else -5.0
        if rl_agent.rewards:
            rl_agent.rewards[-1] += bonus
        else:
            # Edge case: no actions were taken
            rl_agent.record_reward(bonus)

        return result

    def _run_rl_turn(self, env: Environment, rl_agent) -> None:
        """Run one turn for the RL agent, recording each action's reward.

        Invariant: each call to select_action (which stores a log_prob + value)
        is matched by exactly one call to record_reward, keeping all three
        lists (log_probs, values, rewards) aligned at the same length.
        """
        pid = rl_agent.player_id
        old_points = env.state.players[pid].progress_points

        for _ in range(self.max_actions_per_turn):
            obs = env.state.get_observation(pid)
            valid = get_valid_actions(env.state, pid)
            action = rl_agent.select_action(obs, valid)
            # select_action stored one log_prob + one value → record one reward

            if isinstance(action, EndTurn):
                new_points = env.state.players[pid].progress_points
                rl_agent.record_reward(float(new_points - old_points))
                env._end_turn()
                return

            execute_action(env.state, pid, action, env.agents)

            new_points = env.state.players[pid].progress_points
            step_reward = float(new_points - old_points)
            rl_agent.record_reward(step_reward)
            old_points = new_points

            if env.state.game_over:
                return

        # Forced end turn — no select_action call, so no reward to record
        env._end_turn()

    def _update(self, rl_agent) -> Optional[dict]:
        """Compute REINFORCE loss and update network weights."""
        log_probs, values, rewards = rl_agent.get_trajectory()

        if not log_probs:
            rl_agent.clear_trajectory()
            return None

        # Compute discounted returns
        returns = self._compute_returns(rewards)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns for stability
        if len(returns) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

        # Stack tensors
        log_prob_tensor = torch.stack(log_probs)
        value_tensor = torch.stack(values).squeeze()

        # Advantages: how much better was actual return vs predicted value
        advantages = returns_tensor - value_tensor.detach()

        # Policy loss: -log_prob * advantage
        policy_loss = -(log_prob_tensor * advantages).mean()

        # Value loss: MSE between predicted value and actual return
        value_loss = torch.nn.functional.mse_loss(value_tensor, returns_tensor)

        # Entropy bonus: encourage exploration
        entropy = -(torch.exp(log_prob_tensor) * log_prob_tensor).mean()

        # Total loss
        total_loss = (
            policy_loss
            + self.value_coeff * value_loss
            - self.entropy_coeff * entropy
        )

        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()

        rl_agent.clear_trajectory()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
        }

    def _compute_returns(self, rewards: list[float]) -> list[float]:
        """Compute discounted returns: R_t = r_t + gamma * R_{t+1}."""
        returns = []
        running = 0.0
        for r in reversed(rewards):
            running = r + self.gamma * running
            returns.insert(0, running)
        return returns
