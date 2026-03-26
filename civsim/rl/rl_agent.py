"""RL Agent that uses a trained PolicyValueNetwork to select actions.

During gameplay:
    1. Encode observation → state feature vector
    2. Encode each valid action → action feature vectors
    3. Forward pass through network → log-probabilities + state value
    4. Sample action from probability distribution

During training (see trainer.py):
    The agent stores (log_prob, value, reward) tuples for each step,
    which the trainer uses to compute policy gradient updates.
"""
from __future__ import annotations

import random
from typing import Optional

from ..agents import Agent
from ..data_types import (
    Action, Build, BuildType, BuyDevCard, DevCardType, DivineIntervention,
    EdgeID, EndTurn, IntersectionID, Observation, PlayDevCard,
    ResourceType, TradeProposal,
)
from .features import encode_action, encode_observation

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class RLAgent(Agent):
    """Reinforcement Learning agent powered by a neural network.

    Can operate in two modes:
        - training=True:  samples from the policy distribution, stores
                          trajectories for gradient updates
        - training=False: picks the highest-probability action (greedy)

    Args:
        player_id: which player this agent controls
        network:   a PolicyValueNetwork instance (shared or per-agent)
        seed:      RNG seed for reproducibility
        training:  whether to sample stochastically and record trajectories
        temperature: softmax temperature (>1 = more exploration, <1 = more greedy)
    """

    def __init__(
        self,
        player_id: int,
        network=None,
        seed: Optional[int] = None,
        training: bool = False,
        temperature: float = 1.0,
    ):
        super().__init__(player_id, seed)
        self.network = network
        self.training = training
        self.temperature = temperature

        # Trajectory storage for training
        self.saved_log_probs: list = []
        self.saved_values: list = []
        self.rewards: list[float] = []

    def select_action(self, obs: Observation, valid_actions: list[Action]) -> Action:
        if not HAS_TORCH or self.network is None:
            return self.rng.choice(valid_actions)

        if len(valid_actions) == 1:
            if self.training:
                # Store dummy values so trajectory stays aligned with rewards
                self.saved_log_probs.append(torch.tensor(0.0, requires_grad=True))
                self.saved_values.append(torch.tensor([0.0], requires_grad=True))
            return valid_actions[0]

        # Encode state
        state_vec = encode_observation(obs)
        state_tensor = torch.tensor(state_vec, dtype=torch.float32)

        # Encode each valid action
        action_vecs = [encode_action(a) for a in valid_actions]
        action_tensor = torch.tensor(action_vecs, dtype=torch.float32)

        # All actions are valid — mask is all 1s
        mask = torch.ones(len(valid_actions), dtype=torch.float32)

        # Forward pass
        with torch.no_grad() if not self.training else _nullcontext():
            log_probs, value = self.network(state_tensor, action_tensor, mask)

        # Apply temperature
        if self.temperature != 1.0:
            log_probs = log_probs / self.temperature

        if self.training:
            # Sample from distribution
            probs = torch.exp(log_probs)
            # Clamp to avoid numerical issues
            probs = torch.clamp(probs, min=1e-8)
            probs = probs / probs.sum()
            dist = torch.distributions.Categorical(probs)
            idx = dist.sample()

            # Store for training
            self.saved_log_probs.append(dist.log_prob(idx))
            self.saved_values.append(value)
        else:
            # Greedy: pick highest probability
            idx = torch.argmax(log_probs)

        return valid_actions[idx.item()]

    def record_reward(self, reward: float) -> None:
        """Called after each action to record the reward for training."""
        self.rewards.append(reward)

    def clear_trajectory(self) -> None:
        """Clear stored trajectory data after a training update."""
        self.saved_log_probs.clear()
        self.saved_values.clear()
        self.rewards.clear()

    def get_trajectory(self) -> tuple[list, list, list[float]]:
        """Return (log_probs, values, rewards) for the current episode."""
        return self.saved_log_probs, self.saved_values, self.rewards

    # ── Draft & Utility Methods ──────────────────────────────────────

    def select_draft_action(self, obs: Observation, valid_positions: list[IntersectionID]) -> IntersectionID:
        # Use pip-counting heuristic (same as GreedyAgent/MCTSAgent)
        best_score = -1.0
        best_pos = valid_positions[0]
        board = obs.board
        for iid in valid_positions:
            score = 0.0
            tiles = board.get_tiles_for_intersection(iid)
            for tile in tiles:
                if tile.dice_number is not None:
                    pips = 6 - abs(7 - tile.dice_number)
                    score += pips
            if score > best_score:
                best_score = score
                best_pos = iid
        return best_pos

    def select_draft_road(self, obs: Observation, valid_edges: list[EdgeID]) -> EdgeID:
        return self.rng.choice(valid_edges)

    def respond_to_trade(self, proposal: TradeProposal, obs: Observation) -> bool:
        give_total = sum(proposal.requesting.values())
        get_total = sum(proposal.offering.values())
        return get_total >= give_total

    def select_target(self, opponents: list[int]) -> int:
        return opponents[0]

    def select_resource_type(self) -> ResourceType:
        return ResourceType.WHEAT

    def select_two_resources(self) -> tuple[ResourceType, ResourceType]:
        return ResourceType.WHEAT, ResourceType.METAL


class _nullcontext:
    """Minimal no-op context manager (avoid importing contextlib)."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
