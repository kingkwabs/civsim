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
        threshold_override: bool = True,
    ):
        super().__init__(player_id, seed)
        self.network = network
        self.training = training
        self.temperature = temperature
        # Inference-time threshold override: when close to WIN_THRESHOLD,
        # bypass the policy and force a Build that crosses the threshold.
        # Addresses the empirically-confirmed "stall at 8 PP" pattern
        # where the trained policy plateaus near a winning position but
        # rarely takes the closing build move. Disabled during training
        # so we don't bias the gradient signal.
        self.threshold_override = threshold_override

        # Trajectory storage for training
        self.saved_log_probs: list = []
        self.saved_values: list = []
        self.rewards: list[float] = []

    def select_action(self, obs: Observation, valid_actions: list[Action]) -> Action:
        # Threshold override fires before the network — at high PP we want
        # the closing move, not whatever the trained policy thinks is best
        # (which is usually a cautious trade or EndTurn).
        if self.threshold_override and not self.training:
            forced = self._threshold_override_action(obs, valid_actions)
            if forced is not None:
                if HAS_TORCH and self.training:
                    self.saved_log_probs.append(torch.tensor(0.0, requires_grad=True))
                    self.saved_values.append(torch.tensor([0.0], requires_grad=True))
                return forced

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

    def _threshold_override_action(
        self, obs: Observation, valid_actions: list[Action]
    ) -> Optional[Action]:
        """Return a Build action that crosses (or sets up crossing of) the
        win threshold, or None if no such play is available.

        Priorities (each only fires if it would put us at WIN_THRESHOLD):
          1. City upgrade (CITY_PP - SETTLEMENT_PP = +2 PP, single action)
          2. New settlement build (+SETTLEMENT_PP = +2 PP, single action)
          3. Road build, *only if* we also have resources for a settlement
             this turn (the agent will pick the settlement on its next
             call, since the new road opens fresh valid settlement spots).
        """
        from ..data_types import (
            BUILDING_COSTS, Build, BuildType, CITY_PP,
            ReestablishBuilding, ResourceType, SETTLEMENT_PP, WIN_THRESHOLD,
        )

        # Current PP from board state (don't trust stale cached value)
        my_pp = 0
        for inter in obs.board.intersections.values():
            if inter.owner != self.player_id or inter.building is None:
                continue
            if inter.building == BuildType.SETTLEMENT:
                my_pp += SETTLEMENT_PP
            elif inter.building == BuildType.CITY:
                my_pp += CITY_PP

        # Only fire when within one build of the win threshold; earlier
        # plays should follow the trained policy's lead.
        if my_pp < WIN_THRESHOLD - SETTLEMENT_PP:
            return None

        city_action = None
        settlement_action = None
        road_action = None
        reestablish_action = None
        for a in valid_actions:
            if isinstance(a, ReestablishBuilding):
                # Re-establish grants the building's full PP (no prior owner)
                gain = SETTLEMENT_PP if a.build_type == BuildType.SETTLEMENT else CITY_PP
                if my_pp + gain >= WIN_THRESHOLD and reestablish_action is None:
                    reestablish_action = a
                continue
            if not isinstance(a, Build):
                continue
            if a.build_type == BuildType.CITY:
                if my_pp + (CITY_PP - SETTLEMENT_PP) >= WIN_THRESHOLD:
                    city_action = a
            elif a.build_type == BuildType.SETTLEMENT:
                if my_pp + SETTLEMENT_PP >= WIN_THRESHOLD:
                    settlement_action = a
            elif a.build_type == BuildType.ROAD and road_action is None:
                road_action = a

        # Prefer re-establish (often grabs a city = +4 PP instantly)
        if reestablish_action is not None:
            return reestablish_action
        if city_action is not None:
            return city_action
        if settlement_action is not None:
            return settlement_action

        # Road-then-settlement: only worth doing if we have settlement
        # resources after paying for the road. Otherwise we'd waste a turn.
        if road_action is not None:
            my_res = obs.my_resources
            road_cost = BUILDING_COSTS[BuildType.ROAD]
            sett_cost = BUILDING_COSTS[BuildType.SETTLEMENT]
            after_road_ok = all(
                my_res.get(r, 0) - road_cost.get(r, 0) >= sett_cost.get(r, 0)
                for r in ResourceType
            )
            if after_road_ok:
                return road_action

        return None

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
        from ..agents import _score_draft_position
        return _score_draft_position(obs, valid_positions)

    def select_draft_road(self, obs: Observation, valid_edges: list[EdgeID]) -> EdgeID:
        return self.rng.choice(valid_edges)

    def respond_to_trade(self, proposal: TradeProposal, obs: Observation) -> bool:
        # Use the shared resource-weighted heuristic. The policy network is
        # trained to *propose* trades, not to decide on incoming ones —
        # delegating to the heuristic gives a sensible deterministic counterparty.
        from ..agents import _evaluate_trade_for_responder
        return _evaluate_trade_for_responder(proposal, obs)

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
