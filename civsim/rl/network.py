"""Neural network for the RL agent (Actor-Critic architecture).

Architecture:
    State features (36) ──→ Shared trunk (128 → 64) ──┬──→ Policy head (64 → action_scores)
                                                       └──→ Value head  (64 → 1 scalar)

The policy head scores each valid action using a learned action-embedding dot product:
    score(action_i) = MLP(state_features)ᵀ · MLP(action_features_i)

This lets the network generalize across actions with similar features
(e.g., all "build settlement" actions share the same category encoding).
"""
from __future__ import annotations

import math
from typing import Optional

from .features import STATE_DIM, ACTION_DIM

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:

    class PolicyValueNetwork(nn.Module):
        """Actor-Critic network with action-feature scoring."""

        def __init__(
            self,
            state_dim: int = STATE_DIM,
            action_dim: int = ACTION_DIM,
            hidden_dim: int = 128,
        ):
            super().__init__()

            # Shared state encoder trunk
            self.state_trunk = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
            )

            # Policy head: projects state into action-scoring space
            self.policy_state_proj = nn.Linear(hidden_dim // 2, 32)

            # Action encoder: projects action features into same space
            self.action_encoder = nn.Sequential(
                nn.Linear(action_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
            )

            # Value head: estimates V(s)
            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                    nn.init.zeros_(m.bias)
            # Smaller init for value and policy outputs (standard RL practice)
            nn.init.orthogonal_(self.value_head[-1].weight, gain=0.01)
            nn.init.orthogonal_(self.policy_state_proj.weight, gain=0.01)

        def forward(
            self,
            state_features: torch.Tensor,
            action_features: torch.Tensor,
            action_mask: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                state_features:  (batch, STATE_DIM) or (STATE_DIM,)
                action_features: (batch, num_actions, ACTION_DIM) or (num_actions, ACTION_DIM)
                action_mask:     (batch, num_actions) or (num_actions,) — 1=valid, 0=invalid

            Returns:
                log_probs: (batch, num_actions) log probabilities over actions
                value:     (batch, 1) state value estimate
            """
            single = state_features.dim() == 1
            if single:
                state_features = state_features.unsqueeze(0)
                action_features = action_features.unsqueeze(0)
                if action_mask is not None:
                    action_mask = action_mask.unsqueeze(0)

            # Shared trunk
            trunk = self.state_trunk(state_features)  # (batch, hidden//2)

            # Policy: score each action via dot product
            state_proj = self.policy_state_proj(trunk)  # (batch, 32)
            action_emb = self.action_encoder(action_features)  # (batch, num_actions, 32)
            # Dot product: (batch, 1, 32) @ (batch, 32, num_actions) → (batch, 1, num_actions)
            logits = torch.bmm(
                state_proj.unsqueeze(1),
                action_emb.transpose(1, 2),
            ).squeeze(1)  # (batch, num_actions)

            # Apply mask: set invalid actions to -inf
            if action_mask is not None:
                logits = logits.masked_fill(action_mask == 0, float("-inf"))

            log_probs = F.log_softmax(logits, dim=-1)

            # Value head
            value = self.value_head(trunk)  # (batch, 1)

            if single:
                log_probs = log_probs.squeeze(0)
                value = value.squeeze(0)

            return log_probs, value

        def save(self, path: str) -> None:
            torch.save(self.state_dict(), path)

        def load(self, path: str) -> None:
            self.load_state_dict(torch.load(path, weights_only=True))

else:

    class PolicyValueNetwork:
        """Stub when PyTorch is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for the RL agent. "
                "Install with: pip install torch"
            )
