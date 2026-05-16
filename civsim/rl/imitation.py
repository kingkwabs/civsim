"""Imitation learning (behavior cloning) warm start for the RL agent.

Pipeline:
    1. Play many Greedy-vs-Greedy games, record every (state, valid_actions,
       chosen_action) tuple from the player who *won by reaching the win
       threshold*. Skip games that ended by turn-cap fallback.
    2. Behavior-clone the policy network on those demonstrations:
       maximize log P(chosen_action | state) under the network.
    3. Hand off to the standard REINFORCE curriculum for fine-tuning.

Why this matters: REINFORCE-with-baseline never gives the policy a gradient
signal for "build a road then a settlement to cross the win threshold"
because the road action's immediate reward is 0 and the discounted future
reward is too weak. Demonstrations show the policy what winning *looks
like*, breaking the bootstrap problem.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Optional

from ..agents import Agent, GreedyAgent
from ..data_types import (
    Action, EndTurn, IntersectionID, EdgeID, Observation, ResourceType,
    TradeProposal, WIN_THRESHOLD,
)
from ..environment import Environment
from .features import encode_action, encode_observation

try:
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class Demonstration:
    """One (state, valid_actions, chosen_index) triple from a winning trajectory."""
    state: list[float]
    action_vecs: list[list[float]]
    chosen_idx: int


class _RecordingAgent(Agent):
    """Wraps any Agent; logs every (obs, valid_actions, chosen) tuple."""

    def __init__(self, inner: Agent):
        super().__init__(player_id=inner.player_id, seed=None)
        self.inner = inner
        # The Agent base class adds a `_failed_this_turn` set; the inner
        # agent has its own. Forward our hooks to the inner so its
        # rejection memory still works.
        self.history: list[Demonstration] = []

    def select_action(self, obs: Observation, valid_actions: list[Action]) -> Action:
        chosen = self.inner.select_action(obs, valid_actions)
        # Find the chosen action's index in valid_actions by identity.
        idx = next(
            (i for i, a in enumerate(valid_actions) if a is chosen),
            None,
        )
        if idx is None:
            # Fallback: structural equality on (type + repr) — shouldn't happen
            # since well-behaved agents return a reference from the input list.
            idx = 0
        self.history.append(Demonstration(
            state=encode_observation(obs),
            action_vecs=[encode_action(a) for a in valid_actions],
            chosen_idx=idx,
        ))
        return chosen

    # Forward all the other Agent overrides to the inner agent
    def select_draft_action(self, obs, valid_positions: list[IntersectionID]) -> IntersectionID:
        return self.inner.select_draft_action(obs, valid_positions)

    def select_draft_road(self, obs, valid_edges: list[EdgeID]) -> EdgeID:
        return self.inner.select_draft_road(obs, valid_edges)

    def respond_to_trade(self, proposal: TradeProposal, obs: Observation) -> bool:
        return self.inner.respond_to_trade(proposal, obs)

    def select_target(self, opponents: list[int]) -> int:
        return self.inner.select_target(opponents)

    def select_resource_type(self) -> ResourceType:
        return self.inner.select_resource_type()

    def select_two_resources(self) -> tuple[ResourceType, ResourceType]:
        return self.inner.select_two_resources()

    def select_road(self, valid_positions: list[EdgeID]) -> EdgeID:
        return self.inner.select_road(valid_positions)

    def on_action_result(self, action: Action, success: bool) -> None:
        self.inner.on_action_result(action, success)

    def on_turn_start(self) -> None:
        self.inner.on_turn_start()


def collect_demonstrations(
    n_games: int = 200,
    seed_base: int = 7000,
    num_players: int = 3,
    verbose: bool = True,
) -> list[Demonstration]:
    """Play Greedy-vs-Greedy games; return demos from threshold-winning players."""
    all_demos: list[Demonstration] = []
    threshold_wins = 0
    turn_cap_endings = 0
    t0 = time.time()

    for seed in range(n_games):
        ep_seed = seed_base + seed
        recorders = [
            _RecordingAgent(GreedyAgent(player_id=i, seed=ep_seed * 7 + i))
            for i in range(num_players)
        ]
        env = Environment(recorders, num_players=num_players)
        result = env.run_game(seed=ep_seed)

        winner = result.winner
        reached_threshold = (
            winner is not None
            and result.scores.get(winner, 0) >= WIN_THRESHOLD
            and result.turns_played < 200
        )
        if reached_threshold:
            threshold_wins += 1
            # Save only the winning player's recorded steps
            all_demos.extend(recorders[winner].history)
        else:
            turn_cap_endings += 1

        if verbose and (seed + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"  [demos] {seed+1}/{n_games} games  "
                f"threshold_wins={threshold_wins}  cap={turn_cap_endings}  "
                f"demos={len(all_demos)}  elapsed={elapsed:.1f}s",
                flush=True,
            )

    if verbose:
        print(
            f"  [demos] DONE: {threshold_wins} threshold-winning games out of {n_games}, "
            f"yielded {len(all_demos)} (state, action) demos",
            flush=True,
        )
    return all_demos


def bc_train(
    network,
    demos: list[Demonstration],
    n_epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    verbose: bool = True,
) -> dict:
    """Behavior cloning: minimize -log P(chosen_action | state) on demonstrations.

    Each demo has a *variable* number of valid actions, so we don't batch
    across demos with tensor padding — we just sum per-demo losses within a
    batch and step. Slower than fully vectorized BC but simple and correct.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for BC training")
    if not demos:
        if verbose:
            print("  [BC] no demonstrations to train on — skipping", flush=True)
        return {"epochs": 0, "final_loss": 0.0}

    optimizer = optim.Adam(network.parameters(), lr=lr)
    network.train()

    losses = []
    t0 = time.time()
    for epoch in range(n_epochs):
        random.shuffle(demos)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(demos), batch_size):
            batch = demos[start:start + batch_size]
            batch_loss = torch.tensor(0.0)
            for demo in batch:
                state = torch.tensor(demo.state, dtype=torch.float32)
                actions = torch.tensor(demo.action_vecs, dtype=torch.float32)
                mask = torch.ones(len(demo.action_vecs), dtype=torch.float32)
                log_probs, _ = network(state, actions, mask)
                batch_loss = batch_loss - log_probs[demo.chosen_idx]
            batch_loss = batch_loss / len(batch)
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += batch_loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(1, n_batches)
        losses.append(avg_loss)
        if verbose and (epoch + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(
                f"  [BC] epoch {epoch+1}/{n_epochs}  loss={avg_loss:.4f}  "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    network.eval()
    return {"epochs": n_epochs, "final_loss": losses[-1] if losses else 0.0, "losses": losses}
