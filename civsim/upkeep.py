"""Upkeep computation helpers.

Pure read-only utilities that answer:
  - what is this player's total per-turn upkeep?
  - which resources are they short on?
  - how many buildings would they lose on EndTurn right now?

Used by agent heuristics, MCTS rollout biasing, and the RL state encoder
so every layer of the stack shares the same upkeep signal.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .data_types import (
    BuildType, MAINTENANCE_COSTS, Observation, Player, ResourceDict,
    ResourceType,
)

if TYPE_CHECKING:
    from .board import Board


def building_counts(board, player_id: int) -> dict[BuildType, int]:
    counts: dict[BuildType, int] = {bt: 0 for bt in BuildType}
    for inter in board.intersections.values():
        if inter.owner == player_id and inter.building is not None:
            counts[inter.building] = counts.get(inter.building, 0) + 1
    return counts


def total_upkeep_cost(board, player_id: int) -> ResourceDict:
    """Total resources the player owes for maintenance this turn."""
    cost: ResourceDict = {}
    for inter in board.intersections.values():
        if inter.owner != player_id or inter.building is None:
            continue
        per = MAINTENANCE_COSTS.get(inter.building, {})
        for r, amt in per.items():
            cost[r] = cost.get(r, 0) + amt
    return cost


def upkeep_gap(resources: ResourceDict, board, player_id: int) -> ResourceDict:
    """Per-resource shortage (0 if fully covered) for this player's upkeep.

    `resources` is the player's current holdings (so callers can probe
    hypothetical balances).
    """
    cost = total_upkeep_cost(board, player_id)
    gap: ResourceDict = {}
    for r, amt in cost.items():
        held = resources.get(r, 0)
        if held < amt:
            gap[r] = amt - held
    return gap


def buildings_at_risk(player: Player, board) -> int:
    """How many buildings would be lost if EndTurn were called right now.

    Greedy resolution: pays maintenance per building in iteration order until
    resources run out (same behavior as `GameState.resolve_maintenance`).
    """
    if player.maintenance_paid_this_turn:
        return 0
    pool: ResourceDict = dict(player.resources)
    lost = 0
    for inter in board.intersections.values():
        if inter.owner != player.player_id or inter.building is None:
            continue
        cost = MAINTENANCE_COSTS.get(inter.building)
        if cost is None:
            continue
        if all(pool.get(r, 0) >= amt for r, amt in cost.items()):
            for r, amt in cost.items():
                pool[r] -= amt
        else:
            lost += 1
    return lost


def buildings_at_risk_from_obs(obs: Observation) -> int:
    """Same as buildings_at_risk but from a partial Observation.

    Reconstructs a minimal Player-like view from the obs's own_resources
    and counts buildings on the board owned by us.
    """
    from .data_types import Player as _Player
    p = _Player(player_id=obs.player_id, resources=dict(obs.my_resources))
    return buildings_at_risk(p, obs.board)


def upkeep_gap_from_obs(obs: Observation) -> ResourceDict:
    return upkeep_gap(obs.my_resources, obs.board, obs.player_id)


def upkeep_pressure_from_obs(obs: Observation, buffer_turns: int = 1) -> ResourceDict:
    """Like upkeep_gap, but treats the next `buffer_turns` turns of upkeep
    as if it were due now. Used to stockpile water/cow *before* a crisis.

    Returns the shortfall against a target reserve of (1 + buffer_turns) ×
    per-turn upkeep cost. Production between turns can fill some of this,
    but agents using the bank's 4:1 ratio can't react fast enough — they
    have to acquire upkeep resources before they're empty.
    """
    cost = total_upkeep_cost(obs.board, obs.player_id)
    target = {r: amt * (1 + buffer_turns) for r, amt in cost.items()}
    gap: ResourceDict = {}
    for r, amt in target.items():
        held = obs.my_resources.get(r, 0)
        if held < amt:
            gap[r] = amt - held
    return gap
