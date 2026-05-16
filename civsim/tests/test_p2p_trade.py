"""Tests for player-to-player trade proposals, acceptance, and execution."""
from __future__ import annotations

from civsim.action_executors import execute_action
from civsim.actions import _get_p2p_trade_actions, get_valid_actions
from civsim.agents import (
    GreedyAgent, RandomAgent, _evaluate_trade_for_responder,
)
from civsim.data_types import (
    Observation, OpponentView, ResourceType, TradeProposal,
)
from civsim.environment import Environment
from civsim.game_state import GameState


def _stuff(state: GameState, pid: int, **resources: int) -> None:
    for r_name, amt in resources.items():
        state.players[pid].resources[ResourceType[r_name.upper()]] = amt


def _obs_with_resources(pid: int, **resources: int) -> Observation:
    return Observation(
        player_id=pid,
        my_resources={ResourceType[r.upper()]: amt for r, amt in resources.items()},
        my_dev_cards=[],
        opponents=[],
        board=None,
        bank_supply={},
        current_dice=None,
        turn_number=0,
    )


# ── Proposal Generation ─────────────────────────────────────────────────

def test_no_p2p_proposals_when_no_resources():
    state = GameState.new_game(num_players=3, seed=1)
    # Nobody has any resources → no p2p proposals (no opponents have anything)
    proposals = _get_p2p_trade_actions(state, 0)
    assert proposals == []


def test_p2p_proposals_target_each_active_opponent():
    state = GameState.new_game(num_players=3, seed=1)
    _stuff(state, 0, wood=2)
    # Give opponents some resources so they show up as valid targets
    _stuff(state, 1, water=1)
    _stuff(state, 2, water=1)

    proposals = _get_p2p_trade_actions(state, 0)
    targets = {p.target_player for p in proposals}
    assert targets == {1, 2}
    # All must be p2p (target set)
    assert all(p.target_player is not None for p in proposals)


def test_p2p_proposal_offers_only_held_resources():
    state = GameState.new_game(num_players=3, seed=2)
    _stuff(state, 0, wood=1)  # only wood
    _stuff(state, 1, water=1)

    proposals = _get_p2p_trade_actions(state, 0)
    for p in proposals:
        for r in p.offering:
            assert r == ResourceType.WOOD, f"offered {r}, only have wood"


def test_p2p_proposals_included_in_valid_actions():
    state = GameState.new_game(num_players=3, seed=3)
    _stuff(state, 0, wood=2)
    _stuff(state, 1, water=2)

    valid = get_valid_actions(state, 0)
    p2p = [a for a in valid if isinstance(a, TradeProposal)
           and a.target_player is not None]
    assert p2p, "expected at least one p2p trade proposal"


# ── Responder Heuristic ──────────────────────────────────────────────────

def test_responder_rejects_unaffordable_trade():
    obs = _obs_with_resources(1, wood=1)
    proposal = TradeProposal(
        offering={ResourceType.STONE: 1},
        requesting={ResourceType.WATER: 1},  # responder has no water
        target_player=1,
    )
    assert _evaluate_trade_for_responder(proposal, obs) is False


def test_responder_protects_upkeep_reserves():
    """Even if weighted value favors the responder, refuse to drop water to 0."""
    obs = _obs_with_resources(1, water=1, wood=0)
    proposal = TradeProposal(
        offering={ResourceType.STONE: 5},  # very generous
        requesting={ResourceType.WATER: 1},
        target_player=1,
    )
    assert _evaluate_trade_for_responder(proposal, obs) is False


def test_responder_accepts_weighted_gain():
    """1 water for 1 wood — responder gains net weighted value."""
    obs = _obs_with_resources(1, wood=2)
    proposal = TradeProposal(
        offering={ResourceType.WATER: 1},
        requesting={ResourceType.WOOD: 1},
        target_player=1,
    )
    assert _evaluate_trade_for_responder(proposal, obs) is True


def test_responder_accepts_surplus_for_gap():
    """Equal-value trade where responder has surplus and the offered resource fills a gap."""
    obs = _obs_with_resources(1, stone=5, wood=0)
    proposal = TradeProposal(
        offering={ResourceType.WOOD: 1},
        requesting={ResourceType.STONE: 1},
        target_player=1,
    )
    assert _evaluate_trade_for_responder(proposal, obs) is True


# ── Execution & Stats ────────────────────────────────────────────────────

def test_executing_accepted_trade_transfers_resources_and_bumps_stats():
    state = GameState.new_game(num_players=3, seed=4)
    _stuff(state, 0, water=2)
    _stuff(state, 1, wood=3)

    proposal = TradeProposal(
        offering={ResourceType.WATER: 1},
        requesting={ResourceType.WOOD: 1},
        target_player=1,
    )
    agents = {0: RandomAgent(0, seed=0), 1: GreedyAgent(1, seed=1)}
    result = execute_action(state, 0, proposal, agents)

    assert result.success, f"expected accept, got: {result.description}"
    assert state.players[0].resources[ResourceType.WATER] == 1
    assert state.players[0].resources[ResourceType.WOOD] == 1
    assert state.players[1].resources[ResourceType.WATER] == 1
    assert state.players[1].resources[ResourceType.WOOD] == 2

    # Both sides should have player_trades += 1
    assert state.players[0].stats.player_trades == 1
    assert state.players[1].stats.player_trades == 1
    assert state.players[0].stats.bank_trades == 0
    # Resource flows tracked on both sides
    assert state.players[0].stats.total_resources_spent >= 1
    assert state.players[0].stats.total_resources_earned >= 1
    assert state.players[1].stats.total_resources_spent >= 1
    assert state.players[1].stats.total_resources_earned >= 1


def test_executing_rejected_trade_leaves_state_unchanged():
    state = GameState.new_game(num_players=3, seed=5)
    _stuff(state, 0, wood=1)
    _stuff(state, 1, water=1)  # only 1 water — responder won't part with it

    proposal = TradeProposal(
        offering={ResourceType.WOOD: 1},
        requesting={ResourceType.WATER: 1},
        target_player=1,
    )
    agents = {0: GreedyAgent(0), 1: GreedyAgent(1)}
    before_r0 = dict(state.players[0].resources)
    before_r1 = dict(state.players[1].resources)
    result = execute_action(state, 0, proposal, agents)

    assert result.success is False
    assert state.players[0].resources == before_r0
    assert state.players[1].resources == before_r1
    assert state.players[0].stats.player_trades == 0
    assert state.players[1].stats.player_trades == 0


def test_p2p_trade_rejected_in_simulation_without_agents():
    """MCTS rollouts call execute_action without agents — p2p trades must
    not silently fall through to the bank trade path."""
    state = GameState.new_game(num_players=3, seed=6)
    _stuff(state, 0, wood=1)

    proposal = TradeProposal(
        offering={ResourceType.WOOD: 1},
        requesting={ResourceType.WATER: 1},
        target_player=1,
    )
    result = execute_action(state, 0, proposal, agents=None)
    assert result.success is False
    # Resources untouched — no bank-trade fall-through
    assert state.players[0].resources[ResourceType.WOOD] == 1


def test_full_game_completes_with_p2p_trade_enabled():
    """End-to-end smoke test: a game with p2p trades terminates cleanly."""
    agents = [GreedyAgent(player_id=i, seed=200 + i) for i in range(3)]
    env = Environment(agents, num_players=3)
    result = env.run_game()
    assert result is not None
    # Stats schema check: p2p counter exists and is non-negative
    for stats in result.player_stats.values():
        assert stats.player_trades >= 0
        assert stats.bank_trades >= 0
