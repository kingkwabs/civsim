"""Valid action enumeration for CivSim."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .data_types import (
    Action, Build, BuildType, BUILDING_COSTS, BuyDevCard, DEV_CARD_COST,
    DIVINE_COST, DivineIntervention, EndTurn, PlayDevCard, PortType,
    REESTABLISHMENT_COSTS, ReestablishBuilding, ResourceDict, ResourceType,
    TradeProposal, TRADE_CAP_PER_TURN,
)

if TYPE_CHECKING:
    from .game_state import GameState


def get_valid_actions(state: GameState, player_id: int) -> list[Action]:
    actions: list[Action] = []
    actions += _get_valid_build_actions(state, player_id)
    actions += _get_valid_reestablish_actions(state, player_id)
    # Throttle trades: once a player has burned TRADE_CAP_PER_TURN attempts
    # this turn, stop offering trade actions entirely.
    if state.players[player_id].trades_this_turn < TRADE_CAP_PER_TURN:
        actions += _get_valid_trade_actions(state, player_id)
        actions += _get_p2p_trade_actions(state, player_id)
    if _can_buy_dev_card(state, player_id):
        actions.append(BuyDevCard())
    if _can_play_dev_card(state, player_id):
        actions += _get_playable_dev_cards(state, player_id)
    if _can_divine_intervention(state, player_id):
        actions.append(DivineIntervention())
    actions.append(EndTurn())
    return actions


def _get_valid_reestablish_actions(
    state: GameState, player_id: int
) -> list[ReestablishBuilding]:
    """Return all abandoned buildings the player can afford to reclaim.

    Re-establishment costs more than the original build (see
    REESTABLISHMENT_COSTS) because the player is paying a premium to
    take over someone else's lost work.
    """
    player = state.players[player_id]
    actions: list[ReestablishBuilding] = []
    for inter in state.board.intersections.values():
        if inter.building is None or inter.owner is not None:
            continue
        cost = REESTABLISHMENT_COSTS.get(inter.building)
        if cost is None:
            continue
        if _player_can_afford(player.resources, cost):
            actions.append(ReestablishBuilding(inter.building, inter.inter_id))
    return actions


def _player_can_afford(resources: ResourceDict, cost: ResourceDict) -> bool:
    return all(resources.get(r, 0) >= amt for r, amt in cost.items())


def _get_valid_build_actions(state: GameState, player_id: int) -> list[Build]:
    player = state.players[player_id]
    actions: list[Build] = []

    # Roads
    if _player_can_afford(player.resources, BUILDING_COSTS[BuildType.ROAD]):
        for eid in state.board.get_valid_road_positions(player_id):
            actions.append(Build(BuildType.ROAD, eid))

    # Settlements
    if _player_can_afford(player.resources, BUILDING_COSTS[BuildType.SETTLEMENT]):
        for iid in state.board.get_valid_settlement_positions(player_id):
            actions.append(Build(BuildType.SETTLEMENT, iid))

    # Cities
    if _player_can_afford(player.resources, BUILDING_COSTS[BuildType.CITY]):
        for iid in state.board.get_valid_city_positions(player_id):
            actions.append(Build(BuildType.CITY, iid))

    return actions


def _get_valid_trade_actions(state: GameState, player_id: int) -> list[TradeProposal]:
    """Generate bank/port trade actions."""
    player = state.players[player_id]
    trades: list[TradeProposal] = []
    ports = state.board.get_player_ports(player_id)

    # Port-based trades (2:1 specific)
    for port in ports:
        if port.port_type == PortType.GENERIC:
            # 3:1 generic: offer 3 of any resource for 1 of any other
            for offer_r in ResourceType:
                if player.resources.get(offer_r, 0) >= 3:
                    for want_r in ResourceType:
                        if want_r != offer_r and state.bank.get(want_r, 0) >= 1:
                            trades.append(TradeProposal(
                                offering={offer_r: 3},
                                requesting={want_r: 1},
                                port_type=PortType.GENERIC,
                            ))
        else:
            # 2:1 specific port
            from .data_types import PORT_RESOURCE_MAP
            port_resource = PORT_RESOURCE_MAP.get(port.port_type)
            if port_resource and player.resources.get(port_resource, 0) >= 2:
                for want_r in ResourceType:
                    if want_r != port_resource and state.bank.get(want_r, 0) >= 1:
                        trades.append(TradeProposal(
                            offering={port_resource: 2},
                            requesting={want_r: 1},
                            port_type=port.port_type,
                        ))

    # 4:1 bank trade (always available)
    for offer_r in ResourceType:
        if player.resources.get(offer_r, 0) >= 4:
            for want_r in ResourceType:
                if want_r != offer_r and state.bank.get(want_r, 0) >= 1:
                    trades.append(TradeProposal(
                        offering={offer_r: 4},
                        requesting={want_r: 1},
                    ))

    return trades


def _get_p2p_trade_actions(state: GameState, player_id: int) -> list[TradeProposal]:
    """Generate player-to-player trade proposals.

    Bounded space: for each opponent and each resource the proposer has at
    least one of, propose a 1-for-1 swap for each other resource. Plus a
    handful of 2-for-1 surplus swaps (offering 2 of a stockpiled resource).

    Worst-case per turn ≈ 2 opp × 6 offer × 5 want = 60 actions, usually far
    fewer since players don't hold all resource types. This keeps the action
    space tractable for MCTS/RL while still covering meaningful trades.
    """
    player = state.players[player_id]
    proposals: list[TradeProposal] = []

    opponents = [
        pid for pid, p in state.players.items()
        if pid != player_id and p.is_active and p.total_resources > 0
    ]
    if not opponents:
        return proposals

    held = [r for r in ResourceType if player.resources.get(r, 0) >= 1]
    surplus = [r for r in ResourceType if player.resources.get(r, 0) >= 2]

    for target in opponents:
        # 1-for-1 swaps
        for offer_r in held:
            for want_r in ResourceType:
                if want_r == offer_r:
                    continue
                proposals.append(TradeProposal(
                    offering={offer_r: 1},
                    requesting={want_r: 1},
                    target_player=target,
                ))
        # 2-for-1: spend a stockpile to get something scarce
        for offer_r in surplus:
            for want_r in ResourceType:
                if want_r == offer_r:
                    continue
                proposals.append(TradeProposal(
                    offering={offer_r: 2},
                    requesting={want_r: 1},
                    target_player=target,
                ))

    return proposals


def _can_buy_dev_card(state: GameState, player_id: int) -> bool:
    player = state.players[player_id]
    return (
        _player_can_afford(player.resources, DEV_CARD_COST) and
        len(state.dev_card_deck) > 0
    )


def _can_play_dev_card(state: GameState, player_id: int) -> bool:
    player = state.players[player_id]
    return len(player.dev_cards) > 0 and not player.dev_card_played_this_turn


def _get_playable_dev_cards(state: GameState, player_id: int) -> list[PlayDevCard]:
    player = state.players[player_id]
    if player.dev_card_played_this_turn:
        return []
    # One of each type in hand
    seen = set()
    cards = []
    for card in player.dev_cards:
        if card not in seen:
            seen.add(card)
            cards.append(PlayDevCard(card))
    return cards


def _can_divine_intervention(state: GameState, player_id: int) -> bool:
    player = state.players[player_id]
    return _player_can_afford(player.resources, DIVINE_COST)


# ── RL Encoding Helpers ──────────────────────────────────────────────────

def action_to_index(action: Action, valid_actions: list[Action]) -> int:
    for i, a in enumerate(valid_actions):
        if a == action:
            return i
    raise ValueError(f"Action {action} not in valid actions")


def index_to_action(index: int, valid_actions: list[Action]) -> Action:
    return valid_actions[index]
