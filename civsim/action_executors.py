"""Action execution logic for CivSim."""
from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional

from .data_types import (
    Action, ActionResult, ActionType, Build, BuildType, BUILDING_COSTS,
    BuyDevCard, CITY_PP, DEV_CARD_COST, DevCardType, DIVINE_COST,
    DIVINE_EVENTS, DivineIntervention, EndTurn, MAINTENANCE_COSTS,
    PlayDevCard, REESTABLISHMENT_COSTS, ReestablishBuilding, ResourceDict,
    ResourceType, SETTLEMENT_PP, TradeProposal,
)

if TYPE_CHECKING:
    from .agents import Agent
    from .game_state import GameState


def execute_action(
    state: GameState,
    player_id: int,
    action: Action,
    agents: Optional[dict[int, Agent]] = None,
) -> ActionResult:
    if isinstance(action, Build):
        return execute_build(state, player_id, action)
    elif isinstance(action, TradeProposal):
        return execute_trade(state, player_id, action, agents)
    elif isinstance(action, BuyDevCard):
        return execute_buy_dev_card(state, player_id)
    elif isinstance(action, PlayDevCard):
        return execute_play_dev_card(state, player_id, action, agents)
    elif isinstance(action, DivineIntervention):
        return execute_divine_intervention(state, player_id)
    elif isinstance(action, EndTurn):
        return ActionResult(success=True, action=action, description="Turn ended")
    elif isinstance(action, ReestablishBuilding):
        return execute_reestablish(state, player_id, action)
    else:
        return ActionResult(success=False, action=action, description="Unknown action type")


def execute_reestablish(
    state: GameState, player_id: int, action: ReestablishBuilding
) -> ActionResult:
    """Claim an abandoned building at the (higher) re-establishment cost."""
    player = state.players[player_id]
    inter = state.board.intersections.get(action.position)
    if inter is None or inter.building is None or inter.owner is not None:
        return ActionResult(
            success=False, action=action,
            description="Target intersection is not abandoned",
        )
    if inter.building != action.build_type:
        return ActionResult(
            success=False, action=action,
            description=f"Building type mismatch (expected {action.build_type.name})",
        )
    cost = REESTABLISHMENT_COSTS.get(action.build_type)
    if cost is None:
        return ActionResult(success=False, action=action,
                            description="No re-establishment cost defined")
    if not player.can_afford(cost):
        return ActionResult(success=False, action=action,
                            description="Insufficient resources to re-establish")

    _deduct_resources(state, player_id, cost)
    state.board.claim_building(player_id, action.position)
    # Award progress points immediately; end-of-turn update_progress_points
    # will normalize against the live board anyway.
    if action.build_type == BuildType.SETTLEMENT:
        player.progress_points += SETTLEMENT_PP
    else:
        player.progress_points += CITY_PP
    player.stats.record_build(action.build_type)
    return ActionResult(
        success=True, action=action,
        description=f"Re-established {action.build_type.name.lower()} at i{action.position}",
    )


# ── Build ────────────────────────────────────────────────────────────────

def execute_build(state: GameState, player_id: int, action: Build) -> ActionResult:
    player = state.players[player_id]
    cost = BUILDING_COSTS[action.build_type]

    _deduct_resources(state, player_id, cost)
    player.stats.record_build(action.build_type)

    if action.build_type == BuildType.ROAD:
        state.board.place_road(player_id, action.position)
        return ActionResult(success=True, action=action, description=f"Built road at edge {action.position}")

    elif action.build_type == BuildType.SETTLEMENT:
        state.board.place_settlement(player_id, action.position)
        from .data_types import SETTLEMENT_PP
        player.progress_points += SETTLEMENT_PP
        return ActionResult(success=True, action=action, description=f"Built settlement at {action.position}")

    elif action.build_type == BuildType.CITY:
        state.board.upgrade_to_city(player_id, action.position)
        from .data_types import CITY_PP, SETTLEMENT_PP
        player.progress_points += (CITY_PP - SETTLEMENT_PP)  # diff from settlement
        return ActionResult(success=True, action=action, description=f"Upgraded to city at {action.position}")

    return ActionResult(success=False, action=action, description="Invalid build type")


# ── Trade ────────────────────────────────────────────────────────────────

def execute_trade(
    state: GameState,
    player_id: int,
    action: TradeProposal,
    agents: Optional[dict[int, Agent]] = None,
) -> ActionResult:
    player = state.players[player_id]
    # Count every trade attempt — success or fail — toward the per-turn cap
    # so a stubborn proposer can't burn the whole turn on rejected trades.
    player.trades_this_turn += 1

    if action.target_player is not None:
        # Player-to-player trade — requires the target agent to consent.
        # Without `agents` (e.g. inside MCTS rollouts), treat as rejected
        # rather than silently falling through to the bank-trade path.
        if not agents:
            return ActionResult(
                success=False, action=action,
                description="Trade unresolved (no responder available)",
            )

        target = state.players[action.target_player]
        target_agent = agents.get(action.target_player)
        if target_agent is None:
            return ActionResult(success=False, action=action, description="Target agent not found")

        # Sanity: target must actually have the requested resources
        if not target.can_afford(action.requesting):
            return ActionResult(
                success=False, action=action,
                description=f"Trade rejected: P{action.target_player} can't cover request",
            )

        obs = state.get_observation(action.target_player)
        accepted = target_agent.respond_to_trade(action, obs)
        if not accepted:
            return ActionResult(
                success=False, action=action,
                description=f"Trade rejected by P{action.target_player}",
            )

        # Execute swap
        _transfer_resources_between(state, player_id, action.target_player,
                                    action.offering, action.requesting)
        player.stats.trades_made += 1
        player.stats.player_trades += 1
        target.stats.trades_made += 1
        target.stats.player_trades += 1
        return ActionResult(
            success=True, action=action,
            description=f"Trade with P{action.target_player}: "
                        f"gave {_fmt_bundle(action.offering)} "
                        f"got {_fmt_bundle(action.requesting)}",
        )

    else:
        # Bank trade (4:1, 3:1 generic port, or 2:1 specific port)
        for r, amt in action.offering.items():
            player.resources[r] -= amt
            state.bank[r] = state.bank.get(r, 0) + amt
            player.stats.total_resources_spent += amt
        for r, amt in action.requesting.items():
            state.bank[r] -= amt
            player.resources[r] = player.resources.get(r, 0) + amt
            player.stats.total_resources_earned += amt
        player.stats.trades_made += 1
        player.stats.bank_trades += 1
        # Build a description that includes the ratio + whether a port
        # was used — otherwise port utilization is invisible in the log.
        offered = sum(action.offering.values())
        requested = sum(action.requesting.values()) or 1
        ratio = offered // requested
        if action.port_type is not None:
            kind = "generic port" if action.port_type.name == "GENERIC" else f"{action.port_type.name.lower()} port"
            desc = f"Port trade ({ratio}:1, {kind}): {_fmt_bundle(action.offering)} → {_fmt_bundle(action.requesting)}"
        else:
            desc = f"Bank trade ({ratio}:1): {_fmt_bundle(action.offering)} → {_fmt_bundle(action.requesting)}"
        return ActionResult(success=True, action=action, description=desc)


# ── Buy Dev Card ─────────────────────────────────────────────────────────

def execute_buy_dev_card(state: GameState, player_id: int) -> ActionResult:
    if not state.dev_card_deck:
        return ActionResult(success=False, action=BuyDevCard(), description="Deck is empty")
    _deduct_resources(state, player_id, DEV_CARD_COST)
    card = state.dev_card_deck.pop()
    state.players[player_id].dev_cards.append(card)
    state.players[player_id].stats.dev_cards_bought += 1

    return ActionResult(
        success=True,
        action=BuyDevCard(),
        description=f"Bought dev card: {card.name}",
        details={"card": card},
    )


# ── Play Dev Card ────────────────────────────────────────────────────────

def execute_play_dev_card(
    state: GameState,
    player_id: int,
    action: PlayDevCard,
    agents: Optional[dict[int, Agent]] = None,
) -> ActionResult:
    player = state.players[player_id]
    player.dev_cards.remove(action.card_type)
    player.dev_card_played_this_turn = True
    player.stats.dev_cards_played += 1

    if action.card_type == DevCardType.EXPANSIONIST:
        return _execute_expansionist(state, player_id, agents)
    elif action.card_type == DevCardType.ESPIONAGE:
        return _execute_espionage(state, player_id, agents)
    elif action.card_type == DevCardType.MAINTENANCE:
        return _execute_maintenance_card(state, player_id)
    elif action.card_type == DevCardType.INVENTION:
        return _execute_invention(state, player_id, agents)
    elif action.card_type == DevCardType.PLUNDER:
        return _execute_plunder(state, player_id, agents)

    return ActionResult(success=False, action=action, description="Unknown card type")


def _execute_expansionist(
    state: GameState, player_id: int, agents: Optional[dict[int, Agent]]
) -> ActionResult:
    agent = agents.get(player_id) if agents else None
    roads_placed = 0
    for _ in range(2):
        valid = state.board.get_valid_road_positions(player_id)
        if not valid:
            break
        if agent:
            pos = agent.select_road(valid)
        else:
            pos = state.rng.choice(valid)
        state.board.place_road(player_id, pos)
        roads_placed += 1

    return ActionResult(
        success=True,
        action=PlayDevCard(DevCardType.EXPANSIONIST),
        description=f"Expansionist: placed {roads_placed} free roads",
    )


def _execute_espionage(
    state: GameState, player_id: int, agents: Optional[dict[int, Agent]]
) -> ActionResult:
    opponents = [pid for pid in state.players if pid != player_id and state.players[pid].is_active]
    if not opponents:
        return ActionResult(success=True, action=PlayDevCard(DevCardType.ESPIONAGE),
                            description="Espionage: no opponents")

    agent = agents.get(player_id) if agents else None
    if agent:
        target_id = agent.select_target(opponents)
    else:
        target_id = state.rng.choice(opponents)

    target = state.players[target_id]
    # Build pool of target's resources
    pool: list[ResourceType] = []
    for r, amt in target.resources.items():
        pool.extend([r] * amt)

    steal_count = min(2, len(pool))
    if steal_count == 0:
        return ActionResult(success=True, action=PlayDevCard(DevCardType.ESPIONAGE),
                            description="Espionage: target has no resources")

    stolen_items = state.rng.sample(pool, steal_count)
    stolen: ResourceDict = {}
    for r in stolen_items:
        stolen[r] = stolen.get(r, 0) + 1

    for r, amt in stolen.items():
        target.resources[r] -= amt
        state.players[player_id].resources[r] = state.players[player_id].resources.get(r, 0) + amt
        state.players[player_id].stats.total_resources_earned += amt

    return ActionResult(
        success=True,
        action=PlayDevCard(DevCardType.ESPIONAGE),
        description=f"Espionage: stole {stolen} from player {target_id}",
    )


def _execute_maintenance_card(state: GameState, player_id: int) -> ActionResult:
    state.players[player_id].maintenance_paid_this_turn = True
    return ActionResult(
        success=True,
        action=PlayDevCard(DevCardType.MAINTENANCE),
        description="Maintenance card: all upkeep covered this turn",
    )


def _execute_invention(
    state: GameState, player_id: int, agents: Optional[dict[int, Agent]]
) -> ActionResult:
    agent = agents.get(player_id) if agents else None
    if agent:
        r1, r2 = agent.select_two_resources()
    else:
        resources = list(ResourceType)
        r1, r2 = state.rng.choice(resources), state.rng.choice(resources)

    player = state.players[player_id]
    for r in [r1, r2]:
        if state.bank.get(r, 0) > 0:
            player.resources[r] = player.resources.get(r, 0) + 1
            state.bank[r] -= 1
            player.stats.total_resources_earned += 1

    return ActionResult(
        success=True,
        action=PlayDevCard(DevCardType.INVENTION),
        description=f"Invention: created {r1.name} and {r2.name}",
    )


def _execute_plunder(
    state: GameState, player_id: int, agents: Optional[dict[int, Agent]]
) -> ActionResult:
    agent = agents.get(player_id) if agents else None
    if agent:
        resource = agent.select_resource_type()
    else:
        resource = state.rng.choice(list(ResourceType))

    total_taken = 0
    for pid, p in state.players.items():
        if pid == player_id:
            continue
        amount = p.resources.get(resource, 0)
        if amount > 0:
            p.resources[resource] = 0
            state.players[player_id].resources[resource] = (
                state.players[player_id].resources.get(resource, 0) + amount
            )
            state.players[player_id].stats.total_resources_earned += amount
            total_taken += amount

    return ActionResult(
        success=True,
        action=PlayDevCard(DevCardType.PLUNDER),
        description=f"Plunder: took {total_taken} {resource.name} from all opponents",
    )


# ── Divine Intervention ──────────────────────────────────────────────────

def execute_divine_intervention(state: GameState, player_id: int) -> ActionResult:
    _deduct_resources(state, player_id, DIVINE_COST)
    outcome = _sample_divine_event(state.rng)
    _apply_divine_outcome(state, player_id, outcome)
    state.players[player_id].stats.divine_interventions += 1
    state.players[player_id].stats.record_divine_outcome(outcome)
    return ActionResult(
        success=True,
        action=DivineIntervention(),
        description=f"Divine intervention: {outcome}",
        details={"outcome": outcome},
    )


def _sample_divine_event(rng: random.Random) -> str:
    roll = rng.random()
    cumulative = 0.0
    for event_name, info in DIVINE_EVENTS.items():
        cumulative += info["prob"]
        if roll < cumulative:
            return event_name
    return "blessing"  # fallback


def _apply_divine_outcome(state: GameState, player_id: int, outcome: str) -> None:
    player = state.players[player_id]

    if outcome == "blessing":
        # Receive 3 random resources from bank
        available = [r for r in ResourceType if state.bank.get(r, 0) > 0]
        for _ in range(3):
            if not available:
                break
            r = state.rng.choice(available)
            player.resources[r] = player.resources.get(r, 0) + 1
            state.bank[r] -= 1
            player.stats.total_resources_earned += 1
            if state.bank[r] <= 0:
                available.remove(r)

    elif outcome == "famine":
        # All opponents lose 2 random resources
        for pid, p in state.players.items():
            if pid == player_id:
                continue
            pool: list[ResourceType] = []
            for r, amt in p.resources.items():
                pool.extend([r] * amt)
            lose_count = min(2, len(pool))
            if lose_count > 0:
                lost = state.rng.sample(pool, lose_count)
                for r in lost:
                    p.resources[r] -= 1
                    state.bank[r] = state.bank.get(r, 0) + 1

    elif outcome == "earthquake":
        # Destroy one random road
        all_roads = [e for e in state.board.edges.values() if e.road_owner is not None]
        if all_roads:
            road = state.rng.choice(all_roads)
            road.road_owner = None

    elif outcome == "prosperity":
        # All players receive 1 resource from each settlement
        for pid, p in state.players.items():
            for inter in state.board.intersections.values():
                if inter.owner == pid and inter.building is not None:
                    tiles = state.board.get_tiles_for_intersection(inter.inter_id)
                    for tile in tiles:
                        if tile.resource and state.bank.get(tile.resource, 0) > 0:
                            p.resources[tile.resource] = p.resources.get(tile.resource, 0) + 1
                            state.bank[tile.resource] -= 1
                            p.stats.total_resources_earned += 1
                            break  # 1 resource per building

    elif outcome == "plague":
        # Lose 1 settlement
        settlements = [
            i for i in state.board.intersections.values()
            if i.owner == player_id and i.building == BuildType.SETTLEMENT
        ]
        if settlements:
            from .data_types import SETTLEMENT_PP
            victim = state.rng.choice(settlements)
            victim.building = None
            victim.owner = None
            player.progress_points -= SETTLEMENT_PP
            player.stats.buildings_lost += 1

    elif outcome == "miracle":
        # Gain 2 free progress points
        player.progress_points += 2


# ── Helpers ──────────────────────────────────────────────────────────────

def _deduct_resources(state: GameState, player_id: int, cost: ResourceDict) -> None:
    player = state.players[player_id]
    for r, amt in cost.items():
        player.resources[r] -= amt
        state.bank[r] = state.bank.get(r, 0) + amt
        player.stats.total_resources_spent += amt


def _transfer_resources_between(
    state: GameState,
    from_id: int,
    to_id: int,
    from_gives: ResourceDict,
    to_gives: ResourceDict,
) -> None:
    from_p = state.players[from_id]
    to_p = state.players[to_id]
    for r, amt in from_gives.items():
        from_p.resources[r] -= amt
        to_p.resources[r] = to_p.resources.get(r, 0) + amt
        from_p.stats.total_resources_spent += amt
        to_p.stats.total_resources_earned += amt
    for r, amt in to_gives.items():
        to_p.resources[r] -= amt
        from_p.resources[r] = from_p.resources.get(r, 0) + amt
        to_p.stats.total_resources_spent += amt
        from_p.stats.total_resources_earned += amt


def _return_to_bank(state: GameState, resources: ResourceDict) -> None:
    for r, amt in resources.items():
        state.bank[r] = state.bank.get(r, 0) + amt


def _fmt_bundle(bundle: ResourceDict) -> str:
    parts = [f"{amt}{r.name[0]}" for r, amt in bundle.items() if amt]
    return "+".join(parts) if parts else "∅"
