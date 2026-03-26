"""Feature encoding: convert Observations and Actions into numeric vectors.

State features (STATE_DIM = 36):
    [0:6]   my resources (wood, stone, metal, wheat, water, cow)
    [6:11]  my dev cards count per type (5 types)
    [11]    my progress points (normalized by WIN_THRESHOLD)
    [12]    my total resources
    [13:15] dice (d1/6, d2/6) — normalized
    [14]    turn number (normalized by MAX_TURNS)
    [16:22] bank supply per resource (normalized by 19)
    [22:26] opponent 0: total_resources, dev_cards, progress, is_active
    [26:30] opponent 1: total_resources, dev_cards, progress, is_active
    [30]    my settlement count
    [31]    my city count
    [32]    my road count
    [33]    total opponent buildings
    [34]    available settlement positions count (normalized)
    [35]    available city upgrade positions count (normalized)

Action features (ACTION_DIM = 13):
    One-hot action category (6) + subtype details (7):
    [0:6]   action type one-hot: BUILD, TRADE, BUY_DEV, PLAY_DEV, DIVINE, END_TURN
    [6:9]   build subtype one-hot: ROAD, SETTLEMENT, CITY (0s if not build)
    [9:14]  dev card subtype one-hot: 5 types (0s if not play_dev)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..data_types import (
    Action, ActionType, Build, BuildType, BuyDevCard, DevCardType,
    DivineIntervention, EndTurn, MAX_TURNS, Observation, PlayDevCard,
    ResourceType, TradeProposal, WIN_THRESHOLD,
)

if TYPE_CHECKING:
    pass

STATE_DIM = 36
ACTION_DIM = 14


def encode_observation(obs: Observation) -> list[float]:
    """Convert an Observation into a fixed-size float vector."""
    f: list[float] = []

    # My resources [0:6]
    for r in ResourceType:
        f.append(obs.my_resources.get(r, 0) / 19.0)

    # My dev cards per type [6:11]
    dev_counts = {dt: 0 for dt in DevCardType}
    for card in obs.my_dev_cards:
        dev_counts[card] += 1
    for dt in DevCardType:
        f.append(dev_counts[dt] / 4.0)

    # My progress points [11]
    my_pp = 0
    for opp in obs.opponents:
        pass  # we need our own points
    # Count from board
    board = obs.board
    my_settlements = 0
    my_cities = 0
    my_roads = 0
    opp_buildings = 0
    for inter in board.intersections.values():
        if inter.owner == obs.player_id:
            if inter.building == BuildType.SETTLEMENT:
                my_settlements += 1
                my_pp += 1
            elif inter.building == BuildType.CITY:
                my_cities += 1
                my_pp += 2
        elif inter.owner is not None and inter.building is not None:
            opp_buildings += 1
    for edge in board.edges.values():
        if edge.road_owner == obs.player_id:
            my_roads += 1

    f.append(my_pp / WIN_THRESHOLD)

    # Total resources [12]
    f.append(sum(obs.my_resources.values()) / 30.0)

    # Dice [13:15]
    if obs.current_dice:
        f.append(obs.current_dice[0] / 6.0)
        f.append(obs.current_dice[1] / 6.0)
    else:
        f.append(0.0)
        f.append(0.0)

    # Turn number [15]
    f.append(obs.turn_number / MAX_TURNS)

    # Bank supply [16:22]
    for r in ResourceType:
        f.append(obs.bank_supply.get(r, 0) / 19.0)

    # Opponents [22:30] — pad to exactly 2 opponents
    sorted_opps = sorted(obs.opponents, key=lambda o: o.player_id)
    for i in range(2):
        if i < len(sorted_opps):
            opp = sorted_opps[i]
            f.append(opp.total_resources / 30.0)
            f.append(opp.num_dev_cards / 10.0)
            f.append(opp.progress_points / WIN_THRESHOLD)
            f.append(1.0 if opp.is_active else 0.0)
        else:
            f.extend([0.0, 0.0, 0.0, 0.0])

    # Board summary [30:36]
    f.append(my_settlements / 5.0)
    f.append(my_cities / 4.0)
    f.append(my_roads / 15.0)
    f.append(opp_buildings / 20.0)

    # Available positions (give the agent a sense of board openness)
    valid_settlements = len(board.get_valid_settlement_positions(obs.player_id))
    valid_cities = len(board.get_valid_city_positions(obs.player_id))
    f.append(valid_settlements / 20.0)
    f.append(valid_cities / 5.0)

    assert len(f) == STATE_DIM, f"Expected {STATE_DIM}, got {len(f)}"
    return f


def encode_action(action: Action) -> list[float]:
    """Convert an Action into a fixed-size float vector."""
    f = [0.0] * ACTION_DIM

    # Action type one-hot [0:6]
    type_map = {
        ActionType.BUILD: 0,
        ActionType.TRADE: 1,
        ActionType.BUY_DEV: 2,
        ActionType.PLAY_DEV: 3,
        ActionType.DIVINE: 4,
        ActionType.END_TURN: 5,
    }
    f[type_map[action.action_type]] = 1.0

    # Build subtype [6:9]
    if isinstance(action, Build):
        build_map = {BuildType.ROAD: 6, BuildType.SETTLEMENT: 7, BuildType.CITY: 8}
        f[build_map[action.build_type]] = 1.0

    # Dev card subtype [9:14]
    if isinstance(action, PlayDevCard):
        dev_map = {
            DevCardType.EXPANSIONIST: 9,
            DevCardType.ESPIONAGE: 10,
            DevCardType.MAINTENANCE: 11,
            DevCardType.INVENTION: 12,
            DevCardType.PLUNDER: 13,
        }
        f[dev_map[action.card_type]] = 1.0

    return f
