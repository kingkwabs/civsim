"""Core game state management for CivSim."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Optional

from .board import Board
from .data_types import (
    BANK_STARTING_SUPPLY, BARN_COW_PER_PLAYER, BARN_PROBABILITY,
    BuildType, CITY_PP, DEV_CARD_COUNTS, DevCardType,
    GameResult, MAINTENANCE_COSTS, MAINTENANCE_TURN_INTERVAL,
    MaintenanceEvent, MAX_TURNS,
    Observation, OpponentView, Player, PlayerStats, RAIN_PROBABILITY,
    RAIN_WATER_PER_PLAYER, ResourceDict, ResourceType, SETTLEMENT_PP,
    WIN_THRESHOLD,
)


@dataclass
class GameState:
    board: Board = field(default_factory=Board)
    players: dict[int, Player] = field(default_factory=dict)
    bank: ResourceDict = field(default_factory=dict)
    dev_card_deck: list[DevCardType] = field(default_factory=list)
    current_player: int = 0
    current_dice: Optional[tuple[int, int]] = None
    turn_number: int = 0
    num_players: int = 3
    game_over: bool = False
    winner: Optional[int] = None
    rng: random.Random = field(default_factory=random.Random)

    @classmethod
    def new_game(cls, num_players: int = 3, seed: Optional[int] = None) -> GameState:
        rng = random.Random(seed)
        state = cls(
            board=Board.standard_layout(seed),
            num_players=num_players,
            rng=rng,
        )
        # Init players
        for pid in range(num_players):
            state.players[pid] = Player(player_id=pid)

        # Stock bank
        state.bank = {r: BANK_STARTING_SUPPLY for r in ResourceType}

        # Shuffle dev card deck
        deck: list[DevCardType] = []
        for card_type, count in DEV_CARD_COUNTS.items():
            deck.extend([card_type] * count)
        rng.shuffle(deck)
        state.dev_card_deck = deck

        # Choose first player
        state.current_player = rng.randint(0, num_players - 1)

        return state

    def roll_dice(self) -> tuple[int, int]:
        d1 = self.rng.randint(1, 6)
        d2 = self.rng.randint(1, 6)
        self.current_dice = (d1, d2)
        return (d1, d2)

    def roll_rain(self) -> bool:
        """Sample whether it rains this turn. Returns True with probability
        RAIN_PROBABILITY using the game's RNG (reproducible from seed)."""
        return self.rng.random() < RAIN_PROBABILITY

    def roll_barn_day(self) -> bool:
        """Sample whether barn day fires this turn. Symmetric to rain;
        BARN_PROBABILITY chance, drawn from the same RNG for reproducibility."""
        return self.rng.random() < BARN_PROBABILITY

    def apply_rain(self) -> list[int]:
        """Grant RAIN_WATER_PER_PLAYER water to every active player
        *additively* — not deducted from the bank. Returns the list of
        recipient player_ids so callers can render/log.
        """
        return self._apply_weather_grant(ResourceType.WATER, RAIN_WATER_PER_PLAYER, "rain")

    def apply_barn_day(self) -> list[int]:
        """Grant BARN_COW_PER_PLAYER cow to every active player additively."""
        return self._apply_weather_grant(ResourceType.COW, BARN_COW_PER_PLAYER, "barn")

    def _apply_weather_grant(self, resource: ResourceType, amount: int, kind: str) -> list[int]:
        recipients: list[int] = []
        for pid, p in self.players.items():
            if not p.is_active:
                continue
            p.resources[resource] = p.resources.get(resource, 0) + amount
            p.stats.total_resources_earned += amount
            if kind == "rain":
                p.stats.rain_received += amount
            elif kind == "barn":
                p.stats.barn_received += amount
            recipients.append(pid)
        return recipients

    def dice_sum(self) -> int:
        if self.current_dice is None:
            return 0
        return self.current_dice[0] + self.current_dice[1]

    def produce_resources(self) -> dict[int, ResourceDict]:
        """Produce resources for all players based on dice roll."""
        production: dict[int, ResourceDict] = {pid: {} for pid in self.players}
        ds = self.dice_sum()
        if ds == 0:
            return production

        for inter in self.board.intersections.values():
            if inter.building is None or inter.owner is None:
                continue
            player_id = inter.owner
            amount = 1 if inter.building == BuildType.SETTLEMENT else 2

            for tile in self.board.get_tiles_for_intersection(inter.inter_id):
                if tile.resource is None or tile.dice_number != ds:
                    continue
                resource = tile.resource
                # Check bank has enough
                available = self.bank.get(resource, 0)
                give = min(amount, available)
                if give > 0:
                    self.players[player_id].resources[resource] = (
                        self.players[player_id].resources.get(resource, 0) + give
                    )
                    self.bank[resource] -= give
                    self.players[player_id].stats.total_resources_earned += give
                    production[player_id][resource] = (
                        production[player_id].get(resource, 0) + give
                    )

        return production

    def resolve_maintenance(self, player_id: int) -> list[MaintenanceEvent]:
        player = self.players[player_id]
        events: list[MaintenanceEvent] = []

        # Skip if maintenance card was played
        if player.maintenance_paid_this_turn:
            player.maintenance_paid_this_turn = False
            return events

        for inter in self.board.intersections.values():
            if inter.owner != player_id or inter.building is None:
                continue
            if inter.building not in MAINTENANCE_COSTS:
                continue

            cost = MAINTENANCE_COSTS[inter.building]
            if player.can_afford(cost):
                # Pay maintenance
                for r, amt in cost.items():
                    player.resources[r] -= amt
                    self.bank[r] = self.bank.get(r, 0) + amt
                    player.stats.total_resources_spent += amt
                events.append(MaintenanceEvent(
                    inter_id=inter.inter_id,
                    player_id=player_id,
                    building_type=inter.building,
                    paid=True,
                    cost=dict(cost),
                ))
            else:
                # Lose ownership — update_progress_points is also called
                # right after this returns, so this decrement is for
                # immediate-feedback callers (and gets normalized by the
                # full recompute).
                self.board.abandon_building(inter.inter_id)
                player.progress_points -= (
                    SETTLEMENT_PP if inter.building == BuildType.SETTLEMENT else CITY_PP
                )
                player.stats.buildings_lost += 1
                player.stats.maintenance_failures += 1
                events.append(MaintenanceEvent(
                    inter_id=inter.inter_id,
                    player_id=player_id,
                    building_type=inter.building,
                    paid=False,
                ))

        return events

    def update_progress_points(self, player_id: int) -> int:
        points = 0
        for inter in self.board.intersections.values():
            if inter.owner == player_id:
                if inter.building == BuildType.SETTLEMENT:
                    points += SETTLEMENT_PP
                elif inter.building == BuildType.CITY:
                    points += CITY_PP
        self.players[player_id].progress_points = points
        return points

    def is_game_over(self) -> bool:
        # Check progress point threshold
        for player in self.players.values():
            if player.is_active and player.progress_points >= WIN_THRESHOLD:
                self.game_over = True
                self.winner = player.player_id
                return True

        # Check if only one player remains
        active = [p for p in self.players.values() if p.is_active]
        if len(active) <= 1:
            self.game_over = True
            self.winner = active[0].player_id if active else None
            return True

        # Turn cap
        if self.turn_number >= MAX_TURNS:
            self.game_over = True
            best = max(self.players.values(), key=lambda p: p.progress_points)
            self.winner = best.player_id
            return True

        return False

    def get_scores(self) -> dict[int, int]:
        return {pid: p.progress_points for pid, p in self.players.items()}

    def get_observation(self, player_id: int) -> Observation:
        player = self.players[player_id]
        opponents = []
        for pid, p in self.players.items():
            if pid != player_id:
                opponents.append(OpponentView(
                    player_id=pid,
                    total_resources=p.total_resources,
                    num_dev_cards=len(p.dev_cards),
                    progress_points=p.progress_points,
                    is_active=p.is_active,
                ))

        # Vague bank representation (rounded to nearest 5)
        vague_bank = {r: (amt // 5) * 5 for r, amt in self.bank.items()}

        return Observation(
            player_id=player_id,
            my_resources=dict(player.resources),
            my_dev_cards=list(player.dev_cards),
            opponents=opponents,
            board=self.board,
            bank_supply=vague_bank,
            current_dice=self.current_dice,
            turn_number=self.turn_number,
        )

    def get_final_results(self) -> GameResult:
        stats: dict[int, PlayerStats] = {}
        for pid, p in self.players.items():
            buildings = sum(
                1 for i in self.board.intersections.values()
                if i.owner == pid and i.building is not None
            )
            # Snapshot live stats and fold in end-of-game-only fields
            p.stats.final_resources = dict(p.resources)
            p.stats.buildings_owned = buildings
            stats[pid] = p.stats

        return GameResult(
            winner=self.winner,
            scores=self.get_scores(),
            turns_played=self.turn_number,
            player_stats=stats,
        )

    def advance_player(self) -> int:
        next_id = (self.current_player + 1) % self.num_players
        attempts = 0
        while not self.players[next_id].is_active:
            next_id = (next_id + 1) % self.num_players
            attempts += 1
            if attempts >= self.num_players:
                break
        self.current_player = next_id
        self.turn_number += 1
        return next_id

    def clone(self) -> GameState:
        return copy.deepcopy(self)

    # ── Turn transitions (shared by Environment and MCTS) ───────────────
    #
    # Both the live game loop and any agent that simulates futures need to
    # apply the exact same turn-transition rules: maintenance every Nth
    # own-turn, weather rolls before dice production, per-turn flag resets.
    # Putting them here means MCTS and the real env can't drift.

    def end_player_turn(self, player_id: int) -> tuple[list[MaintenanceEvent], bool]:
        """Resolve end-of-turn for `player_id`. Returns (maintenance_events,
        skipped_maintenance). Maintenance only fires every Nth own-turn."""
        p = self.players[player_id]
        p.turns_since_maintenance += 1
        if p.turns_since_maintenance >= MAINTENANCE_TURN_INTERVAL:
            events = self.resolve_maintenance(player_id)
            p.turns_since_maintenance = 0
            self.update_progress_points(player_id)
            return events, False
        self.update_progress_points(player_id)
        return [], True

    def start_player_turn(self) -> tuple[bool, bool]:
        """Roll weather, then dice, then produce resources. Resets per-turn
        flags for the current player. Returns (rained, barned)."""
        rained = self.roll_rain()
        if rained:
            self.apply_rain()
        barned = self.roll_barn_day()
        if barned:
            self.apply_barn_day()

        self.roll_dice()
        self.produce_resources()

        player = self.players[self.current_player]
        player.dev_card_played_this_turn = False
        player.maintenance_paid_this_turn = False
        return rained, barned
