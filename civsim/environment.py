"""Game loop orchestration for CivSim."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .action_executors import execute_action
from .actions import get_valid_actions
from .agents import Agent
from .data_types import (
    ActionResult, BuildType, EndTurn, GameResult, MAINTENANCE_TURN_INTERVAL,
    Observation, ResourceType, STARTING_WATER_STOCKPILE,
)
from .game_state import GameState


@dataclass
class Environment:
    agents: dict[int, Agent]
    num_players: int = 3
    state: Optional[GameState] = None
    renderer: Optional[Any] = None  # TerminalRenderer or compatible

    def __init__(
        self,
        agents: list[Agent],
        num_players: int = 3,
        renderer: Optional[Any] = None,
    ):
        self.num_players = num_players
        self.agents = {a.player_id: a for a in agents}
        self.renderer = renderer

    def reset(self, seed: Optional[int] = None) -> Observation:
        self.state = GameState.new_game(num_players=self.num_players, seed=seed)
        self.run_snake_draft()
        self._start_turn()
        if self.renderer is not None:
            self.renderer.render_event("=== Draft complete, game begins ===")
            self.renderer.render(self.state)
        return self.state.get_observation(self.state.current_player)

    def run_snake_draft(self) -> None:
        """Snake draft: players place 2 settlements + 2 roads each."""
        n = self.num_players
        # Snake order: [0,1,2,2,1,0] for 3 players
        order = list(range(n)) + list(range(n - 1, -1, -1))

        for pid in order:
            agent = self.agents[pid]
            obs = self.state.get_observation(pid)

            # Place settlement
            valid_settlements = self.state.board.get_valid_draft_settlements(pid)
            if not valid_settlements:
                continue
            settlement_id = agent.select_draft_action(obs, valid_settlements)
            self.state.board.place_settlement(pid, settlement_id)
            self.state.players[pid].progress_points += 1
            self.state.players[pid].stats.record_build(BuildType.SETTLEMENT)

            # Place road adjacent to settlement
            valid_roads = self.state.board.get_valid_draft_roads(pid, settlement_id)
            if not valid_roads:
                continue
            obs = self.state.get_observation(pid)
            road_id = agent.select_draft_road(obs, valid_roads)
            self.state.board.place_road(pid, road_id)
            self.state.players[pid].stats.record_build(BuildType.ROAD)

        # Distribute starting resources
        self._distribute_starting_resources()

    def _distribute_starting_resources(self) -> None:
        """Each player receives 1 resource from each tile adjacent to their settlements."""
        for inter in self.state.board.intersections.values():
            if inter.owner is None or inter.building is None:
                continue
            pid = inter.owner
            for tile in self.state.board.get_tiles_for_intersection(inter.inter_id):
                if tile.resource is not None:
                    r = tile.resource
                    if self.state.bank.get(r, 0) > 0:
                        self.state.players[pid].resources[r] += 1
                        self.state.bank[r] -= 1
                        self.state.players[pid].stats.total_resources_earned += 1

        # Additive starting water stockpile (does not deduct from bank).
        # Addresses the structural water shortage where draft-tile production
        # alone leaves most players unable to cover even one round of upkeep.
        for pid, p in self.state.players.items():
            p.resources[ResourceType.WATER] = (
                p.resources.get(ResourceType.WATER, 0) + STARTING_WATER_STOCKPILE
            )
            p.stats.total_resources_earned += STARTING_WATER_STOCKPILE

    def step(self, action) -> tuple[Observation, float, bool, dict]:
        """Execute one action. Returns (observation, reward, done, info)."""
        pid = self.state.current_player
        old_score = self.state.players[pid].progress_points

        result = execute_action(self.state, pid, action, self.agents)

        if isinstance(action, EndTurn):
            self._end_turn()

        new_score = self.state.players[self.state.current_player].progress_points
        reward = float(new_score - old_score)
        done = self.state.game_over

        obs = self.state.get_observation(self.state.current_player)
        info = {"action_result": result}

        return obs, reward, done, info

    def run_game(self, seed: Optional[int] = None) -> GameResult:
        """Play a complete game and return results. Pass `seed` to fully
        determinize the game state RNG (board layout, dice, weather,
        divine outcomes); leave None for a fresh random game."""
        obs = self.reset(seed=seed)

        while not self.state.game_over:
            pid = self.state.current_player
            agent = self.agents[pid]

            # Action loop for current player
            self._run_action_loop(pid, agent)

            if self.state.game_over:
                break

        return self.state.get_final_results()

    def _start_turn(self) -> None:
        """Roll weather (rain / barn day), roll dice, produce resources.

        Delegates to GameState.start_player_turn() so MCTS rollouts run the
        exact same transition logic and don't drift from real gameplay.
        """
        rained, barned = self.state.start_player_turn()
        if self.renderer is not None:
            if rained:
                self.renderer.render_event(
                    f"[RAIN]  Turn {self.state.turn_number}: +1 water for all players"
                )
            if barned:
                self.renderer.render_event(
                    f"[BARN]  Turn {self.state.turn_number}: +1 cow for all players"
                )

    def _run_action_loop(self, player_id: int, agent: Agent) -> None:
        """Inner loop: get valid actions, agent picks, execute, repeat until EndTurn.

        Safeguard: if the agent makes too many consecutive no-progress
        attempts (e.g. repeatedly proposing trades that get rejected),
        force-end the turn. Agents also receive a per-action callback so
        they can remember what just failed and pick differently next time.
        """
        agent.on_turn_start()
        max_actions = 100
        max_consecutive_failures = 5
        consecutive_failures = 0

        for _ in range(max_actions):
            obs = self.state.get_observation(player_id)
            valid = get_valid_actions(self.state, player_id)
            action = agent.select_action(obs, valid)

            if isinstance(action, EndTurn):
                self._end_turn()
                return

            result = execute_action(self.state, player_id, action, self.agents)
            agent.on_action_result(action, result.success)
            if self.renderer is not None:
                self.renderer.render(self.state, last_action=result)

            # Check win condition after every action — a build can push us
            # to WIN_THRESHOLD mid-turn, and we shouldn't keep playing past
            # that. Without this, maintenance at end-of-turn can silently
            # revert the win.
            if self.state.is_game_over():
                if self.renderer is not None:
                    self.renderer.render_event(
                        f"=== Game over: winner = P{self.state.winner} "
                        f"(turn {self.state.turn_number}) ==="
                    )
                return

            if result.success:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    self._end_turn()
                    return

        # Forced end turn if action cap reached
        self._end_turn()

    def _end_turn(self) -> None:
        """Resolve maintenance (every Nth own-turn per player), advance, check."""
        pid = self.state.current_player
        events, skipped_maintenance = self.state.end_player_turn(pid)

        if self.renderer is not None:
            lost = sum(1 for e in events if not e.paid)
            note = f"P{pid} end-of-turn"
            if skipped_maintenance:
                note += " (no upkeep this turn)"
            elif lost:
                note += f" — lost {lost} building(s) to upkeep"
            self.renderer.render_event(note)

        if self.state.is_game_over():
            if self.renderer is not None:
                self.renderer.render(self.state)
                self.renderer.render_event(
                    f"=== Game over: winner = P{self.state.winner} "
                    f"(turn {self.state.turn_number}) ==="
                )
            return

        self.state.advance_player()
        self._start_turn()
        if self.renderer is not None:
            self.renderer.render(self.state)
