"""Game loop orchestration for CivSim."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .action_executors import execute_action
from .actions import get_valid_actions
from .agents import Agent
from .data_types import (
    ActionResult, EndTurn, GameResult, Observation, ResourceType,
)
from .game_state import GameState


@dataclass
class Environment:
    agents: dict[int, Agent]
    num_players: int = 3
    state: Optional[GameState] = None

    def __init__(self, agents: list[Agent], num_players: int = 3):
        self.num_players = num_players
        self.agents = {a.player_id: a for a in agents}

    def reset(self, seed: Optional[int] = None) -> Observation:
        self.state = GameState.new_game(num_players=self.num_players, seed=seed)
        self.run_snake_draft()
        self._start_turn()
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

            # Place road adjacent to settlement
            valid_roads = self.state.board.get_valid_draft_roads(pid, settlement_id)
            if not valid_roads:
                continue
            obs = self.state.get_observation(pid)
            road_id = agent.select_draft_road(obs, valid_roads)
            self.state.board.place_road(pid, road_id)

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

    def run_game(self) -> GameResult:
        """Play a complete game and return results."""
        obs = self.reset()

        while not self.state.game_over:
            pid = self.state.current_player
            agent = self.agents[pid]

            # Action loop for current player
            self._run_action_loop(pid, agent)

            if self.state.game_over:
                break

        return self.state.get_final_results()

    def _start_turn(self) -> None:
        """Roll dice and produce resources."""
        self.state.roll_dice()
        self.state.produce_resources()
        # Reset per-turn flags
        player = self.state.players[self.state.current_player]
        player.dev_card_played_this_turn = False
        player.maintenance_paid_this_turn = False

    def _run_action_loop(self, player_id: int, agent: Agent) -> None:
        """Inner loop: get valid actions, agent picks, execute, repeat until EndTurn."""
        max_actions = 100  # safety cap per turn
        for _ in range(max_actions):
            obs = self.state.get_observation(player_id)
            valid = get_valid_actions(self.state, player_id)
            action = agent.select_action(obs, valid)

            if isinstance(action, EndTurn):
                self._end_turn()
                return

            execute_action(self.state, player_id, action, self.agents)

            if self.state.game_over:
                return

        # Forced end turn if action cap reached
        self._end_turn()

    def _end_turn(self) -> None:
        """Resolve maintenance, advance player, check game over."""
        pid = self.state.current_player
        self.state.resolve_maintenance(pid)
        self.state.update_progress_points(pid)

        if self.state.is_game_over():
            return

        self.state.advance_player()
        self._start_turn()
