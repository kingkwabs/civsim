"""Tournament runner, metrics tracking, and replay logging for CivSim."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

from .agents import Agent
from .data_types import GameResult
from .environment import Environment


@dataclass
class TournamentResult:
    results: list[GameResult]
    win_counts: dict[str, int]
    avg_game_length: float
    total_time: float


@dataclass
class MetricsTracker:
    game_results: list[GameResult] = field(default_factory=list)
    agent_names: dict[int, str] = field(default_factory=dict)

    def record_game(self, result: GameResult) -> None:
        self.game_results.append(result)

    def get_win_rates(self) -> dict[int, float]:
        if not self.game_results:
            return {}
        wins: dict[int, int] = {}
        for result in self.game_results:
            if result.winner is not None:
                wins[result.winner] = wins.get(result.winner, 0) + 1
        total = len(self.game_results)
        return {pid: count / total for pid, count in wins.items()}

    def get_avg_game_length(self) -> float:
        if not self.game_results:
            return 0.0
        return sum(r.turns_played for r in self.game_results) / len(self.game_results)

    def get_agent_comparison(self) -> dict[int, dict]:
        comparison: dict[int, dict] = {}
        for pid in self.agent_names:
            wins = sum(1 for r in self.game_results if r.winner == pid)
            avg_score = 0.0
            if self.game_results:
                scores = [r.scores.get(pid, 0) for r in self.game_results]
                avg_score = sum(scores) / len(scores)
            comparison[pid] = {
                "name": self.agent_names.get(pid, f"Player {pid}"),
                "wins": wins,
                "win_rate": wins / max(1, len(self.game_results)),
                "avg_score": avg_score,
            }
        return comparison

    def summary(self) -> str:
        lines = [f"Games played: {len(self.game_results)}"]
        lines.append(f"Avg game length: {self.get_avg_game_length():.1f} turns")
        for pid, info in self.get_agent_comparison().items():
            lines.append(
                f"  {info['name']}: {info['wins']} wins "
                f"({info['win_rate']:.1%}), avg score {info['avg_score']:.1f}"
            )
        return "\n".join(lines)


@dataclass
class TurnLog:
    turn: int
    player_id: int
    action: str
    result: str


@dataclass
class ReplayLogger:
    logs: list[TurnLog] = field(default_factory=list)

    def log_turn(self, turn: int, player_id: int, action: str, result: str) -> None:
        self.logs.append(TurnLog(turn=turn, player_id=player_id, action=action, result=result))

    def save_replay(self, filepath: str) -> None:
        data = [{"turn": l.turn, "player": l.player_id,
                 "action": l.action, "result": l.result} for l in self.logs]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_replay(cls, filepath: str) -> list[TurnLog]:
        with open(filepath) as f:
            data = json.load(f)
        return [TurnLog(**entry) for entry in data]


class GameRunner:
    def __init__(self, agent_factories: list[type], num_players: int = 3):
        self.agent_factories = agent_factories
        self.num_players = num_players
        self.metrics = MetricsTracker()

    def _create_agents(self, seed: Optional[int] = None) -> list[Agent]:
        agents = []
        for pid in range(self.num_players):
            factory = self.agent_factories[pid % len(self.agent_factories)]
            agent = factory(player_id=pid, seed=seed + pid if seed else None)
            agents.append(agent)
            self.metrics.agent_names[pid] = type(agent).__name__
        return agents

    def run_single_game(self, seed: Optional[int] = None) -> GameResult:
        agents = self._create_agents(seed)
        env = Environment(agents, num_players=self.num_players)
        result = env.run_game()
        self.metrics.record_game(result)
        return result

    def run_tournament(self, n_games: int, seed: Optional[int] = None) -> TournamentResult:
        results: list[GameResult] = []
        start = time.time()

        for i in range(n_games):
            game_seed = (seed + i * 1000) if seed else None
            result = self.run_single_game(game_seed)
            results.append(result)

        elapsed = time.time() - start
        win_counts: dict[str, int] = {}
        for r in results:
            if r.winner is not None:
                name = self.metrics.agent_names.get(r.winner, f"Player {r.winner}")
                win_counts[name] = win_counts.get(name, 0) + 1

        avg_length = sum(r.turns_played for r in results) / max(1, len(results))

        return TournamentResult(
            results=results,
            win_counts=win_counts,
            avg_game_length=avg_length,
            total_time=elapsed,
        )
