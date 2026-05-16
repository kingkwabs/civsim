"""Play a single rendered CivSim game in the terminal.

Usage:
    python -m civsim.demo                # 3 GreedyAgents, color, slow
    python -m civsim.demo --agents random,greedy,mcts
    python -m civsim.demo --no-color --pause 0
"""
from __future__ import annotations

import argparse
import sys

from .agents import GreedyAgent, MCTSAgent, RandomAgent
from .environment import Environment
from .visualization import TerminalRenderer, render_final_stats


AGENT_REGISTRY = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "mcts": MCTSAgent,
}


def _parse_agents(spec: str) -> list[type]:
    names = [s.strip().lower() for s in spec.split(",") if s.strip()]
    try:
        return [AGENT_REGISTRY[n] for n in names]
    except KeyError as e:
        raise SystemExit(f"Unknown agent: {e.args[0]}. "
                         f"Choices: {sorted(AGENT_REGISTRY)}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", default="greedy,greedy,greedy",
                   help="Comma-separated agent types per seat.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pause", type=float, default=0.1,
                   help="Seconds between rendered frames.")
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--clear", action="store_true",
                   help="Clear screen between frames.")
    p.add_argument("--no-buildings", action="store_true",
                   help="Hide the buildings/roads listing.")
    args = p.parse_args(argv)

    agent_classes = _parse_agents(args.agents)
    agents = [cls(player_id=i, seed=args.seed + i)
              for i, cls in enumerate(agent_classes)]

    renderer = TerminalRenderer(
        use_color=not args.no_color,
        clear_screen=args.clear,
        pause_per_action=args.pause,
        show_buildings=not args.no_buildings,
    )

    env = Environment(agents, num_players=len(agents), renderer=renderer)
    result = env.run_game()

    print()
    print(render_final_stats(result, use_color=not args.no_color))
    return 0


if __name__ == "__main__":
    sys.exit(main())
