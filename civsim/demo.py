"""Play a single rendered CivSim game in the terminal.

Examples:
    # 3 GreedyAgents at slow speed (good for reading along)
    python -m civsim.demo

    # Mix: random vs greedy vs trained RL
    python -m civsim.demo --agents random,greedy,rl

    # Three trained RL agents (showcase the headline result)
    python -m civsim.demo --agents rl,rl,rl

    # Cinematic mode: clear screen between frames
    python -m civsim.demo --agents greedy,greedy,rl --clear --pause 0.3

    # Silent run for a quick screen capture, no color
    python -m civsim.demo --no-color --pause 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from .agents import GreedyAgent, MCTSAgent, RandomAgent
from .environment import Environment
from .visualization import TerminalRenderer, render_final_stats


def _build_factories(
    spec: str, rl_model_path: str, mcts_sims: int,
) -> list[Callable[[int, int], object]]:
    """Translate an --agents string like 'random,greedy,rl' into a list of
    factories that take (player_id, seed) and return the constructed agent."""
    rl_network = None  # loaded lazily, only if needed

    def make_random(pid, seed):
        return RandomAgent(player_id=pid, seed=seed)

    def make_greedy(pid, seed):
        return GreedyAgent(player_id=pid, seed=seed)

    def make_mcts(pid, seed):
        return MCTSAgent(player_id=pid, seed=seed, n_simulations=mcts_sims)

    def make_rl(pid, seed):
        nonlocal rl_network
        if rl_network is None:
            from .rl import PolicyValueNetwork, RLAgent  # local import: torch is optional
            if not Path(rl_model_path).exists():
                raise SystemExit(
                    f"RL model not found at {rl_model_path!r}. "
                    f"Train one first: python -m civsim.rl_train --episodes-random 300 "
                    f"--episodes-greedy 200 --episodes-selfplay 300"
                )
            rl_network = PolicyValueNetwork()
            rl_network.load(rl_model_path)
            rl_network.eval()
        from .rl import RLAgent
        return RLAgent(player_id=pid, network=rl_network, seed=seed, training=False)

    REG = {
        "random": make_random,
        "greedy": make_greedy,
        "mcts":   make_mcts,
        "rl":     make_rl,
    }

    names = [s.strip().lower() for s in spec.split(",") if s.strip()]
    try:
        return [REG[n] for n in names]
    except KeyError as e:
        raise SystemExit(
            f"Unknown agent: {e.args[0]!r}. Choices: {sorted(REG)}"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", default="greedy,greedy,greedy",
                   help="Comma-separated agent types: random | greedy | mcts | rl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pause", type=float, default=0.1,
                   help="Seconds between rendered frames (0 = fastest).")
    p.add_argument("--no-color", action="store_true",
                   help="Disable ANSI colors.")
    p.add_argument("--clear", action="store_true",
                   help="Clear screen between frames (animation-like).")
    p.add_argument("--no-buildings", action="store_true",
                   help="Hide the buildings/roads listing.")
    p.add_argument("--rl-model", default="rl_model.pt",
                   help="Path to a trained RL model (used by 'rl' agents).")
    p.add_argument("--mcts-sims", type=int, default=50,
                   help="n_simulations for any 'mcts' agent.")
    p.add_argument("--gui", action="store_true",
                   help="Use the matplotlib GUI renderer instead of terminal.")
    args = p.parse_args(argv)

    factories = _build_factories(args.agents, args.rl_model, args.mcts_sims)
    agents = [f(pid=i, seed=args.seed + i) for i, f in enumerate(factories)]

    if args.gui:
        from .visualization import GUIRenderer
        renderer = GUIRenderer(agents=agents, pause_per_frame=args.pause)
    else:
        renderer = TerminalRenderer(
            use_color=not args.no_color,
            clear_screen=args.clear,
            pause_per_action=args.pause,
            show_buildings=not args.no_buildings,
        )

    print(f"Agents: {args.agents}  seed={args.seed}  pause={args.pause}s", flush=True)
    env = Environment(agents, num_players=len(agents), renderer=renderer)
    result = env.run_game(seed=args.seed)

    print()
    print(render_final_stats(result, use_color=not args.no_color))

    # Keep GUI window open after game ends so the final board state is
    # visible for screen recording. User closes the window to exit.
    if args.gui:
        try:
            import matplotlib.pyplot as plt
            print("\n(GUI: close the window to exit)")
            plt.show(block=True)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
