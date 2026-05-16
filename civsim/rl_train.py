"""End-to-end RL training driver for CivSim.

Curriculum:
    1. Phase 1 — train vs RandomAgents
    2. Phase 2 — train vs GreedyAgents
    3. Phase 3 (optional) — self-play with snapshot pool
    4. Eval — N games of trained policy vs 2 GreedyAgents

Self-play uses a rotating pool of frozen network snapshots so the policy
trains against its own past selves. Mixing Greedy in keeps the policy
from collapsing to a brittle equilibrium.

Usage:
    python -m civsim.rl_train --episodes-random 200 --episodes-greedy 100 --episodes-selfplay 200
    python -m civsim.rl_train --smoke
"""
from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from pathlib import Path

from .agents import GreedyAgent, RandomAgent
from .environment import Environment
from .rl import PolicyValueNetwork, RLAgent, RLTrainer, bc_train, collect_demonstrations


def make_selfplay_factory(
    current_network,
    snapshots: list,
    greedy_fraction: float = 0.3,
):
    """Build an opponent_factory for self-play.

    With probability `greedy_fraction` the opponent slot is a GreedyAgent;
    otherwise it's an RLAgent pointing at a frozen snapshot. Snapshots are
    a list of PolicyValueNetwork instances (frozen — never gradient-updated).

    The current network is excluded as an opponent (only past versions are).
    """
    rng = random.Random(0)

    def factory(episode_idx: int, seed: int) -> list:
        # Per-episode RNG keeps mixes reproducible across runs
        rng_local = random.Random(seed)
        opponents = []
        for i in (1, 2):
            opp_seed = seed * 7 + i
            if not snapshots or rng_local.random() < greedy_fraction:
                opponents.append(GreedyAgent(player_id=i, seed=opp_seed))
            else:
                snap = rng_local.choice(snapshots)
                opponents.append(RLAgent(
                    player_id=i, network=snap, seed=opp_seed,
                    training=False,  # frozen, no gradients
                ))
        return opponents

    return factory


def selfplay_phase(
    network,
    n_episodes: int,
    snapshot_interval: int = 25,
    max_snapshots: int = 5,
    greedy_fraction: float = 0.3,
    print_every: int = 25,
    lr: float = 3e-4,
    gamma: float = 0.99,
):
    """Run a self-play training phase.

    Every `snapshot_interval` episodes, freeze a copy of the current network
    and add it to the opponent pool. Cap the pool at `max_snapshots` (drop
    oldest first) so memory stays bounded.
    """
    snapshots: list = []

    def factory_with_snapshots(episode_idx, seed):
        return make_selfplay_factory(network, snapshots, greedy_fraction)(episode_idx, seed)

    trainer = RLTrainer(network, num_players=3, lr=lr, gamma=gamma,
                        opponent_factory=factory_with_snapshots)

    chunk = max(1, snapshot_interval)
    for start in range(0, n_episodes, chunk):
        n_this = min(chunk, n_episodes - start)
        trainer.train(n_episodes=n_this, print_every=print_every, seed=2000 + start)
        # Snapshot the current network so future episodes train against it
        snap = copy.deepcopy(network)
        for p in snap.parameters():
            p.requires_grad_(False)
        snap.eval()
        snapshots.append(snap)
        if len(snapshots) > max_snapshots:
            snapshots.pop(0)
        print(f"  [self-play] pool size = {len(snapshots)} (after episode {start + n_this})", flush=True)

    return trainer.stats


def evaluate(network, n_games: int, opponent_cls=GreedyAgent, seed_base: int = 9000) -> dict:
    """Play `n_games` with trained RL agent vs two opponents (no exploration)."""
    wins = ties = losses = 0
    pp_avg = 0.0
    turn_avg = 0.0
    for seed in range(n_games):
        rl = RLAgent(
            player_id=0,
            network=network,
            seed=seed_base + seed,
            training=False,  # greedy / argmax
        )
        opponents = [
            opponent_cls(player_id=i, seed=seed_base + seed * 7 + i)
            for i in (1, 2)
        ]
        env = Environment([rl] + opponents, num_players=3)
        r = env.run_game()
        pp_avg += r.scores[0]
        turn_avg += r.turns_played
        if r.winner == 0:
            wins += 1
        elif r.winner is None:
            ties += 1
        else:
            losses += 1
    return {
        "wins": wins, "ties": ties, "losses": losses,
        "win_rate": wins / n_games,
        "avg_pp": pp_avg / n_games,
        "avg_turns": turn_avg / n_games,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes-random", type=int, default=200,
                   help="Phase 1: episodes vs RandomAgents")
    p.add_argument("--episodes-greedy", type=int, default=150,
                   help="Phase 2: episodes vs GreedyAgents")
    p.add_argument("--episodes-selfplay", type=int, default=0,
                   help="Phase 3: episodes of snapshot-pool self-play")
    p.add_argument("--snapshot-interval", type=int, default=25,
                   help="Snapshot the network every N self-play episodes")
    p.add_argument("--max-snapshots", type=int, default=5,
                   help="Cap snapshot pool size (drop oldest when full)")
    p.add_argument("--greedy-fraction", type=float, default=0.3,
                   help="Probability each self-play opponent slot is Greedy")
    p.add_argument("--eval-games", type=int, default=20,
                   help="How many games for final eval vs Greedy")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--print-every", type=int, default=25)
    p.add_argument("--save-path", default="rl_model.pt")
    p.add_argument("--bc-games", type=int, default=0,
                   help="Phase 0: collect demonstrations from N Greedy-vs-Greedy games "
                        "where someone reached the win threshold, then behavior-clone "
                        "the network on them before phase 1. 0 disables.")
    p.add_argument("--bc-epochs", type=int, default=20,
                   help="Epochs of behavior cloning over the demo set")
    p.add_argument("--bc-lr", type=float, default=1e-3,
                   help="Learning rate for BC pretraining")
    p.add_argument("--smoke", action="store_true",
                   help="Fast sanity check: 10+10+10 episodes, tiny eval")
    args = p.parse_args(argv)

    if args.smoke:
        args.episodes_random = 10
        args.episodes_greedy = 10
        args.episodes_selfplay = 10
        args.snapshot_interval = 5
        args.eval_games = 3
        args.print_every = 5
        if args.bc_games:
            args.bc_games = 20
            args.bc_epochs = 5

    network = PolicyValueNetwork()

    if args.bc_games > 0:
        print(f"=== Phase 0: collecting demonstrations from {args.bc_games} "
              f"Greedy-vs-Greedy games ===", flush=True)
        t0 = time.time()
        demos = collect_demonstrations(n_games=args.bc_games)
        print(f"  Collected {len(demos)} demos in {time.time()-t0:.1f}s", flush=True)
        if demos:
            print(f"=== Phase 0b: behavior cloning "
                  f"({args.bc_epochs} epochs, lr={args.bc_lr}) ===", flush=True)
            t0 = time.time()
            bc_train(network, demos, n_epochs=args.bc_epochs, lr=args.bc_lr)
            print(f"  BC done in {time.time()-t0:.1f}s", flush=True)
        else:
            print(f"  WARNING: 0 threshold-winning games out of {args.bc_games}. "
                  f"Skipping BC.", flush=True)
        print(flush=True)

    print(f"=== Phase 1: training vs Random ({args.episodes_random} episodes) ===", flush=True)
    t0 = time.time()
    trainer = RLTrainer(network, num_players=3, lr=args.lr, gamma=args.gamma,
                         opponent_type=RandomAgent)
    stats1 = trainer.train(n_episodes=args.episodes_random, print_every=args.print_every, seed=1234)
    print(f"Phase 1 done in {time.time()-t0:.1f}s. Final win rate vs Random: "
          f"{stats1.win_rates[-1]:.1%}", flush=True)

    # Save mid-training checkpoint
    network.save(args.save_path.replace(".pt", "_phase1.pt"))

    print(flush=True)
    print(f"=== Phase 2: training vs Greedy ({args.episodes_greedy} episodes) ===", flush=True)
    t0 = time.time()
    trainer2 = RLTrainer(network, num_players=3, lr=args.lr, gamma=args.gamma,
                         opponent_type=GreedyAgent)
    stats2 = trainer2.train(n_episodes=args.episodes_greedy, print_every=args.print_every, seed=5678)
    print(f"Phase 2 done in {time.time()-t0:.1f}s. Final win rate vs Greedy: "
          f"{stats2.win_rates[-1]:.1%}", flush=True)

    network.save(args.save_path.replace(".pt", "_phase2.pt"))

    if args.episodes_selfplay > 0:
        print(flush=True)
        print(f"=== Phase 3: self-play with snapshot pool "
              f"({args.episodes_selfplay} episodes, snapshot every "
              f"{args.snapshot_interval}, greedy_fraction={args.greedy_fraction}) ===",
              flush=True)
        t0 = time.time()
        selfplay_phase(
            network,
            n_episodes=args.episodes_selfplay,
            snapshot_interval=args.snapshot_interval,
            max_snapshots=args.max_snapshots,
            greedy_fraction=args.greedy_fraction,
            print_every=args.print_every,
            lr=args.lr,
            gamma=args.gamma,
        )
        print(f"Phase 3 done in {time.time()-t0:.1f}s", flush=True)

    network.save(args.save_path)
    print(f"\nSaved final model to {args.save_path}", flush=True)

    if args.eval_games > 0:
        print(flush=True)
        print(f"=== Final eval: {args.eval_games} games vs Greedy (greedy/argmax) ===", flush=True)
        results = evaluate(network, args.eval_games, GreedyAgent)
        print(f"  Wins: {results['wins']}/{args.eval_games}  "
              f"({results['win_rate']:.1%})", flush=True)
        print(f"  Avg PP: {results['avg_pp']:.1f}  Avg turns: {results['avg_turns']:.0f}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
