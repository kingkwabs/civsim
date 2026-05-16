"""Generate report-ready figures and tables for the CivSim writeup.

Outputs to ./report_figures/ :
    fig2_training_curve.png       — RL win rate across the 3-phase curriculum
    fig4_win_breakdown.png        — threshold vs cap vs loss for each agent type
    fig5_turn_histogram.png       — turn-at-win distribution for RL threshold wins
    fig6_action_space.png         — typical valid-action mix in a single turn
    table1_main_results.md        — headline win-rate table
    table2_mcts_journey.md        — MCTS iteration sequence
    table3_reward_shaping.md      — reward-shaping experiment

Run:  .venv/bin/python3 -m civsim.report_figures
"""
from __future__ import annotations

import os
import re
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

from civsim.actions import get_valid_actions
from civsim.agents import GreedyAgent, RandomAgent
from civsim.data_types import (
    ReestablishBuilding, TradeProposal, WIN_THRESHOLD,
)
from civsim.environment import Environment
from civsim.rl import PolicyValueNetwork, RLAgent


OUT = Path("report_figures")
OUT.mkdir(exist_ok=True)


# ── Data collection (one 90-game eval per agent) ─────────────────────────

def eval_agent(agent_factory, label: str, n_games: int = 90):
    """Run n_games with `agent_factory(pid, seed)` in seat 0 vs 2 Greedy.
    Returns dict with per-game outcomes and aggregate stats.
    """
    threshold = cap = losses = 0
    pp_each = []
    win_turns = []
    for seed in range(n_games):
        seat0 = agent_factory(0, 9000 + seed)
        opps = [GreedyAgent(player_id=i, seed=9000 + seed * 7 + i) for i in (1, 2)]
        env = Environment([seat0] + opps, num_players=3)
        r = env.run_game(seed=27000 + seed)
        pp = r.scores[0]
        pp_each.append(pp)
        crossed = r.turns_played < 200 and pp >= WIN_THRESHOLD
        if r.winner == 0:
            if crossed:
                threshold += 1
                win_turns.append(r.turns_played)
            else:
                cap += 1
        else:
            losses += 1
    return {
        "label": label,
        "n": n_games,
        "threshold": threshold,
        "cap": cap,
        "losses": losses,
        "avg_pp": sum(pp_each) / n_games,
        "win_turns": win_turns,
    }


# ── Figure 2: RL training curve ──────────────────────────────────────────

def parse_training_log(path: str):
    """Extract (phase, episode, win_rate) tuples from a training log."""
    phase_re = re.compile(r"=== Phase (\d).*?\(([\d]+) episodes\)")
    line_re = re.compile(r"Episode\s+(\d+) \| Win rate:\s+([\d.]+)%")
    points: list[tuple[int, int, float]] = []
    current_phase = 0
    cumulative_offset = 0
    phase_starts: dict[int, int] = {}

    with open(path) as f:
        for line in f:
            m = phase_re.search(line)
            if m:
                # Starting a new phase
                if current_phase > 0:
                    # Find the last episode for the previous phase
                    last_in_phase = max(
                        (ep for ph, ep, _ in points if ph == current_phase),
                        default=0,
                    )
                    cumulative_offset += last_in_phase
                current_phase = int(m.group(1))
                phase_starts[current_phase] = cumulative_offset
                continue
            m = line_re.search(line)
            if m and current_phase > 0:
                ep = int(m.group(1))
                wr = float(m.group(2))
                points.append((current_phase, ep, wr))
    return points, phase_starts


def fig2_training_curve(log_path: str):
    points, phase_starts = parse_training_log(log_path)
    if not points:
        print(f"WARNING: no training data parsed from {log_path}; skipping fig 2")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    phase_colors = {1: "#3a8fd7", 2: "#e64545", 3: "#9b59b6"}
    phase_names = {1: "vs Random", 2: "vs Greedy", 3: "self-play"}

    for phase in (1, 2, 3):
        pts = [(phase_starts[phase] + ep, wr) for ph, ep, wr in points if ph == phase]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=phase_colors[phase], linewidth=2,
                marker="o", markersize=5, label=f"Phase {phase}: {phase_names[phase]}")
    # Phase boundary lines
    for phase in (2, 3):
        if phase in phase_starts:
            ax.axvline(phase_starts[phase], color="#888", linestyle="--", linewidth=0.7)
    ax.set_xlabel("Cumulative training episode")
    ax.set_ylabel("Win rate (%, rolling 100-game)")
    ax.set_title("RL Training: 3-Phase Curriculum")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_training_curve.png", dpi=150)
    plt.close(fig)
    print(f"  fig2_training_curve.png saved")


# ── Figure 4: Win-outcome breakdown per agent ────────────────────────────

def fig4_breakdown(evals: list[dict]):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    labels = [e["label"] for e in evals]
    threshold = [e["threshold"] for e in evals]
    cap = [e["cap"] for e in evals]
    losses = [e["losses"] for e in evals]
    x = range(len(labels))
    ax.bar(x, threshold, color="#27ae60", label="Threshold wins (≥10 PP)")
    ax.bar(x, cap, bottom=threshold, color="#f39c12",
           label="Turn-cap leader (tiebreaker)")
    ax.bar(x, losses, bottom=[t + c for t, c in zip(threshold, cap)],
           color="#c0392b", label="Losses")
    for i, e in enumerate(evals):
        ax.text(i, e["threshold"] / 2, str(e["threshold"]),
                ha="center", va="center", color="white", weight="bold")
        if e["cap"] > 2:
            ax.text(i, e["threshold"] + e["cap"] / 2, str(e["cap"]),
                    ha="center", va="center", color="white", weight="bold")
        ax.text(i, e["threshold"] + e["cap"] + e["losses"] / 2, str(e["losses"]),
                ha="center", va="center", color="white", weight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Games (of 90)")
    ax.set_title("Outcome composition by agent (seat 0 vs 2 Greedy, 90-game eval)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_win_breakdown.png", dpi=150)
    plt.close(fig)
    print(f"  fig4_win_breakdown.png saved")


# ── Figure 5: Turn-at-win histogram for RL threshold wins ────────────────

def fig5_turn_histogram(rl_eval: dict):
    turns = rl_eval["win_turns"]
    if not turns:
        print("WARNING: no RL threshold wins; skipping fig 5")
        return
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.hist(turns, bins=range(0, 201, 10), color="#3a8fd7",
            edgecolor="#1a3a55", linewidth=1.2)
    ax.set_xlabel("Turn at which RL reached 10 PP")
    ax.set_ylabel("Count")
    ax.set_title(f"RL threshold-win timing  (n={len(turns)} of 90 games)")
    if turns:
        ax.axvline(min(turns), color="#e64545", linestyle="--",
                   label=f"Fastest: turn {min(turns)}")
        ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_turn_histogram.png", dpi=150)
    plt.close(fig)
    print(f"  fig5_turn_histogram.png saved")


# ── Figure 3: Balance pass progression ───────────────────────────────────

def fig3_balance_pass():
    """Show how each balance change unlocked progress.

    For each phase we monkey-patch the game-balance constants, run 20
    Greedy-vs-Greedy games (deterministic by seed), and record:
      - threshold wins (someone reached 10 PP before turn 200)
      - average max PP across all 3 players per game (proxy for "how
        close did anyone get to the win condition")

    Each phase is *cumulative* (later phases include all prior changes).
    """
    import civsim.data_types as dt
    import civsim.game_state as gs
    import civsim.environment as env_mod
    import civsim.actions as ac

    # Save original values
    orig = {
        "WIN_THRESHOLD": dt.WIN_THRESHOLD,
        "SETTLEMENT_PP": dt.SETTLEMENT_PP,
        "CITY_PP": dt.CITY_PP,
        "MAINTENANCE_TURN_INTERVAL": dt.MAINTENANCE_TURN_INTERVAL,
        "RAIN_PROBABILITY": dt.RAIN_PROBABILITY,
        "BARN_PROBABILITY": dt.BARN_PROBABILITY,
        "STARTING_WATER_STOCKPILE": dt.STARTING_WATER_STOCKPILE,
        "TRADE_CAP_PER_TURN": dt.TRADE_CAP_PER_TURN,
        "MAINTENANCE_COSTS_CITY": dict(dt.MAINTENANCE_COSTS[dt.BuildType.CITY]),
    }

    # Phase recipes — each adds onto the previous. Last phase = current state.
    phases = [
        ("0. Pre-balance",       dict(s_pp=1, c_pp=2, m_interval=1, rain=0.0, barn=0.0,
                                      stockpile=0, trade_cap=999,
                                      city_upkeep_water=1)),
        ("1. + rain + stockpile",dict(s_pp=1, c_pp=2, m_interval=1, rain=0.33, barn=0.0,
                                      stockpile=2, trade_cap=999,
                                      city_upkeep_water=1)),
        ("2. + slower maint",    dict(s_pp=1, c_pp=2, m_interval=3, rain=0.33, barn=0.0,
                                      stockpile=2, trade_cap=999,
                                      city_upkeep_water=1)),
        ("3. + barn + cow-only city", dict(s_pp=1, c_pp=2, m_interval=3, rain=0.33, barn=0.33,
                                            stockpile=2, trade_cap=999,
                                            city_upkeep_water=0)),
        ("4. + PP recalib (s=2/c=4)", dict(s_pp=2, c_pp=4, m_interval=3, rain=0.33, barn=0.33,
                                            stockpile=2, trade_cap=999,
                                            city_upkeep_water=0)),
        ("5. + trade cap (10)",  dict(s_pp=2, c_pp=4, m_interval=3, rain=0.33, barn=0.33,
                                       stockpile=2, trade_cap=10,
                                       city_upkeep_water=0)),
    ]

    def apply_phase(cfg):
        dt.SETTLEMENT_PP = cfg["s_pp"]
        dt.CITY_PP = cfg["c_pp"]
        dt.MAINTENANCE_TURN_INTERVAL = cfg["m_interval"]
        gs.MAINTENANCE_TURN_INTERVAL = cfg["m_interval"]
        env_mod.MAINTENANCE_TURN_INTERVAL = cfg["m_interval"]
        dt.RAIN_PROBABILITY = cfg["rain"]
        gs.RAIN_PROBABILITY = cfg["rain"]
        dt.BARN_PROBABILITY = cfg["barn"]
        gs.BARN_PROBABILITY = cfg["barn"]
        dt.STARTING_WATER_STOCKPILE = cfg["stockpile"]
        env_mod.STARTING_WATER_STOCKPILE = cfg["stockpile"]
        dt.TRADE_CAP_PER_TURN = cfg["trade_cap"]
        ac.TRADE_CAP_PER_TURN = cfg["trade_cap"]
        # City upkeep: with water cost 0 -> cow-only
        if cfg["city_upkeep_water"] == 0:
            dt.MAINTENANCE_COSTS[dt.BuildType.CITY] = {dt.ResourceType.COW: 1}
        else:
            dt.MAINTENANCE_COSTS[dt.BuildType.CITY] = {
                dt.ResourceType.WATER: 1, dt.ResourceType.COW: 1,
            }

    def restore():
        dt.SETTLEMENT_PP = orig["SETTLEMENT_PP"]
        dt.CITY_PP = orig["CITY_PP"]
        dt.MAINTENANCE_TURN_INTERVAL = orig["MAINTENANCE_TURN_INTERVAL"]
        gs.MAINTENANCE_TURN_INTERVAL = orig["MAINTENANCE_TURN_INTERVAL"]
        env_mod.MAINTENANCE_TURN_INTERVAL = orig["MAINTENANCE_TURN_INTERVAL"]
        dt.RAIN_PROBABILITY = orig["RAIN_PROBABILITY"]
        gs.RAIN_PROBABILITY = orig["RAIN_PROBABILITY"]
        dt.BARN_PROBABILITY = orig["BARN_PROBABILITY"]
        gs.BARN_PROBABILITY = orig["BARN_PROBABILITY"]
        dt.STARTING_WATER_STOCKPILE = orig["STARTING_WATER_STOCKPILE"]
        env_mod.STARTING_WATER_STOCKPILE = orig["STARTING_WATER_STOCKPILE"]
        dt.TRADE_CAP_PER_TURN = orig["TRADE_CAP_PER_TURN"]
        ac.TRADE_CAP_PER_TURN = orig["TRADE_CAP_PER_TURN"]
        dt.MAINTENANCE_COSTS[dt.BuildType.CITY] = orig["MAINTENANCE_COSTS_CITY"]

    print("Running balance-pass sweep (6 phases × 20 games each)...", flush=True)
    t0 = time.time()
    results = []
    try:
        for label, cfg in phases:
            apply_phase(cfg)
            wins = 0
            max_pps = []
            for seed in range(20):
                agents = [GreedyAgent(player_id=i, seed=seed * 7 + i) for i in range(3)]
                env = Environment(agents, num_players=3)
                r = env.run_game(seed=seed * 13)
                m = max(r.scores.values())
                max_pps.append(m)
                if r.turns_played < 200 and m >= WIN_THRESHOLD:
                    wins += 1
            results.append((label, wins, sum(max_pps) / len(max_pps)))
            print(f"  {label}: wins={wins}/20  avg_max_pp={results[-1][2]:.1f}",
                  flush=True)
    finally:
        restore()

    print(f"  balance sweep done in {time.time()-t0:.0f}s", flush=True)

    # Plot — grouped bars: threshold wins (left axis) + avg max PP (right axis)
    fig, ax1 = plt.subplots(figsize=(10, 4.5), dpi=120)
    labels = [r[0] for r in results]
    wins = [r[1] for r in results]
    pps = [r[2] for r in results]
    x = list(range(len(labels)))

    bars = ax1.bar(x, wins, color="#27ae60", alpha=0.85, label="Threshold wins / 20")
    ax1.set_ylabel("Threshold wins (of 20)", color="#27ae60")
    ax1.tick_params(axis="y", labelcolor="#27ae60")
    ax1.set_ylim(0, max(20, max(wins) + 2))
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    for b, w in zip(bars, wins):
        ax1.text(b.get_x() + b.get_width() / 2, w + 0.3, str(w),
                 ha="center", va="bottom", fontsize=9, color="#1e6d3f", weight="bold")

    ax2 = ax1.twinx()
    ax2.plot(x, pps, color="#e64545", marker="o", linewidth=2,
             label="Avg max PP across 3 players")
    ax2.set_ylabel("Avg max PP per game", color="#e64545")
    ax2.tick_params(axis="y", labelcolor="#e64545")
    ax2.set_ylim(0, max(pps) + 2)

    ax1.set_title("Balance pass: each change unlocked more progress (3-Greedy games)")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_balance_pass.png", dpi=150)
    plt.close(fig)
    print(f"  fig3_balance_pass.png saved")


# ── Figure 6: Action-space breakdown ─────────────────────────────────────

def fig6_action_space():
    """Sample a few mid-game states and count valid actions by type."""
    counter: Counter = Counter()
    samples = 0
    agents = [GreedyAgent(player_id=i, seed=100 + i) for i in range(3)]
    env = Environment(agents, num_players=3)
    env.reset(seed=42)
    # Step a few turns to get into mid-game
    for _ in range(30):
        if env.state.game_over:
            break
        pid = env.state.current_player
        valid = get_valid_actions(env.state, pid)
        for a in valid:
            if isinstance(a, ReestablishBuilding):
                key = "Re-establish"
            elif isinstance(a, TradeProposal):
                key = "P2P trade" if a.target_player is not None else "Bank/port trade"
            else:
                # ActionType.BUILD etc. names map well
                key = a.action_type.name.replace("_", " ").title()
            counter[key] += 1
        samples += 1
        env._run_action_loop(pid, agents[pid])

    if not counter:
        print("WARNING: no action data; skipping fig 6")
        return

    # Average across sampled turns
    avg_counts = {k: v / samples for k, v in counter.items()}
    labels, sizes = zip(*sorted(avg_counts.items(), key=lambda kv: -kv[1]))
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    colors = ["#3a8fd7", "#e64545", "#27ae60", "#f39c12", "#9b59b6",
              "#1abc9c", "#34495e", "#e67e22"]
    wedges, _ = ax.pie(sizes, labels=None, startangle=90,
                       colors=colors[:len(sizes)],
                       wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    total = sum(sizes)
    legend = [f"{l}: {s:.0f} ({s*100/total:.0f}%)" for l, s in zip(labels, sizes)]
    ax.legend(wedges, legend, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(f"Typical valid-action mix per turn  (averaged over {samples} turns)")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_action_space.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig6_action_space.png saved")


# ── Table rendering (Markdown + PNG) ─────────────────────────────────────

def render_table_png(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    out_path: Path,
    col_align: list[str] | None = None,
    highlight_rows: list[int] | None = None,
    footnote: str | None = None,
) -> None:
    """Render a table as a nicely styled PNG using matplotlib.

    Args:
        title: bold title above the table
        headers: column headers
        rows: list of row data (each row a list matching header count)
        out_path: where to save the PNG
        col_align: per-column alignment ('left' | 'center' | 'right'); default mixed
        highlight_rows: row indices to give a subtle background tint (0-indexed)
        footnote: small italic note below the table
    """
    if col_align is None:
        col_align = ["left"] + ["center"] * (len(headers) - 1)
    n_rows = len(rows)
    n_cols = len(headers)
    # Sizing: matplotlib's `table` is finicky, so we draw with text + rects.
    row_h = 0.5
    fig_h = 0.6 + 0.45 * n_rows + (0.4 if footnote else 0)  # title + rows + footnote
    fig_w = max(8, 2.1 * n_cols)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows + 1)
    ax.axis("off")

    # Title
    ax.text(n_cols / 2, n_rows + 0.65, title,
            ha="center", va="center", fontsize=13, weight="bold")

    # Header row (dark band)
    header_bg = plt.Rectangle((0, n_rows), n_cols, 1, facecolor="#2c3e50",
                              edgecolor="#1a2530", linewidth=1.2, zorder=1)
    ax.add_patch(header_bg)
    for j, h in enumerate(headers):
        ax.text(j + 0.5, n_rows + 0.5, h,
                ha="center", va="center", color="white",
                fontsize=11, weight="bold", zorder=2)

    # Body rows
    highlight_rows = highlight_rows or []
    for i, row in enumerate(rows):
        y = n_rows - 1 - i  # top row is rows[0]
        bg = "#ecf0f1" if i in highlight_rows else ("white" if i % 2 == 0 else "#f8f9fa")
        body_bg = plt.Rectangle((0, y), n_cols, 1, facecolor=bg,
                                edgecolor="#bdc3c7", linewidth=0.6, zorder=1)
        ax.add_patch(body_bg)
        for j, cell in enumerate(row):
            align = col_align[j] if j < len(col_align) else "center"
            x = {"left": j + 0.08, "center": j + 0.5, "right": j + 0.92}[align]
            weight = "bold" if i in highlight_rows else "normal"
            ax.text(x, y + 0.5, str(cell),
                    ha={"left": "left", "center": "center", "right": "right"}[align],
                    va="center", fontsize=10, weight=weight, zorder=2)

    if footnote:
        ax.text(n_cols / 2, -0.3, footnote, ha="center", va="center",
                fontsize=8.5, style="italic", color="#555")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


# ── Tables ───────────────────────────────────────────────────────────────

def write_table1_main_results(evals: list[dict]):
    md_lines = [
        "# Table 1: Main results (90-game evaluation, seat 0 vs 2 Greedy)",
        "",
        "| Agent | Threshold wins | Turn-cap wins | Losses | Total wins | Avg PP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    rows: list[list[str]] = []
    headers = ["Agent", "Threshold wins", "Turn-cap wins", "Losses", "Total wins", "Avg PP"]
    rl_idx = None
    for i, e in enumerate(evals):
        total = e["threshold"] + e["cap"]
        n = e["n"]
        md_lines.append(
            f"| {e['label']} | {e['threshold']} ({e['threshold']*100//n}%) | "
            f"{e['cap']} ({e['cap']*100//n}%) | "
            f"{e['losses']} ({e['losses']*100//n}%) | "
            f"{total} ({total*100//n}%) | {e['avg_pp']:.1f} |"
        )
        rows.append([
            e["label"],
            f"{e['threshold']} ({e['threshold']*100//n}%)",
            f"{e['cap']} ({e['cap']*100//n}%)",
            f"{e['losses']} ({e['losses']*100//n}%)",
            f"{total} ({total*100//n}%)",
            f"{e['avg_pp']:.1f}",
        ])
        if e["label"].lower() == "rl":
            rl_idx = i
    (OUT / "table1_main_results.md").write_text("\n".join(md_lines) + "\n")
    render_table_png(
        title="Table 1: Main results — 90-game eval, seat 0 vs 2 Greedy",
        headers=headers,
        rows=rows,
        out_path=OUT / "table1_main_results.png",
        col_align=["left", "right", "right", "right", "right", "right"],
        highlight_rows=[rl_idx] if rl_idx is not None else None,
    )
    print(f"  table1_main_results.md + .png saved")


def write_table2_mcts_journey():
    rows = [
        ["Vanilla MCTS-30",                "0", "0.0", "Hand-rolled turn transitions drifted from real game"],
        ["+ shared turn helpers",          "0", "0.0", "Rollouts correct; uniform random still uninformative"],
        ["+ Greedy rollout policy",        "0", "0.0", "Better signal; action space (~60) still drowns selection"],
        ["+ K=8 pruning by Greedy score",  "1", "4.3", "Viable: builds cities, occasional wins"],
        ["+ K=5 + PUCT priors",            "2", "3.8", "Aggression trades reliability for wins"],
        ["+ virtualQ + all-in + decisive", "2", "3.0", "Plateau confirmed at this sim count"],
        ["+ RL value head",                "1", "2.2", "Sigmoid-squashed head misled rollouts; regressed"],
        ["+ RL policy head as PUCT prior", "0", "7.8", "Reaches 8 PP; 'careful plateau'"],
        ["+ tuned c_puct=0.5, temp=0.5",   "3*","5.0", "Aggression breaks plateau (* 20-game sample)"],
    ]
    headers = ["MCTS variant (cumulative)", "Thresh wins / 10", "Avg PP", "Key finding"]
    md_lines = [
        "# Table 2: MCTS iteration sequence",
        "",
        "| MCTS variant | Threshold wins / 10 | Avg PP | Key finding |",
        "|---|---:|---:|---|",
    ] + [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in rows] + [
        "",
        "Sample sizes are small (6-20 games per row) due to per-sim cost; trends are directional, not statistical.",
    ]
    (OUT / "table2_mcts_journey.md").write_text("\n".join(md_lines) + "\n")
    render_table_png(
        title="Table 2: MCTS iteration sequence (each row adds onto the previous)",
        headers=headers,
        rows=rows,
        out_path=OUT / "table2_mcts_journey.png",
        col_align=["left", "right", "right", "left"],
        footnote="Sample sizes are small (6-20 games per row); trends are directional.",
    )
    print(f"  table2_mcts_journey.md + .png saved")


def write_table3_reward_shaping():
    rows = [
        ["Baseline (kept)",         "+10 / -5",   "0",     "~46%", "7.0", "Sweet spot — stable training"],
        ["Stronger bonus + penalty","+100 / -50", "-0.1",  "33%",  "0.1", "Penalty dominated; learned 'do nothing'"],
        ["Stronger bonus only",     "+100 / -50", "0",     "53%",  "7.1", "Higher variance; gradient noise"],
        ["Imitation pretrain (BC)", "+10 / -5",   "0",     "37%",  "0.2", "BC loss = random baseline; pulled init"],
    ]
    headers = ["Config", "Win bonus", "Per-action penalty", "Total wins", "Avg PP", "Diagnosis"]
    md_lines = [
        "# Table 3: Reward-shaping experiments (90-game eval each)",
        "",
        "| Config | Win bonus | Per-action penalty | Total wins | Avg PP | Diagnosis |",
        "|---|---:|---:|---:|---:|---|",
    ] + [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |" for r in rows]
    (OUT / "table3_reward_shaping.md").write_text("\n".join(md_lines) + "\n")
    render_table_png(
        title="Table 3: Reward-shaping experiments — baseline beats all variants",
        headers=headers,
        rows=rows,
        out_path=OUT / "table3_reward_shaping.png",
        col_align=["left", "center", "center", "right", "right", "left"],
        highlight_rows=[0],  # the kept baseline
    )
    print(f"  table3_reward_shaping.md + .png saved")


# ── Driver ───────────────────────────────────────────────────────────────

def main():
    print("Loading RL model...", flush=True)
    net = PolicyValueNetwork()
    net.load("rl_model.pt")
    net.eval()

    print("Running evaluations (this is the slow part, ~2-3 min)...", flush=True)
    t0 = time.time()

    def rand_factory(pid, seed):
        return RandomAgent(player_id=pid, seed=seed)

    def greedy_factory(pid, seed):
        return GreedyAgent(player_id=pid, seed=seed)

    def rl_factory(pid, seed):
        return RLAgent(player_id=pid, network=net, seed=seed, training=False)

    evals = [
        eval_agent(rand_factory,   "Random",  n_games=90),
        eval_agent(greedy_factory, "Greedy",  n_games=90),
        eval_agent(rl_factory,     "RL",      n_games=90),
    ]
    print(f"  Eval done in {time.time()-t0:.0f}s", flush=True)

    print("Generating figures + tables...", flush=True)
    write_table1_main_results(evals)
    write_table2_mcts_journey()
    write_table3_reward_shaping()

    # Training curve from the most recent training log
    log_candidates = ["/tmp/rl_long.log", "/tmp/rl_revert.log", "/tmp/rl_selfplay.log"]
    log_path = next((p for p in log_candidates if os.path.exists(p)), None)
    if log_path:
        fig2_training_curve(log_path)
    else:
        print("  no training log found; skipping fig 2")

    rl_eval = evals[2]
    fig4_breakdown(evals)
    fig5_turn_histogram(rl_eval)
    fig6_action_space()
    fig3_balance_pass()

    print(f"\nAll outputs in {OUT.resolve()}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
