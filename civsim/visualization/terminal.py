"""Terminal-based ASCII rendering for CivSim.

Two renderers:
- render_board: hex grid showing tiles (resource + dice number).
- render_dashboard: per-player resources, dev cards, progress points.

The TerminalRenderer class wraps both and is wired into Environment via
its optional `renderer` parameter — silent by default, opt-in for demos.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from ..data_types import (
    ActionResult, BuildType, DevCardType, GameResult, PlayerStats,
    PortType, ResourceType, WIN_THRESHOLD,
)


# ── ANSI Colors ──────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

_RESOURCE_COLOR = {
    ResourceType.WOOD:  "\033[32m",  # green
    ResourceType.STONE: "\033[90m",  # gray
    ResourceType.METAL: "\033[36m",  # cyan
    ResourceType.WHEAT: "\033[33m",  # yellow
    ResourceType.WATER: "\033[34m",  # blue
    ResourceType.COW:   "\033[97m",  # bright white
}

_RESOURCE_GLYPH = {
    ResourceType.WOOD:  "W",
    ResourceType.STONE: "S",
    ResourceType.METAL: "M",
    ResourceType.WHEAT: "H",
    ResourceType.WATER: "A",
    ResourceType.COW:   "C",
}

_PLAYER_COLOR = [
    "\033[91m",  # bright red
    "\033[94m",  # bright blue
    "\033[93m",  # bright yellow
    "\033[95m",  # bright magenta
    "\033[96m",  # bright cyan
    "\033[92m",  # bright green
]

_BUILD_GLYPH = {
    BuildType.ROAD:       "-",
    BuildType.SETTLEMENT: "s",
    BuildType.CITY:       "C",
}


def _color(s: str, code: str, use_color: bool) -> str:
    return f"{code}{s}{RESET}" if use_color else s


def _player_tag(pid: int, use_color: bool) -> str:
    return _color(f"P{pid}", _PLAYER_COLOR[pid % len(_PLAYER_COLOR)] + BOLD, use_color)


# ── Hex Board Rendering ──────────────────────────────────────────────────

# Tile is rendered as a 7-wide, 3-tall ASCII cell:
#   .-----.
#   | W:8 |
#   '-----'
# Same hex-y row tiles are 8 char-columns apart; alternating rows are
# offset by 4 cols and separated vertically by 4 text rows so the
# staggered layout reads as a hex grid.

TILE_WIDTH = 7
TILE_HEIGHT = 3
COL_UNIT = 4    # text columns per (2q + r) hex unit step of 1
ROW_UNIT = 4    # text rows per r step of 1
LEFT_MARGIN = 2


def _tile_origin(q: int, r: int, r_min: int, col_min: int) -> tuple[int, int]:
    """Top-left text (col, row) of the tile centered at axial (q, r)."""
    col_unit = 2 * q + r
    text_col = (col_unit - col_min) * COL_UNIT + LEFT_MARGIN
    text_row = (r - r_min) * ROW_UNIT
    return text_col, text_row


def render_board(board, use_color: bool = True) -> str:
    """Render the hex board as a multiline ASCII string."""
    if not board.tiles:
        return "(empty board)"

    tiles = list(board.tiles.values())
    r_min = min(t.r for t in tiles)
    r_max = max(t.r for t in tiles)
    col_units = [2 * t.q + t.r for t in tiles]
    col_min = min(col_units)
    col_max = max(col_units)

    width = (col_max - col_min) * COL_UNIT + TILE_WIDTH + LEFT_MARGIN + 2
    height = (r_max - r_min) * ROW_UNIT + TILE_HEIGHT + 1

    # 2D grid of (char, ansi_prefix). ansi_prefix applied only to non-space chars.
    canvas: list[list[str]] = [[" "] * width for _ in range(height)]

    for tile in tiles:
        col0, row0 = _tile_origin(tile.q, tile.r, r_min, col_min)
        glyph = _RESOURCE_GLYPH.get(tile.resource, ".") if tile.resource else "."
        dice = f"{tile.dice_number:>2}" if tile.dice_number else "  "

        # Top border + bottom border
        top = ".-----."
        bot = "'-----'"
        # Middle: " G:DD "  e.g.  " W: 8 "
        mid = f" {glyph}:{dice} "

        color = _RESOURCE_COLOR.get(tile.resource, "") if tile.resource else DIM
        for i, ch in enumerate(top):
            canvas[row0][col0 + i] = ch
        for i, ch in enumerate(mid):
            # Colorize only the glyph and dice digits
            if ch != " ":
                canvas[row0 + 1][col0 + i] = _color(ch, color, use_color) if use_color else ch
            else:
                canvas[row0 + 1][col0 + i] = " "
        for i, ch in enumerate(bot):
            canvas[row0 + 2][col0 + i] = ch

        # Tile id in top-right corner of cell (overwrites part of top border)
        tid = str(tile.tile_id)
        if len(tid) <= 2:
            # place at col0+1..col0+2
            for i, ch in enumerate(tid):
                canvas[row0][col0 + 1 + i] = _color(ch, DIM, use_color) if use_color else ch

    lines = ["".join(row).rstrip() for row in canvas]
    # Strip leading/trailing empty rows
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


# ── Buildings & Roads Listing ────────────────────────────────────────────

def render_buildings(board, use_color: bool = True) -> str:
    """List settlements/cities and roads grouped by player."""
    # Group buildings
    by_player_buildings: dict[int, list[tuple[int, BuildType, list[int]]]] = {}
    abandoned: list[tuple[int, BuildType]] = []
    for inter in board.intersections.values():
        if inter.building is None:
            continue
        if inter.owner is None:
            abandoned.append((inter.inter_id, inter.building))
            continue
        tile_ids = list(inter.adjacent_tiles)
        by_player_buildings.setdefault(inter.owner, []).append(
            (inter.inter_id, inter.building, tile_ids)
        )

    # Group roads
    by_player_roads: dict[int, list[int]] = {}
    for edge in board.edges.values():
        if edge.road_owner is not None:
            by_player_roads.setdefault(edge.road_owner, []).append(edge.edge_id)

    lines = []
    all_pids = sorted(set(by_player_buildings) | set(by_player_roads))
    if not all_pids and not abandoned:
        return "Buildings: (none)"

    lines.append("Buildings & Roads:")
    for pid in all_pids:
        tag = _player_tag(pid, use_color)
        buildings = by_player_buildings.get(pid, [])
        roads = by_player_roads.get(pid, [])
        b_parts = []
        for inter_id, btype, tiles in buildings:
            glyph = _BUILD_GLYPH[btype]
            b_parts.append(f"{glyph}@i{inter_id}(t{','.join(str(t) for t in tiles)})")
        b_str = "  ".join(b_parts) if b_parts else "(no buildings)"
        r_str = f"  roads: {len(roads)} [{','.join(f'e{r}' for r in roads)}]" if roads else ""
        lines.append(f"  {tag}: {b_str}{r_str}")

    if abandoned:
        parts = [f"{_BUILD_GLYPH[bt]}@i{iid}" for iid, bt in abandoned]
        lines.append(f"  abandoned: {'  '.join(parts)}")

    return "\n".join(lines)


# ── Dashboard ────────────────────────────────────────────────────────────

def _format_resources(resources: dict[ResourceType, int], use_color: bool) -> str:
    parts = []
    for r in ResourceType:
        amt = resources.get(r, 0)
        glyph = _RESOURCE_GLYPH[r]
        color = _RESOURCE_COLOR[r]
        label = _color(glyph, color, use_color)
        parts.append(f"{label}:{amt}")
    return " ".join(parts)


def _format_dev_cards(cards: list[DevCardType]) -> str:
    if not cards:
        return "(none)"
    counts: dict[DevCardType, int] = {}
    for c in cards:
        counts[c] = counts.get(c, 0) + 1
    return ", ".join(f"{c.name.title()}x{n}" for c, n in counts.items())


def render_dashboard(state, use_color: bool = True) -> str:
    """Render bank, dice, turn, and per-player status."""
    lines = []
    dice_str = "—"
    if state.current_dice:
        d1, d2 = state.current_dice
        dice_str = f"{d1}+{d2}={d1+d2}"
    current_tag = _player_tag(state.current_player, use_color)
    lines.append(
        f"Turn {state.turn_number}  |  Current: {current_tag}  |  Dice: {dice_str}  "
        f"|  Win@{WIN_THRESHOLD}pp"
    )
    lines.append(f"Bank: {_format_resources(state.bank, use_color)}  "
                 f"|  Deck: {len(state.dev_card_deck)} cards")
    lines.append("")

    for pid in sorted(state.players.keys()):
        p = state.players[pid]
        tag = _player_tag(pid, use_color)
        status = "" if p.is_active else _color(" [OUT]", DIM, use_color)
        marker = "►" if pid == state.current_player else " "
        pp = p.progress_points
        pp_str = _color(f"{pp:>2}pp", BOLD, use_color)
        lines.append(
            f"{marker} {tag}{status}  {pp_str}  "
            f"res: {_format_resources(p.resources, use_color)}  "
            f"({p.total_resources} total)  "
            f"dev: {_format_dev_cards(p.dev_cards)}"
        )
        lines.append(f"    {_format_stats_inline(p.stats, use_color)}")

    return "\n".join(lines)


def _format_stats_inline(stats: PlayerStats, use_color: bool) -> str:
    """One-line per-player stat summary for the live dashboard."""
    built = stats.buildings_built
    s = built.get(BuildType.SETTLEMENT, 0)
    c = built.get(BuildType.CITY, 0)
    r = built.get(BuildType.ROAD, 0)
    parts = [
        f"built S{s}/C{c}/R{r}",
        f"lost {stats.buildings_lost}",
        f"trades {stats.trades_made}(b{stats.bank_trades}/p{stats.player_trades})",
        f"dev {stats.dev_cards_bought}b/{stats.dev_cards_played}p",
        f"divine {stats.divine_interventions}",
        f"rain {stats.rain_received}",
        f"barn {stats.barn_received}",
        f"flow +{stats.total_resources_earned}/-{stats.total_resources_spent}",
    ]
    return _color("  ".join(parts), DIM, use_color)


def render_final_stats(result: GameResult, use_color: bool = True) -> str:
    """End-of-game per-player stat summary."""
    lines = ["=== Final Stats ==="]
    winner_tag = (
        _player_tag(result.winner, use_color) if result.winner is not None else "—"
    )
    lines.append(f"Winner: {winner_tag}  |  Turns: {result.turns_played}")
    lines.append("")
    for pid in sorted(result.player_stats.keys()):
        s = result.player_stats[pid]
        tag = _player_tag(pid, use_color)
        score = result.scores.get(pid, 0)
        lines.append(f"{tag}  score={score}  owned={s.buildings_owned}")
        built = s.buildings_built
        lines.append(
            f"   built: S{built.get(BuildType.SETTLEMENT, 0)} "
            f"C{built.get(BuildType.CITY, 0)} "
            f"R{built.get(BuildType.ROAD, 0)}  "
            f"lost: {s.buildings_lost} "
            f"(maintenance failures: {s.maintenance_failures})"
        )
        lines.append(
            f"   trades: {s.trades_made} "
            f"(bank {s.bank_trades}, p2p {s.player_trades})  "
            f"dev cards: bought {s.dev_cards_bought}, played {s.dev_cards_played}  "
            f"divine: {s.divine_interventions}"
        )
        if s.divine_outcomes:
            outcomes = ", ".join(f"{k}:{v}" for k, v in s.divine_outcomes.items())
            lines.append(f"   divine outcomes: {outcomes}")
        lines.append(
            f"   weather: {s.rain_received} water (rain), {s.barn_received} cow (barn)"
        )
        lines.append(
            f"   resource flow: +{s.total_resources_earned} -{s.total_resources_spent} "
            f"(net {s.total_resources_earned - s.total_resources_spent})"
        )
    return "\n".join(lines)


# ── Full State Render ────────────────────────────────────────────────────

def render_full_state(
    state,
    last_action: Optional[ActionResult] = None,
    use_color: bool = True,
    show_buildings: bool = True,
) -> str:
    """Render board + buildings list + dashboard + last-action footer."""
    sections = []
    sections.append(render_board(state.board, use_color=use_color))
    if show_buildings:
        sections.append(render_buildings(state.board, use_color=use_color))
    sections.append(render_dashboard(state, use_color=use_color))
    if last_action is not None:
        ok = "✓" if last_action.success else "✗"
        desc = last_action.description or type(last_action.action).__name__
        sections.append(f"Last: {ok} {desc}")
    return "\n\n".join(sections)


# ── TerminalRenderer Class ───────────────────────────────────────────────

@dataclass
class TerminalRenderer:
    """Renders game state to stdout. Pass to Environment(..., renderer=...).

    Options:
      use_color: emit ANSI color codes
      clear_screen: clear terminal between frames for animation-like updates
      pause_per_action: seconds to sleep between frames (0 = no pause)
      show_buildings: include the buildings/roads listing below the board
    """
    use_color: bool = True
    clear_screen: bool = False
    pause_per_action: float = 0.0
    show_buildings: bool = True
    out = sys.stdout

    def render(self, state, last_action: Optional[ActionResult] = None) -> None:
        if self.clear_screen:
            # ANSI: cursor home + clear-to-end
            self.out.write("\033[H\033[J")
        text = render_full_state(
            state,
            last_action=last_action,
            use_color=self.use_color,
            show_buildings=self.show_buildings,
        )
        self.out.write(text + "\n")
        self.out.write("─" * 60 + "\n")
        self.out.flush()
        if self.pause_per_action > 0:
            import time
            time.sleep(self.pause_per_action)

    def render_event(self, message: str) -> None:
        """Print a one-line interstitial event (e.g., 'Turn 5 begins')."""
        self.out.write(message + "\n")
        self.out.flush()
