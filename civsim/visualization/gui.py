"""Colonist-inspired matplotlib GUI renderer for CivSim.

Layout:
    +-----------------------------------+--------------+
    |                                   |  P0 (Greedy) |
    |                                   |  ...stats... |
    |          HEX BOARD                +--------------+
    |   (proper pointy-top hexagons,    |  P1 (MCTS)   |
    |    dice tokens, on-board roads    |  ...stats... |
    |    + settlements + cities,        +--------------+
    |    ports on the border)           |  P2 (RL)     |
    |                                   |  ...stats... |
    +-----------------------------------+--------------+
    |               Action log (last 6 lines)          |
    +--------------------------------------------------+

Hexagons are real RegularPolygons (not ASCII). Roads draw as line
segments between intersections; settlements are colored circles,
cities are larger squares. Active player is highlighted; agent type
(Random/Greedy/MCTS/RL) is shown next to each player ID.

Performance: each `render()` call clears and redraws the board axes.
With pause=0.1s per frame and ~5 actions/turn × 200 turns, a typical
game renders in ~100s — fine for live demos and screen recording.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..data_types import (
    ActionResult, BuildType, DevCardType, ResourceType, WIN_THRESHOLD,
)

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle, RegularPolygon, FancyBboxPatch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Hex geometry (pointy-top) ────────────────────────────────────────────

HEX_SIZE = 1.0
SQRT3 = math.sqrt(3)


def hex_to_pixel(q: int, r: int) -> tuple[float, float]:
    """Axial coords → pixel coords for pointy-top hexes. Y is flipped
    (negated) so that increasing `r` moves visually downward."""
    x = HEX_SIZE * (SQRT3 * q + SQRT3 / 2 * r)
    y = -HEX_SIZE * (3 / 2 * r)
    return (x, y)


# ── Visual palette ───────────────────────────────────────────────────────

RESOURCE_COLORS = {
    ResourceType.WOOD:  "#3a7d3a",
    ResourceType.STONE: "#888a8e",
    ResourceType.METAL: "#b87333",
    ResourceType.WHEAT: "#e6c14b",
    ResourceType.WATER: "#3a8fd7",
    ResourceType.COW:   "#e3c89a",
}
DESERT_COLOR = "#d6b890"

PLAYER_COLORS = ["#e64545", "#3a8fd7", "#f29422", "#9b59b6"]  # red, blue, orange, purple

# Background colors
BG_DARK = "#1e1e2e"
BG_PANEL = "#2a2a3e"
TEXT_DIM = "#9999aa"
TEXT_FG = "#dddde0"
ACCENT = "#fafafa"

RESOURCE_LETTERS = {
    ResourceType.WOOD: "W", ResourceType.STONE: "S", ResourceType.METAL: "M",
    ResourceType.WHEAT: "H", ResourceType.WATER: "A", ResourceType.COW: "C",
}


@dataclass
class GUIRenderer:
    """matplotlib-based renderer. Constructor takes the agents so we can
    show agent type (Random/Greedy/MCTS/RL) per player.

    Hooks into Environment exactly like TerminalRenderer:
      env = Environment(agents, renderer=GUIRenderer(agents))
    """

    agents: list = field(default_factory=list)
    pause_per_frame: float = 0.1
    show_window: bool = True

    fig: object = None
    ax_board: object = None
    ax_log: object = None
    ax_players: list = field(default_factory=list)
    log_messages: list = field(default_factory=list)
    last_action_str: str = ""
    _intersection_positions: dict = field(default_factory=dict)
    _agent_names: dict = field(default_factory=dict)

    def __post_init__(self):
        if not HAS_MPL:
            raise ImportError(
                "matplotlib required for GUIRenderer. Install with: pip install matplotlib"
            )
        self._agent_names = {
            a.player_id: type(a).__name__.replace("Agent", "")
            for a in self.agents
        }
        self._setup_figure()

    # ── Figure layout ───────────────────────────────────────────────

    def _setup_figure(self) -> None:
        plt.ion()
        self.fig = plt.figure(figsize=(16, 9), facecolor=BG_DARK)
        self.fig.canvas.manager.set_window_title("CivSim")

        # GridSpec: board takes left ~60%, players take right ~40%, log at bottom
        gs = self.fig.add_gridspec(
            4, 5, hspace=0.4, wspace=0.25,
            left=0.03, right=0.98, top=0.96, bottom=0.04,
        )
        self.ax_board = self.fig.add_subplot(gs[0:3, 0:3])
        n_players = max(3, len(self.agents))
        self.ax_players = [
            self.fig.add_subplot(gs[i, 3:5]) for i in range(min(3, n_players))
        ]
        self.ax_log = self.fig.add_subplot(gs[3, :])

        for ax in [self.ax_board] + self.ax_players + [self.ax_log]:
            ax.set_facecolor(BG_DARK)
            for spine in ax.spines.values():
                spine.set_color("#444")

        if self.show_window:
            plt.show(block=False)

    # ── Public API (matches TerminalRenderer) ───────────────────────

    def render(self, state, last_action: Optional[ActionResult] = None) -> None:
        if self._intersection_positions == {} or len(self._intersection_positions) != len(state.board.intersections):
            self._intersection_positions = self._compute_intersection_positions(state.board)

        if last_action is not None and last_action.description:
            ok = "✓" if last_action.success else "✗"
            self.last_action_str = f"{ok} {last_action.description}"
            self.log_messages.append(
                f"T{state.turn_number} P{state.current_player}: {self.last_action_str}"
            )
            if len(self.log_messages) > 8:
                self.log_messages = self.log_messages[-8:]

        self._render_board(state)
        for i, ax in enumerate(self.ax_players):
            if i in state.players:
                self._render_player_panel(ax, state, i)
        self._render_log(state)

        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            pass
        if self.pause_per_frame > 0:
            plt.pause(self.pause_per_frame)

    def render_event(self, message: str) -> None:
        """Used for weather, end-of-turn, and game-over notes."""
        self.log_messages.append(message)
        if len(self.log_messages) > 8:
            self.log_messages = self.log_messages[-8:]
        self._render_log(None)
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def close(self) -> None:
        plt.close(self.fig)

    # ── Board ───────────────────────────────────────────────────────

    def _render_board(self, state) -> None:
        ax = self.ax_board
        ax.clear()
        ax.set_facecolor(BG_DARK)
        ax.set_aspect("equal")
        ax.axis("off")

        # Title
        dice = "—"
        if state.current_dice:
            d1, d2 = state.current_dice
            dice = f"{d1}+{d2}={d1+d2}"
        active_color = PLAYER_COLORS[state.current_player % len(PLAYER_COLORS)]
        ax.set_title(
            f"Turn {state.turn_number}     Dice: {dice}     "
            f"Active: P{state.current_player}",
            color=active_color, fontsize=15, weight="bold", pad=12,
        )

        # Tile hexes
        for tile in state.board.tiles.values():
            cx, cy = hex_to_pixel(tile.q, tile.r)
            color = RESOURCE_COLORS.get(tile.resource, DESERT_COLOR) if tile.resource else DESERT_COLOR
            hex_patch = RegularPolygon(
                (cx, cy), 6, radius=HEX_SIZE * 0.97,
                orientation=0,  # pointy-top
                facecolor=color, edgecolor="#0d0d18", linewidth=1.5, zorder=1,
            )
            ax.add_patch(hex_patch)
            # Resource letter (small, top of tile)
            if tile.resource is not None:
                ax.text(
                    cx, cy + 0.45, RESOURCE_LETTERS[tile.resource],
                    ha="center", va="center", fontsize=9,
                    color="#1c1c20", weight="bold", zorder=2,
                )
            # Dice number "token" — white circle with red text for 6/8
            if tile.dice_number is not None:
                is_hot = tile.dice_number in (6, 8)
                token_circ = Circle(
                    (cx, cy - 0.05), 0.32, facecolor="#fefefe",
                    edgecolor="#1c1c20", linewidth=1.2, zorder=3,
                )
                ax.add_patch(token_circ)
                ax.text(
                    cx, cy - 0.05, str(tile.dice_number),
                    ha="center", va="center", fontsize=12,
                    color="#cc1111" if is_hot else "#222",
                    weight="bold", zorder=4,
                )

        # Roads (drawn under buildings so building markers cap them visually)
        for edge in state.board.edges.values():
            if edge.road_owner is None:
                continue
            i1, i2 = edge.intersections
            p1 = self._intersection_positions.get(i1)
            p2 = self._intersection_positions.get(i2)
            if p1 is None or p2 is None:
                continue
            color = PLAYER_COLORS[edge.road_owner % len(PLAYER_COLORS)]
            ax.plot(
                [p1[0], p2[0]], [p1[1], p2[1]],
                color=color, linewidth=5.5, solid_capstyle="round", zorder=5,
            )

        # Ports — each port sits at a single border intersection. Render
        # as a colored diamond *outside* the intersection (in the sea
        # area) with a 2:1 / 3:1 ratio label, and a faint line linking
        # the diamond to the coast intersection.
        from ..data_types import PORT_RESOURCE_MAP
        for port in state.board.ports:
            iids = port.intersection_ids
            if not iids:
                continue
            inter_pos = self._intersection_positions.get(iids[0])
            if inter_pos is None:
                continue
            # Push the marker outward from the board center
            ix, iy = inter_pos
            length = math.sqrt(ix * ix + iy * iy) or 1.0
            outward = (ix / length, iy / length)
            marker_pos = (ix + outward[0] * 0.55, iy + outward[1] * 0.55)
            r = PORT_RESOURCE_MAP.get(port.port_type)
            port_color = RESOURCE_COLORS.get(r, "#dcdcdc") if r else "#dcdcdc"
            diamond = RegularPolygon(
                marker_pos, 4, radius=0.32, orientation=math.pi / 4,
                facecolor=port_color, edgecolor="#0d0d18", linewidth=1.5,
                alpha=0.95, zorder=6,
            )
            ax.add_patch(diamond)
            ax.text(
                marker_pos[0], marker_pos[1], f"{port.ratio}:1",
                ha="center", va="center", fontsize=8,
                color="#1c1c20", weight="bold", zorder=7,
            )
            # Faint tether from port marker to its coast intersection
            ax.plot(
                [marker_pos[0], ix], [marker_pos[1], iy],
                color=port_color, linewidth=1.2, alpha=0.5, zorder=2,
            )

        # Settlements (circles) and Cities (squares with cross)
        for inter in state.board.intersections.values():
            if inter.building is None:
                continue
            pos = self._intersection_positions.get(inter.inter_id)
            if pos is None:
                continue
            color = PLAYER_COLORS[inter.owner % len(PLAYER_COLORS)] if inter.owner is not None else "#888"
            if inter.building == BuildType.SETTLEMENT:
                marker = Circle(
                    pos, 0.18, facecolor=color, edgecolor="#0d0d18",
                    linewidth=2.0, zorder=8,
                )
                ax.add_patch(marker)
            else:  # CITY
                marker = Rectangle(
                    (pos[0] - 0.22, pos[1] - 0.22), 0.44, 0.44,
                    facecolor=color, edgecolor="#0d0d18", linewidth=2.0, zorder=8,
                )
                ax.add_patch(marker)
                # Small inner cross to differentiate from settlement at a glance
                ax.plot(
                    [pos[0] - 0.10, pos[0] + 0.10], [pos[1], pos[1]],
                    color="#0d0d18", linewidth=1.5, zorder=9,
                )
                ax.plot(
                    [pos[0], pos[0]], [pos[1] - 0.10, pos[1] + 0.10],
                    color="#0d0d18", linewidth=1.5, zorder=9,
                )

        # Auto-fit bounds
        xs = [hex_to_pixel(t.q, t.r)[0] for t in state.board.tiles.values()]
        ys = [hex_to_pixel(t.q, t.r)[1] for t in state.board.tiles.values()]
        margin = 1.5
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

    def _compute_intersection_positions(self, board) -> dict:
        """Pixel position for each intersection.

        Fully interior intersections (3 adjacent tiles) sit at the centroid
        of the 3 tile centers — which mathematically IS the shared corner
        for a regular hex grid. Border intersections (2 tiles) get pushed
        outward perpendicular to the shared edge so they appear *outside*
        the two tiles, on the coast.
        """
        positions = {}
        for inter in board.intersections.values():
            centers = [hex_to_pixel(board.tiles[tid].q, board.tiles[tid].r)
                       for tid in inter.adjacent_tiles]
            if len(centers) >= 3:
                x = sum(c[0] for c in centers) / len(centers)
                y = sum(c[1] for c in centers) / len(centers)
            elif len(centers) == 2:
                a, b = centers
                mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                # Perpendicular to edge AB
                dx, dy = b[0] - a[0], b[1] - a[1]
                length = math.sqrt(dx * dx + dy * dy) or 1.0
                perp = (-dy / length, dx / length)
                # Push outward (away from the board's mass center, which is ~origin)
                dot = mid[0] * perp[0] + mid[1] * perp[1]
                if dot < 0:
                    perp = (-perp[0], -perp[1])
                # Hex corner is at distance size/sqrt(3) from edge midpoint
                offset = HEX_SIZE / SQRT3
                x = mid[0] + perp[0] * offset
                y = mid[1] + perp[1] * offset
            else:
                x, y = 0.0, 0.0
            positions[inter.inter_id] = (x, y)
        return positions

    # ── Player panels ───────────────────────────────────────────────

    def _render_player_panel(self, ax, state, pid: int) -> None:
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        is_active = (pid == state.current_player)
        bg = "#3a3a55" if is_active else BG_PANEL
        ax.add_patch(Rectangle(
            (0, 0), 1, 1, transform=ax.transAxes,
            facecolor=bg, edgecolor=PLAYER_COLORS[pid % len(PLAYER_COLORS)] if is_active else "#444",
            linewidth=2.5 if is_active else 1.0,
        ))

        player = state.players[pid]
        color = PLAYER_COLORS[pid % len(PLAYER_COLORS)]
        agent_name = self._agent_names.get(pid, "?")
        status = "" if player.is_active else " [OUT]"

        # Header: agent type + PP + active marker
        marker = "◉" if is_active else " "
        ax.text(
            0.03, 0.78, f"{marker} P{pid} ({agent_name}){status}",
            color=color, fontsize=13, weight="bold", transform=ax.transAxes,
        )
        ax.text(
            0.97, 0.78, f"{player.progress_points}/{WIN_THRESHOLD} PP",
            color=ACCENT, fontsize=13, weight="bold", ha="right", transform=ax.transAxes,
        )

        # Resources line (colored letters + counts)
        x = 0.03
        for r in ResourceType:
            amt = player.resources.get(r, 0)
            ax.text(
                x, 0.50, RESOURCE_LETTERS[r],
                color=RESOURCE_COLORS[r], fontsize=11, weight="bold",
                family="monospace", transform=ax.transAxes,
            )
            ax.text(
                x + 0.025, 0.50, f":{amt}",
                color=TEXT_FG, fontsize=10,
                family="monospace", transform=ax.transAxes,
            )
            x += 0.075

        # Dev cards summary
        if player.dev_cards:
            counts = {}
            for c in player.dev_cards:
                counts[c] = counts.get(c, 0) + 1
            dev_str = ", ".join(f"{c.name.title()}×{n}" for c, n in counts.items())
        else:
            dev_str = "no dev cards"
        ax.text(
            0.03, 0.30, f"Dev: {dev_str}",
            color=TEXT_DIM, fontsize=8.5, transform=ax.transAxes,
        )

        # Stats line: built / lost / trades / weather / flow
        s = player.stats
        built = s.buildings_built
        stats_line = (
            f"Built S{built.get(BuildType.SETTLEMENT, 0)}/"
            f"C{built.get(BuildType.CITY, 0)}/"
            f"R{built.get(BuildType.ROAD, 0)}    "
            f"Lost {s.buildings_lost}    "
            f"Trades {s.trades_made} (b{s.bank_trades}/p{s.player_trades})    "
            f"Rain {s.rain_received}    Barn {s.barn_received}    "
            f"Flow +{s.total_resources_earned}/-{s.total_resources_spent}"
        )
        ax.text(
            0.03, 0.10, stats_line,
            color=TEXT_DIM, fontsize=7.5, transform=ax.transAxes,
        )

    # ── Action log ──────────────────────────────────────────────────

    def _render_log(self, state) -> None:
        ax = self.ax_log
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        ax.add_patch(Rectangle(
            (0, 0), 1, 1, transform=ax.transAxes,
            facecolor=BG_PANEL, edgecolor="#444", linewidth=1.0,
        ))

        if state is not None:
            bank_str = "  ".join(
                f"{RESOURCE_LETTERS[r]}:{state.bank.get(r, 0)}" for r in ResourceType
            )
            ax.text(
                0.01, 0.85, f"Bank:  {bank_str}    Deck: {len(state.dev_card_deck)} cards",
                color=TEXT_FG, fontsize=9.5, family="monospace",
                transform=ax.transAxes,
            )

        # Recent log lines (most recent at top)
        for i, msg in enumerate(reversed(self.log_messages[-6:])):
            ax.text(
                0.01, 0.65 - i * 0.11, msg,
                color=TEXT_FG if i == 0 else TEXT_DIM,
                fontsize=8.5, family="monospace",
                transform=ax.transAxes,
            )
