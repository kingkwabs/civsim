"""Tests for terminal visualization."""
from __future__ import annotations

import re

from civsim.agents import RandomAgent
from civsim.board import Board
from civsim.data_types import BuildType, ResourceType
from civsim.environment import Environment
from civsim.game_state import GameState
from civsim.visualization import (
    TerminalRenderer,
    render_board,
    render_buildings,
    render_dashboard,
    render_full_state,
)


ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def test_render_board_nonempty():
    board = Board.standard_layout(seed=1)
    out = render_board(board, use_color=False)
    assert out
    lines = out.splitlines()
    # All 19 tiles laid out across multiple text rows
    assert len(lines) >= 5
    # Every standard resource glyph should appear at least once
    plain = _strip_ansi(out)
    for glyph in ["W", "S", "M", "H", "A"]:  # COW only appears once, may overlap
        assert glyph in plain, f"missing tile glyph {glyph}"


def test_render_board_color_codes_included():
    board = Board.standard_layout(seed=1)
    colored = render_board(board, use_color=True)
    plain = render_board(board, use_color=False)
    assert "\033[" in colored
    assert "\033[" not in plain
    # Stripping ANSI from colored should match plain (modulo whitespace)
    assert _strip_ansi(colored).replace(" ", "") == plain.replace(" ", "")


def test_render_dashboard_shows_all_players():
    state = GameState.new_game(num_players=3, seed=7)
    out = _strip_ansi(render_dashboard(state, use_color=False))
    assert "P0" in out and "P1" in out and "P2" in out
    assert "Turn" in out
    assert "Bank" in out


def test_render_buildings_lists_owners():
    state = GameState.new_game(num_players=3, seed=7)
    # Manually plant a settlement so the buildings render has content
    inter = next(iter(state.board.intersections.values()))
    state.board.place_settlement(0, inter.inter_id)
    edge = state.board.edges[inter.adjacent_edges[0]]
    state.board.place_road(1, edge.edge_id)

    out = _strip_ansi(render_buildings(state.board, use_color=False))
    assert "P0" in out
    assert "P1" in out
    assert f"i{inter.inter_id}" in out


def test_render_full_state_combines_sections():
    state = GameState.new_game(num_players=3, seed=7)
    out = render_full_state(state, use_color=False)
    plain = _strip_ansi(out)
    assert "Turn" in plain
    assert "Bank" in plain
    # Board section present (bottom borders of tiles always have this exact form)
    assert "'-----'" in out


def test_renderer_integrates_with_environment():
    """A rendered run_game should complete and not crash."""
    import io

    buf = io.StringIO()
    renderer = TerminalRenderer(use_color=False, pause_per_action=0.0)
    renderer.out = buf

    agents = [RandomAgent(player_id=i, seed=i + 1) for i in range(3)]
    env = Environment(agents, num_players=3, renderer=renderer)
    result = env.run_game()

    assert result is not None
    output = buf.getvalue()
    assert "Turn" in output
    # Snake draft completion event
    assert "Draft complete" in output
    # Game-over banner
    assert "Game over" in output


def test_renderer_silent_when_not_passed():
    """Default Environment should not render anything."""
    agents = [RandomAgent(player_id=i, seed=i + 1) for i in range(3)]
    env = Environment(agents, num_players=3)
    assert env.renderer is None
    result = env.run_game()
    assert result is not None
