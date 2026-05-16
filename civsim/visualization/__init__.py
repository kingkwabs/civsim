"""Terminal visualization for CivSim."""
from .terminal import (
    TerminalRenderer,
    render_board,
    render_buildings,
    render_dashboard,
    render_final_stats,
    render_full_state,
)

# GUI renderer is optional — requires matplotlib
from .gui import GUIRenderer, HAS_MPL as HAS_GUI  # noqa: F401

__all__ = [
    "TerminalRenderer",
    "render_board",
    "render_buildings",
    "render_dashboard",
    "render_final_stats",
    "render_full_state",
]
if HAS_GUI:
    __all__.append("GUIRenderer")
