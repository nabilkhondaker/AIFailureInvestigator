"""Optional Streamlit dashboard entry (lazy import)."""

from __future__ import annotations


def launch_dashboard() -> None:
    """
    Placeholder for an optional Streamlit app.

    Install extras: pip install -e ".[dashboard]"
    Then implement UI against Investigator API.
    """
    try:
        import streamlit as st  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Streamlit is required for the dashboard. Install with: pip install -e '.[dashboard]'"
        ) from e
    print("Dashboard scaffolding is available; extend visualization.dashboards for a full UI.")
