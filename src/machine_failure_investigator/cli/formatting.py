"""CLI output formatting."""

from __future__ import annotations

from rich.console import Console

console = Console()


def print_success(msg: str) -> None:
    console.print(f"[green]{msg}[/green]")


def print_info(msg: str) -> None:
    console.print(f"[cyan]{msg}[/cyan]")


def print_warning(msg: str) -> None:
    console.print(f"[yellow]{msg}[/yellow]")
