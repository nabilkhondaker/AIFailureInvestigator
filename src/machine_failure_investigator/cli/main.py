"""Typer application entry point for ``neural-failure``."""

from __future__ import annotations

import typer

from machine_failure_investigator.cli import commands
from machine_failure_investigator.version import __version__

app = typer.Typer(
    name="neural-failure",
    help="AI Machine Failure Investigator — synthetic machinery diagnosis toolkit.",
    add_completion=False,
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"machine-failure-investigator {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True, help="Show version"
    ),
) -> None:
    """AI Machine Failure Investigator CLI."""


app.command("generate-data")(commands.cmd_generate_data)
app.command("train")(commands.cmd_train)
app.command("investigate")(commands.cmd_investigate)
app.command("evaluate")(commands.cmd_evaluate)


@app.command("report")
def cmd_report() -> None:
    """Generate a sample investigation report (alias of investigate)."""
    commands.cmd_investigate()


if __name__ == "__main__":
    app()
