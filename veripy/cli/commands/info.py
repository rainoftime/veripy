"""Info command for CLI."""

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


@click.command()
@click.pass_context
def info(ctx: click.Context):
    """Display detailed information about veripy."""
    panel = Panel(
        Text("""
Veripy - Auto-Active Verification for Python

A production-ready verification system inspired by Dafny and Verus,
providing automated verification of Python programs using SMT solving.

Features:
  • Auto-active verification with automatic invariant inference
  • Contract-based specifications (requires/ensures)
  • Loop invariants and termination proofs
  • Quantifier support (forall, exists)
  • Array and data structure verification
  • Recursive function verification
  • SMT-based verification condition generation
  • Multiple output formats (text, JSON)
  • VS Code extension with LSP support

Getting Started:
  1. Import veripy: import veripy as vp
  2. Enable verification: vp.enable_verification()
  3. Define functions with @verify decorator
  4. Add contracts: @verify(requires=['x > 0'], ensures=['result > 0'])
  5. Call vp.verify_all() to verify

For more information, visit: https://github.com/veripy/veripy
        """, justify="left"),
        title="Veripy Information",
        expand=False
    )
    console.print(panel)
