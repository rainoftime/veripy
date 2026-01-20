"""Version command for CLI."""

import click


@click.command()
@click.pass_context
def version(ctx: click.Context):
    """Display version information."""
    from veripy.cli import __version__
    
    click.echo(f"Veripy v{__version__}")
    click.echo()
    click.echo("Dependencies:")
    click.echo("  - z3-solver: SMT solver backend")
    click.echo("  - pyparsing: Expression parsing")
    click.echo("  - apronpy: Abstract interpretation")
    click.echo("  - rich: Terminal output formatting")
    click.echo("  - click: Command-line interface")
