"""Check command for CLI."""

import sys
import json
from pathlib import Path

import click

from veripy.cli.extract import extract_verification_info


@click.command()
@click.argument("files", type=click.Path(exists=True, path_type=Path), nargs=-1)
@click.option("--output", "-o", type=click.Choice(["text", "json"]), default="text",
              help="Output format")
@click.pass_context
def check(ctx: click.Context, files: tuple, output: str):
    """Check files for verification annotations.
    
    This command scans Python files and reports what verification
    annotations (requires, ensures, invariants) are present.
    
    Examples:
    
        veripy check file.py
    
        veripy check --json directory/
    """
    verbose = ctx.obj.get("verbose", False)
    
    if not files:
        click.echo("Error: No files specified", err=True)
        sys.exit(1)
    
    all_info = []
    
    for file_path in files:
        if verbose:
            click.echo(f"\nChecking: {file_path}")
        
        info = extract_verification_info(file_path)
        info["file"] = str(file_path)
        all_info.append(info)
    
    if output == "json":
        click.echo(json.dumps(all_info, indent=2))
    else:
        for info in all_info:
            click.echo(f"\nFile: {info['file']}")
            
            if "error" in info:
                click.echo(f"  Error: {info['error']}")
                continue
            
            if info["functions"]:
                click.echo("  Functions with contracts:")
                for func in info["functions"]:
                    click.echo(f"    - {func['name']} (line {func['line']})")
                    if func["requires"]:
                        click.echo(f"      Requires: {', '.join(func['requires'])}")
                    if func["ensures"]:
                        click.echo(f"      Ensures: {', '.join(func['ensures'])}")
                    if func["decreases"]:
                        click.echo(f"      Decreases: {func['decreases']}")
            
            if info["invariants"]:
                click.echo(f"  Invariants: {len(info['invariants'])}")
                for inv in info["invariants"][:5]:  # Show first 5
                    click.echo(f"    Line {inv['line']}: {inv['expression']}")
