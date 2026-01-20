"""Verify command for CLI."""

import sys
import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from veripy.cli.config import VerificationConfig
from veripy.cli.verification import verify_file

console = Console()


@click.command()
@click.argument("files", type=click.Path(exists=True, path_type=Path), nargs=-1)
@click.option("--function", "-f", multiple=True, help="Specific functions to verify")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), default="text",
              help="Output format")
@click.option("--counterexample/--no-counterexample", default=True,
              help="Show counterexamples when verification fails")
@click.option("--statistics/--no-statistics", default=True,
              help="Show verification statistics")
@click.option("--strict", is_flag=True, help="Enable strict verification mode")
@click.option("--timeout", type=int, help="Timeout in seconds for verification")
@click.option("--solver", type=click.Choice(["z3", "cvc5"]), default="z3",
              help="SMT solver to use")
@click.option("--incremental/--no-incremental", default=True,
              help="Use incremental SMT solving")
@click.option("--cache/--no-cache", default=True,
              help="Use verification cache")
@click.option("--workers", type=int, default=1, help="Number of parallel workers")
@click.pass_context
def verify(ctx: click.Context, files: tuple, function: tuple, output: str,
           counterexample: bool, statistics: bool, strict: bool, timeout: int,
           solver: str, incremental: bool, cache: bool, workers: int):
    """Verify Python files for correctness.
    
    This command analyzes Python files and verifies that functions with
    contracts (requires/ensures) satisfy their specifications.
    
    Examples:
    
        veripy verify file.py
    
        veripy verify --function add --counterexample file.py
    
        veripy verify --output json file1.py file2.py
    """
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)
    
    if not files:
        click.echo("Error: No files specified", err=True)
        click.echo("Run 'veripy verify --help' for usage", err=True)
        sys.exit(1)
    
    # Build configuration
    config = VerificationConfig(
        files=list(files),
        functions=list(function),
        output_format=output,
        show_counterexample=counterexample,
        show_statistics=statistics,
        strict=strict,
        timeout=timeout,
        solver=solver,
        incremental=incremental,
        cache=cache,
        workers=workers
    )
    
    if not quiet:
        click.echo(f"Veripy v0.1.0 - Verification Report")
        click.echo("=" * 60)
    
    all_results = []
    
    # Process files with progress bar if not quiet
    file_iter = tqdm(files, desc="Verifying", disable=quiet) if not quiet else files
    
    for file_path in file_iter:
        if verbose:
            click.echo(f"\nProcessing: {file_path}")
        
        results = verify_file(file_path, config)
        all_results.extend(results)
    
    # Display results
    if output == "json":
        output_data = {
            "summary": {
                "total": len(all_results),
                "passed": sum(1 for r in all_results if r.success),
                "failed": sum(1 for r in all_results if not r.success)
            },
            "results": [r.to_dict() for r in all_results]
        }
        click.echo(json.dumps(output_data, indent=2))
    else:
        # Create results table
        table = Table(title="Verification Results")
        table.add_column("Status", style="bold")
        table.add_column("Function")
        table.add_column("Message")
        
        passed = 0
        failed = 0
        
        for result in all_results:
            status = "[green]✓ PASS[/green]" if result.success else "[red]✗ FAIL[/red]"
            table.add_row(status, result.function_name, result.message)
            
            if result.success:
                passed += 1
            else:
                failed += 1
        
        if not quiet:
            console.print(table)
            
            # Summary
            total = len(all_results)
            click.echo(f"\nSummary: {passed}/{total} passed, {failed}/{total} failed")
            
            if failed > 0:
                sys.exit(1)
    
    if verbose:
        click.echo(f"\nConfiguration used:")
        click.echo(f"  Solver: {solver}")
        click.echo(f"  Incremental: {incremental}")
        click.echo(f"  Cache: {cache}")
        click.echo(f"  Workers: {workers}")
