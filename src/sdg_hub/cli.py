import shutil
import click
from pathlib import Path
from typing import List, Optional
import sys
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.console import Console

console = Console()


def get_examples_dir() -> Path:
    """Get the path to the examples directory."""
    package_dir = Path(__file__).parent.parent.parent
    return package_dir / "examples"


def get_available_examples() -> List[str]:
    """Get list of available example directories."""
    examples_dir = get_examples_dir()
    if not examples_dir.exists():
        return []
    return [d.name for d in examples_dir.iterdir() if d.is_dir()]


def copy_with_progress(src: Path, dest: Path) -> None:
    """Copy file or directory with progress bar."""
    if src.is_dir():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Copying {src.name}...", total=None)
            shutil.copytree(src, dest, dirs_exist_ok=True)
            progress.update(task, completed=True)
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Copying {src.name}...", total=None)
            shutil.copy2(src, dest)
            progress.update(task, completed=True)


@click.group()
@click.version_option()
def main():
    """SDG Hub CLI - A tool for working with synthetic data generation examples."""
    pass


@main.command()
@click.option(
    "--example",
    "-e",
    multiple=True,
    help="Specific example(s) to copy. Can be specified multiple times.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be copied without actually copying.",
)
@click.option(
    "--target-dir",
    "-t",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default=".",
    help="Target directory where examples will be copied. Defaults to current directory.",
)
def init(example: Optional[tuple], dry_run: bool, target_dir: str):
    """Initialize directory with examples.

    By default, copies all examples to the current directory.
    Use --example to copy specific examples.
    Use --target-dir to specify where to copy the examples.
    """
    target_path = Path(target_dir).resolve()

    # Create target directory if it doesn't exist
    if not target_path.exists():
        try:
            target_path.mkdir(parents=True)
        except Exception as e:
            console.print(f"[red]Error creating target directory: {str(e)}[/red]")
            sys.exit(1)

    examples_dir = get_examples_dir()

    if not examples_dir.exists():
        console.print("[red]Error: Examples directory not found.[/red]")
        console.print("Please ensure the package is installed correctly.")
        sys.exit(1)

    available_examples = get_available_examples()
    if not available_examples:
        console.print(
            "[yellow]Warning: No examples found in the examples directory.[/yellow]"
        )
        return

    # If specific examples are requested, validate them
    if example:
        invalid_examples = set(example) - set(available_examples)
        if invalid_examples:
            console.print(
                f"[red]Error: Invalid example(s): {', '.join(invalid_examples)}[/red]"
            )
            console.print(f"Available examples: {', '.join(available_examples)}")
            sys.exit(1)
        examples_to_copy = example
    else:
        examples_to_copy = available_examples

    if dry_run:
        console.print(
            f"[yellow]Dry run: Would copy the following to {target_path}:[/yellow]"
        )
        for example_name in examples_to_copy:
            console.print(f"  - {example_name}")
        return

    # Copy the selected examples
    for example_name in examples_to_copy:
        src = examples_dir / example_name
        dest = target_path / example_name
        try:
            copy_with_progress(src, dest)
        except Exception as e:
            console.print(f"[red]Error copying {example_name}: {str(e)}[/red]")
            sys.exit(1)

    console.print(
        f"[green]Successfully initialized {target_path} with examples![/green]"
    )
    if example:
        console.print(f"Copied examples: {', '.join(examples_to_copy)}")
    else:
        console.print("Copied all available examples.")
