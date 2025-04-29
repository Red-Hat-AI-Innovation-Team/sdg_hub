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
from rich.table import Table

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


def do_copy(copy_func, src: Path, dest: Path):
    """Helper function to copy with progress bar.

    Args:
        copy_func: Function to use for copying
        src: Source path
        dest: Destination path
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Copying {src.name}...", total=None)
        copy_func(src, dest)
        progress.update(task, completed=True)


def copy_with_progress(src: Path, dest: Path) -> None:
    """Copy file or directory with progress bar.

    Args:
        src: Source path
        dest: Destination path
    """
    if src.is_dir():
        do_copy(lambda s, d: shutil.copytree(s, d, dirs_exist_ok=True), src, dest)
    else:
        do_copy(shutil.copy2, src, dest)


@click.group(
    help="""
    SDG Hub CLI - A tool for synthetic data generation for customizing Large Language Models.

    This CLI provides commands to work with synthetic data generation examples
    and tools. Use the 'examples' command to get started with example projects.

    For more information about a specific command, use:
    sdg_hub COMMAND --help
    """
)
@click.version_option()
def main():
    pass


@main.group(
    help="""
    Commands for working with SDG Hub examples.

    Use 'list' to see available examples and 'init' to copy examples
    to your workspace.
    """
)
def examples():
    pass


@examples.command(
    help="""
    List all available examples.

    This command displays a table of all available examples that can be
    initialized using the 'init' command.
    """
)
def list():
    """List all available examples."""
    examples_dir = get_examples_dir()

    if not examples_dir.exists():
        console.print("[red]Error: Examples directory not found.[/red]")
        console.print("Please ensure the package is installed correctly.")
        sys.exit(1)

    available_examples = get_available_examples()
    if not available_examples:
        console.print("[yellow]No examples found in the examples directory.[/yellow]")
        return

    # Create a table to display the examples
    table = Table(title="Available Examples")
    table.add_column("Example Name", style="cyan")
    table.add_column("Description", style="green")

    # Add each example to the table
    for example in sorted(available_examples):
        # Try to read the README.md file for description
        readme_path = examples_dir / example / "README.md"
        description = "No description available"
        if readme_path.exists():
            try:
                with open(readme_path, "r") as f:
                    # Get the first non-empty line as description
                    for line in f:
                        if line.strip() and not line.startswith("#"):
                            description = line.strip()
                            break
            except Exception:
                pass

        table.add_row(example, description)

    console.print(table)
    console.print(
        "\nTo initialize an example, use: sdg_hub examples init --example <name>"
    )


@examples.command(
    help="""
    Initialize a directory with SDG Hub examples.

    This command copies example projects from the SDG Hub package to your
    specified directory. By default, it copies all available examples to the
    current directory.

    Examples:
        \b
        # Copy all examples to current directory
        sdg_hub examples init

        \b
        # Copy specific examples
        sdg_hub examples init --example instructlab --example knowledge_generation_using_nemotron

        \b
        # Copy to a specific directory
        sdg_hub examples init --target-dir /path/to/destination

        \b
        # Preview what would be copied
        sdg_hub examples init --dry-run
    """
)
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
def init(example: Optional[List[str]], dry_run: bool, target_dir: str):
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
