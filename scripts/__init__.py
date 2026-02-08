import subprocess
import sys
import click


@click.command()
def build_static():
    subprocess.run(
        ["python", "docs/reference/elements/slm/one_step_func_plots.py"],
        check=True,
        capture_output=True,
    )


@click.command()
@click.option(
    "--sv-dir",
    default=None,
    is_flag=True,
    flag_value="../SVETlANNa/svetlanna",
    help="Path to the SVETlANNa repository",
)
def dev(sv_dir):
    subprocess.run(
        [
            "mkdocs",
            "serve",
            "--livereload",
            "-w",
            "./docs",
        ]
        + (["-w", sv_dir] if sv_dir else []),
        check=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
        stdin=sys.stdin,
    )
