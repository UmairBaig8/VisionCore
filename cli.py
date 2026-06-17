import typer

from core.agent_loader import AgentLoader
from core.orchestrator import VideoOrchestrator
from core.paths import agents_dir, skills_dir, videos_dir, output_dir

app = typer.Typer(help="Video Analysis Agent Platform")


@app.command()
def agents():
    """Show loaded agents"""
    loader = AgentLoader()
    loaded = loader.load()

    typer.echo("")
    typer.echo("Loaded Agents")
    typer.echo("-------------")
    for name in loaded:
        typer.echo(f"\u2713 {name}")


@app.command()
def skills():
    """Show loaded skills"""
    from core.registry import SkillRegistry

    registry = SkillRegistry()
    registry.load()

    typer.echo("")
    typer.echo("Loaded Skills")
    typer.echo("-------------")
    for skill in registry.skills:
        typer.echo(f"\u2713 {skill}")


@app.command()
def videos():
    """List available videos"""
    vdir = videos_dir()

    if not vdir.exists():
        typer.echo("videos/ folder not found")
        raise typer.Exit(1)

    files = [f for f in vdir.iterdir() if f.is_file()]

    if not files:
        typer.echo("No videos found")
        return

    typer.echo("")
    typer.echo("Available Videos")
    typer.echo("----------------")
    for idx, video in enumerate(files, start=1):
        typer.echo(f"{idx}. {video.name}")


@app.command()
def analyze(
    video: str,
    interval: float = typer.Option(
        0.5, "--interval", "-i",
        help="Frame sampling interval in seconds"
    ),
    depth: str = typer.Option(
        "full", "--depth", "-d",
        help="Analysis depth: scene-only, fast, full"
    ),
):
    """Analyze a video"""
    from pathlib import Path

    video_path = Path(video)
    if not video_path.exists():
        typer.echo(f"Video not found: {video}")
        raise typer.Exit(1)

    if depth not in ("scene-only", "fast", "full"):
        typer.echo(f"Invalid depth: {depth}. Use: scene-only, fast, full")
        raise typer.Exit(1)

    orchestrator = VideoOrchestrator(
        video_path=str(video_path),
        sample_interval=interval,
        depth=depth,
    )
    orchestrator.run()


@app.command()
def stream(
    video: str,
    interval: float = typer.Option(0.5),
    depth: str = typer.Option(
        "fast", "--depth", "-d",
        help="Analysis depth: scene-only, fast, full"
    ),
):
    """Live event stream"""
    from pathlib import Path

    video_path = Path(video)
    if not video_path.exists():
        typer.echo(f"Video not found: {video}")
        raise typer.Exit(1)

    if depth not in ("scene-only", "fast", "full"):
        typer.echo(f"Invalid depth: {depth}. Use: scene-only, fast, full")
        raise typer.Exit(1)

    orchestrator = VideoOrchestrator(
        video_path=str(video_path),
        sample_interval=interval,
        depth=depth,
        stream_mode=True,
    )
    orchestrator.run()


@app.command()
def report(video: str):
    """Generate report from existing timeline"""
    from pathlib import Path

    video_path = Path(video)
    if not video_path.exists():
        typer.echo(f"Video not found: {video}")
        raise typer.Exit(1)

    orchestrator = VideoOrchestrator(
        video_path=str(video_path),
        report_only=True,
    )
    orchestrator.run()


@app.command()
def doctor():
    """Validate environment"""
    typer.echo("")
    typer.echo("Environment Check")
    typer.echo("-----------------")

    checks = {
        "agents": agents_dir().exists(),
        "skills": skills_dir().exists(),
        "videos": videos_dir().exists(),
        "output": output_dir().exists(),
    }

    for name, status in checks.items():
        icon = "\u2713" if status else "\u2717"
        typer.echo(f"{icon} {name}")


@app.command()
def config():
    """Show runtime config"""
    from core.config import load_config

    cfg = load_config()

    typer.echo("")
    typer.echo("Configuration")
    typer.echo("-------------")
    for key, value in cfg.items():
        typer.echo(f"{key}: {value}")


if __name__ == "__main__":
    app()
