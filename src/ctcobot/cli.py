"""
ctcobot CLI — ask questions, run indexing, run evaluation.
"""
import click


@click.group()
def cli():
    """ctcobot — HR Policy Assistant CLI."""
    pass


@cli.command()
@click.option(
    "--question", "-q",
    required=True,
    help="The HR policy question to ask.",
)
def ask(question: str):
    """Ask ctcobot an HR policy question."""
    import json
    import sys
    from pathlib import Path

    # Ensure src is on path when run directly
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from kedro.framework.project import configure_project
    from kedro.framework.session import KedroSession
    from kedro.framework.startup import bootstrap_project

    project_path = Path(__file__).parent.parent.parent
    bootstrap_project(project_path)
    configure_project("ctcobot")

    with KedroSession.create(project_path=project_path) as session:
        context = session.load_context()
        params = context.params

    from ctcobot.pipelines.querying.nodes import (
        build_prompt,
        generate_answer,
        pageindex_retrieve,
    )

    registry_path = project_path / "data/04_feature/doc_registry.json"
    if not registry_path.exists():
        click.echo(
            "❌ doc_registry.json not found. Run 'ctcobot index' first.",
            err=True,
        )
        raise SystemExit(1)

    with open(registry_path) as f:
        doc_registry = json.load(f)

    click.echo(f"\n🔍 Searching handbook for: {question}\n")

    chunks = pageindex_retrieve(
        question=question,
        doc_registry=doc_registry,
        pageindex_model=params["pageindex_model"],
        pageindex_workspace=params["pageindex_workspace"],
        ollama_base_url=params["ollama_base_url"],
        pageindex_top_docs=params["pageindex_top_docs"],
        pageindex_top_sections=params["pageindex_top_sections"],
    )

    prompt_data = build_prompt(question=question, chunks=chunks)

    result = generate_answer(
        prompt_data=prompt_data,
        ollama_base_url=params["ollama_base_url"],
        llm_model=params["llm_model"],
    )

    # Print answer
    click.echo("─" * 60)
    click.echo("📋 ANSWER")
    click.echo("─" * 60)
    click.echo(result["answer"])

    # Print sources
    click.echo("\n📄 SOURCES")
    click.echo("─" * 60)
    for i, source in enumerate(result["sources"], start=1):
        click.echo(f"  {i}. {source['source_path']}")
    click.echo()


@cli.command()
def index():
    """Run the full indexing pipeline (ingest, PageIndex tree build)."""
    import subprocess
    click.echo("🔄 Running indexing pipeline...")
    result = subprocess.run(
        ["kedro", "run", "--pipeline", "indexing"],
        capture_output=False,
    )
    if result.returncode == 0:
        click.echo("✅ Indexing complete.")
    else:
        click.echo("❌ Indexing failed. Check logs above.")


@cli.command()
def evaluate():
    """Run the benchmark evaluation pipeline."""
    import subprocess
    click.echo("📊 Running evaluation pipeline...")
    result = subprocess.run(
        ["kedro", "run", "--pipeline", "evaluation"],
        capture_output=False,
    )
    if result.returncode == 0:
        click.echo("✅ Evaluation complete. Results in data/08_reporting/")
    else:
        click.echo("❌ Evaluation failed. Check logs above.")


# Kedro's find_run_command looks for `project_cli.run`; expose the Click group here.
run = cli


def main():
    cli()


if __name__ == "__main__":
    main()
