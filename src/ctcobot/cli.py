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
@click.option(
    "--top-k", "-k",
    default=5,
    show_default=True,
    help="Number of chunks to retrieve.",
)
def ask(question: str, top_k: int):
    """Ask ctcobot an HR policy question."""
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

    # Run querying nodes directly (faster than kedro run for interactive use)
    from ctcobot.pipelines.querying.nodes import (
        generate_hyde_doc,
        embed_query, retrieve_chunks, build_prompt, generate_answer,
    )

    click.echo(f"\n🔍 Searching handbook for: {question}\n")

    hyde_doc = generate_hyde_doc(
        question=question,
        ollama_base_url=params["ollama_base_url"],
        llm_model=params["llm_model"],
    )

    embedding = embed_query(
        question=hyde_doc,
        ollama_base_url=params["ollama_base_url"],
        embedding_model=params["embedding_model"],
    )

    chunks = retrieve_chunks(
        query_embedding=embedding,
        chroma_persist_path=params["chroma_persist_path"],
        chroma_collection_name=params["chroma_collection_name"],
        top_k=top_k,
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
        click.echo(f"  {i}. {source['source_path']}  (score: {source['score']})")
    click.echo()


@cli.command()
def index():
    """Run the full indexing pipeline (ingest, clean, chunk, embed)."""
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


def main():
    cli()


if __name__ == "__main__":
    main()