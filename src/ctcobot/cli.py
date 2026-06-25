"""
ctcobot CLI — ask questions, run indexing, run evaluation.
"""
import click


@click.group()
def cli():
    """ctcobot — HR Policy Assistant CLI."""
    pass


def _load_kedro_params():
    """Bootstrap Kedro and return the project parameters dict."""
    import sys
    from pathlib import Path

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
        return context.params


@cli.command()
@click.option(
    "--question", "-q",
    required=True,
    help="The HR policy question to ask.",
)
@click.option(
    "--top-k", "-k",
    default=20,
    show_default=True,
    help="Number of chunks for VectorRAG retrieval before reranking.",
)
def ask(question: str, top_k: int):
    """Ask ctcobot an HR policy question (agentic vector + BM25 RAG)."""
    params = _load_kedro_params()

    from ctcobot.pipelines.querying.tools import VectorRAGTool, KeywordRAGTool
    from ctcobot.pipelines.querying.agent import QueryAgent

    vector_tool = VectorRAGTool(
        ollama_base_url=params["ollama_base_url"],
        embedding_model=params["embedding_model"],
        llm_model=params["llm_model"],
        turbovec_persist_path=params["turbovec_persist_path"],
        top_k=top_k,
        reranker_model=params["reranker_model"],
        rerank_top_n=params["rerank_top_n"],
        top_folders=params["top_folders"],
        retrieve_oversample=params["retrieve_oversample"],
        priority_folders=params["priority_folders"],
    )
    keyword_tool = KeywordRAGTool(
        ollama_base_url=params["ollama_base_url"],
        llm_model=params["llm_model"],
        turbovec_persist_path=params["turbovec_persist_path"],
        top_k=params["bm25_top_k"],
        top_folders=params["top_folders"],
        priority_folders=params["priority_folders"],
    )
    agent = QueryAgent(
        tools={"vector_rag": vector_tool, "keyword_rag": keyword_tool},
        ollama_base_url=params["ollama_base_url"],
        agent_model=params["agent_model"],
        llm_model=params["llm_model"],
        reranker_model=params["reranker_model"],
        rerank_top_n=params["rerank_top_n"],
        turbovec_persist_path=params["turbovec_persist_path"],
        agent_reranker_model=params["agent_reranker_model"],
    )

    click.echo(f"\nSearching handbook for: {question}\n")
    result = agent.run(question)

    click.echo("─" * 60)
    click.echo("ANSWER")
    click.echo("─" * 60)
    click.echo(result["answer"])

    click.echo(f"\nTools used: {', '.join(result['tools_used'])}")

    click.echo("\nSOURCES")
    click.echo("─" * 60)
    for i, source in enumerate(result["sources"], start=1):
        click.echo(f"  {i}. {source['source_path']}  (score: {source['score']})")
    click.echo()


@cli.command()
def index():
    """Run the full vector indexing pipeline (ingest, clean, chunk, embed)."""
    import subprocess
    click.echo("Running indexing pipeline...")
    result = subprocess.run(
        ["kedro", "run", "--pipeline", "indexing"],
        capture_output=False,
    )
    if result.returncode == 0:
        click.echo("Indexing complete.")
    else:
        click.echo("Indexing failed. Check logs above.")


@cli.command("index_graph")
def index_graph():
    """Build the LightRAG knowledge graph index (run after index)."""
    import subprocess
    click.echo("Building LightRAG knowledge graph index...")
    click.echo("(This is slow — many LLM calls for entity/relationship extraction.)")
    result = subprocess.run(
        ["kedro", "run", "--pipeline", "lightrag_indexing"],
        capture_output=False,
    )
    if result.returncode == 0:
        click.echo("LightRAG indexing complete.")
    else:
        click.echo("LightRAG indexing failed. Check logs above.")


@cli.command()
def evaluate():
    """Run the benchmark evaluation pipeline."""
    import subprocess
    click.echo("Running evaluation pipeline...")
    result = subprocess.run(
        ["kedro", "run", "--pipeline", "evaluation"],
        capture_output=False,
    )
    if result.returncode == 0:
        click.echo("Evaluation complete. Results in data/08_reporting/")
    else:
        click.echo("Evaluation failed. Check logs above.")


run = cli


def main():
    cli()


if __name__ == "__main__":
    main()
