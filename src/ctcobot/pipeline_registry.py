from ctcobot.pipelines.evaluation.pipeline import create_pipeline as create_evaluation_pipeline
from ctcobot.pipelines.indexing.pipeline import create_pipeline as create_indexing_pipeline
from ctcobot.pipelines.lightrag_indexing.pipeline import create_pipeline as create_lightrag_indexing_pipeline
from ctcobot.pipelines.querying.pipeline import create_pipeline as create_querying_pipeline
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines."""
    indexing_pipeline = create_indexing_pipeline()
    querying_pipeline = create_querying_pipeline()
    evaluation_pipeline = create_evaluation_pipeline()
    lightrag_indexing_pipeline = create_lightrag_indexing_pipeline()

    return {
        "indexing": indexing_pipeline,
        "querying": querying_pipeline,
        "evaluation": evaluation_pipeline,
        "lightrag_indexing": lightrag_indexing_pipeline,
        "__default__": indexing_pipeline,
    }
