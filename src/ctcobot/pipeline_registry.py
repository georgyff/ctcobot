from kedro.pipeline import Pipeline
from ctcobot.pipelines.indexing.pipeline import create_pipeline as create_indexing_pipeline
from ctcobot.pipelines.querying.pipeline import create_pipeline as create_querying_pipeline
from ctcobot.pipelines.evaluation.pipeline import create_pipeline as create_evaluation_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines."""
    indexing_pipeline = create_indexing_pipeline()
    querying_pipeline = create_querying_pipeline()
    evaluation_pipeline = create_evaluation_pipeline()

    return {
        "indexing": indexing_pipeline,
        "querying": querying_pipeline,
        "evaluation": evaluation_pipeline,
        "__default__": indexing_pipeline,
    }