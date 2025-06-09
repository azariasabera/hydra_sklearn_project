from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline
from utils.custom_pipeline import CustomPipeline


def make_pipeline(steps: DictConfig) -> Pipeline:
    """
    Creates a pipeline using the preprocessing steps + model in `steps`

    Args:
        steps (DictConfig): steps in the pipeline, order is according to the sequence in `steps`

    Returns:
        [sklearn.pipeline.Pipeline]: a pipeline with preprocessing steps + model
    """
    step_tuples = [] # since sklearn.pipeline.Pipeline expects tuple: (name, step)

    for step in steps:

        # get name of step and the target
        name, target = next(iter(step.items())) # step.items()[0]

        # instantiate step, and append to step_tuples
        pipeline_step = (name, instantiate(target))
        step_tuples.append(pipeline_step)

    return Pipeline(step_tuples)

def make_custom_pipeline(steps: DictConfig) -> Pipeline:
    step_tuples = []

    for step in steps:
        name, target = next(iter(step.items()))
        pipeline_step = (name, instantiate(target))
        step_tuples.append(pipeline_step)

    return CustomPipeline(step_tuples)
