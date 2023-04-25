"""Wrapper around DeepSparse APIs."""

DEFAULT_MODEL_ID = '<PLACEHOLDER>'
DEFAULT_TASK = "text-classification"
VALID_MODELS = ('<PLACEHOLDER>', '<PLACEHOLDER>')

class DeepSparse():
    """Wrapper around DeepSparse Pipeline API for integrating
    transformers with LLMs.

    To use, you should have the `deepsparse` python package installed.

    Supports `text-classification`.

    Example passing pipeline in directly:
            
        .. code-block:: python
        
            from langchain.transformers.neuralmagic import DeepSparse

            meta_agent = DeepSparse(model='zoo:nlp/<PLACEHOLDER>')
            predict = meta_agent('i'm a prompt')
    """
    
    task: str = DEFAULT_TASK
    model: str = DEFAULT_MODEL_ID
    
    def __init__(self, model: str = DEFAULT_MODEL_ID, task: str = DEFAULT_TASK):
        
        self.task = task
        self.model = model
        
        try:
            from deepsparse import Pipeline
        except ImportError:
            raise ValueError(
                "Could not import deepsparse python package. "
                "Please install it with `pip install deepsparse`."
            )
        self.pipeline = Pipeline.create(task=self.task, model_path=self.model)
        if self.pipeline.model_path not in VALID_MODELS:
            raise ValueError(
                f"Got invalid MODEL {self.pipeline.model_path}, "
                f"currently only {self.VALID_MODELS} are supported"
            )
    
    def __call__(self, prompt: str) -> str:
        
        response = self.pipeline(prompt)
        response = response.labels[0]
        return response