from pydantic import BaseModel
from .core import Invoker, InvokerFactory, Prompty, SimpleModel
from jinja2 import DictLoader, Environment


class Jinja2Renderer(Invoker):
    def __init__(self, prompty: Prompty) -> None:
        self.prompty = prompty
        self.templates = {}
        # generate template dictionary
        cur_prompt = self.prompty
        while cur_prompt:
            self.templates[cur_prompt.file.name] = cur_prompt.content
            cur_prompt = cur_prompt.basePrompty

        self.name = self.prompty.file.name

    def invoke(self, data: BaseModel) -> BaseModel:
        assert isinstance(data, SimpleModel)
        env = Environment(loader=DictLoader(self.templates))
        t = env.get_template(self.name)
        generated = t.render(**data.item)
        return SimpleModel[str](item=generated)


InvokerFactory().register_renderer("jinja2", Jinja2Renderer)
