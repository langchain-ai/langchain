from collections import OrderedDict

from attr import NOTHING

from langchain.concise._utils.templates import (
    extract_input_variables,
    infer_template_format,
)
from langchain.prompts import PromptTemplate


def function(template):
    template_format = infer_template_format(template)
    template_vars = extract_input_variables(template, template_format)
    prompt = PromptTemplate(
        input_variables=template_vars,
        template=template,
        template_format=template_format,
    )

    template_var_vals = {k: NOTHING for k in template_vars}

    def template_fn(**kwargs):
        nonlocal template_var_vals
        template_var_vals.update(kwargs)
        if all(v is not NOTHING for v in template_var_vals.values()):
            return prompt.render(template_var_vals)

    return template_fn
