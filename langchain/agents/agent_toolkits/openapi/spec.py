"""Experimental (not tested) code to simplify an openapi spec for retrieving into a LM's context.
"""

from dataclasses import dataclass
from typing import List, Tuple, Union


def dereference_refs(spec_obj: dict, full_spec: dict):
    """Try to substitute $refs. The goal is to get the complete docs for each endpoint in context for now.

    In the few OpenAPI specs I studied, $refs referenced models (or in OpenAPI terms, components)
    and could be nested. This code most likely misses lots of cases.
    """
    def retrieve_ref_path(path: str, full_spec: dict):
        components = path.split('/')
        if components[0] != '#':
            raise RuntimeError("All $refs I've seen so far are uri fragments (start with hash).")
        out = full_spec
        for component in components[1:]:
            out = out[component]
        return out

    def _dereference_refs(obj: Union[dict, list]) -> Union[dict, list]:
        obj_out = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == '$ref':
                    return _dereference_refs(retrieve_ref_path(v, full_spec))
                elif isinstance(v, list):
                    obj_out[k] = [_dereference_refs(el) for el in v]
                elif isinstance(v, dict):
                    obj_out[k] = _dereference_refs(v)
                else:
                    obj_out[k] = v
            return obj_out
        elif isinstance(obj, list):
            return [_dereference_refs(el) for el in obj]
        else:
            return obj

    return _dereference_refs(spec_obj)


@dataclass
class ReducedOpenAPISpec:
    servers: List[dict]
    description: str
    endpoints: List[Tuple[str, str, dict]]


def reduce_openapi_spec(spec: dict) -> ReducedOpenAPISpec:
    """Simplify/distill/minify a spec somehow.
    I want a smaller target for retrieval and (more importantly) I want smaller results from retrieval.
    I was hoping https://openapi.tools/ would have some useful bits to this end, but doesn't seem so.
    """
    out = ReducedOpenAPISpec(
        servers=spec['servers'],
        description=spec['info']['description'],
        endpoints=[]
    )
    # 1. Consider only get, post endpoints.
    endpoints = [
        (
            f'{operation_name.upper()} {route}',
            docs['description'],
            docs
        )
         for route, operation in spec['paths'].items()
         for operation_name, docs in operation.items()
         if operation_name in ['get', 'post']
    ]

    # 2. Replace any refs so that complete docs are retrieved.
    # Note: proabably want to do this post-retrieval, it blows up the size of the spec.
    endpoints = [
        (name, description, dereference_refs(docs, spec))
        for name, description, docs in endpoints
    ]
    # 3. Strip docs down to required request args + happy path response.
    def reduce_endpoint_docs(docs: dict) -> dict:
        out = {}
        if docs.get("description"):
            out["description"] = docs.get("description")
        if docs.get("parameters"):
            out["parameters"] = [
                parameter
                for parameter in docs.get("parameters", [])
                if parameter.get("required") == True
            ]
        if "200" in docs["responses"]:
            out["responses"] = docs["responses"]["200"]
        return out

    endpoints = [
        (name, description, reduce_endpoint_docs(docs))
        for name, description, docs in endpoints
    ]

    out.endpoints = endpoints
    return out


if __name__ == "__main__":

    import os, yaml, subprocess
    spec_url = "https://raw.githubusercontent.com/APIs-guru/openapi-directory/main/APIs/spotify.com/1.0.0/openapi.yaml"
    subprocess.run(["wget", spec_url])
    with open("./openapi.yaml", "r") as f:
        spec = yaml.load(f, Loader=yaml.Loader)
    obj = spec['paths']['/tracks']
    assert '$ref' in yaml.dump(obj)
    assert '$ref' not in yaml.dump(dereference_refs(obj, spec))
    os.remove("./openapi.yaml")
