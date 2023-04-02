import json
from pydantic import BaseModel
from langchain.chains.base import Chain
from typing import Dict, List, Optional
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.llms.base import BaseLLM
from langchain import VectorDBQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredFileLoader
import tempfile
import magic

import os
import subprocess
import requests


DOCUMENTATION_PROMPT = """I want to use the API to do the following: {question}.
Explain in great detail how to do that. Only answer the specific question. Don't offer alternative or related answers.
Do not invent or hallucinate any information! You can only use information found in the context documents.
If you aren't sure how to do something, or the question can't be answered using the API, then you should just return `ERROR` as the response.
Include an example of how to make the API request using the Python Requests library.
Always include the full URL in every request (including the protocol, domain, and path).
"""

REQUEST_PROMPT = """You are the world's best Python programmer. You are given the below API Documentation:

{api_docs}

Using this documentation, generate the the parameters to use in a Python `requests` call for answering the user question.
You should build the request parameters in order to get a response that is as short as possible, while still getting the necessary information to answer the question. Pay attention to deliberately exclude any unnecessary pieces of data in the API call.

The Python `requests` library is used to make the API call.
The requests library in Python provides several parameters that you can use to customize your HTTP request. Here is a brief overview of some of the most common parameters:

method - (required) method for the new Request object: GET, OPTIONS, HEAD, POST, PUT, PATCH, or DELETE.
url - (required) URL for the new Request object.
params - (optional) Dictionary, list of tuples or bytes to send in the query string for the Request.
json - (optional) A JSON serializable Python object to send in the body of the Request.
headers - (optional) Dictionary of HTTP Headers to send with the Request.

You can only make ONE API call. If you're not sure how to construct the request parameters, then you should just return `ERROR` as the response.
You should return a valid, properly formatted JSON object with the parameters to use in the `requests` call. Make sure to use double quotes surrounding all keys and values. Booleans should be `true` or `false`, not `True` or `False`.

{prompt_parameters}

{last_error}

{last_successful_request}

Question:{question}
JSON object:"""

LLM_API_PROMPT = PromptTemplate(
    input_variables=[
        "api_docs",
        "question",
        "last_error",
        "last_successful_request",
        "prompt_parameters"
    ],
    template=REQUEST_PROMPT,
)


class LLMAPIChain(Chain, BaseModel):
    """Use an LLM to interact with API endpoints."""
    llm: BaseLLM
    parameters: dict
    question_key: str = "request"
    output_key: str = "output"
    verbose: bool = True
    retries: int = 2
    last_successful_request: Optional[dict] = None
    last_error: Optional[str] = ""

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.question_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key, "last_successful_request"]

    def api_docs(self, question = None)-> str:
        """builds the api docs"""
        if 'open_api_url' in self.parameters:
            open_api_url = self.parameters['open_api_url']
            api_docs = self.get_api_docs_from_open_api_url(
                open_api_url, question)
        elif 'documentation_urls' in self.parameters:
            documentation_urls = self.parameters['documentation_urls']
            api_docs = self.get_api_docs_from_url(documentation_urls, question)

        return api_docs

    def generate_markdown_docs_from(self, url, filename=None)-> str:
        path = f"/tmp/{filename}"
        
        try:
            subprocess.check_output(["widdershins", "--version"])
        except FileNotFoundError:
            raise FileNotFoundError(
                "widdershins npm package not installed. Please install it with `npm install -g widdershins`")
        
        cmd = f"widdershins {url} -o {path} --language_tabs python:Python:requests -s false"
        # split the command into a list based on spaces
        cmd = cmd.split(" ")
        output = subprocess.check_output(cmd)
        # open the file and read the contents
        return path

    def get_api_docs_from_open_api_url(self, open_api_url, question)-> str:
        """get the api docs from an open api url"""
        import hashlib

        url_hash = hashlib.sha256(
            ("").join(open_api_url).encode('utf-8')).hexdigest()

        if self.llm.__class__.__name__ != "OpenAIChat":
            self.llm.max_tokens = 2000

        chroma_directory = f"/tmp/{url_hash}"
            
        embeddings = OpenAIEmbeddings()

        if os.path.exists(chroma_directory):
            vectorstore = Chroma(persist_directory=chroma_directory, embedding_function=embeddings)
        else:
            os.mkdir(chroma_directory)
            data = self.generate_markdown_docs_from(
                open_api_url, filename=f"{url_hash}.md")

            loader = TextLoader(data)
            documents = loader.load()

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)

            vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=chroma_directory)

        prompt = DOCUMENTATION_PROMPT
        if self.last_error:
            prompt = prompt + \
                f"Also, keep in mind this failing request when answering: {self.last_error}"


        qa = VectorDBQA.from_chain_type(llm=self.llm, 
                                        chain_type="stuff", 
                                        vectorstore=vectorstore)
        
        formatted_prompt = prompt.format(question=question)            

        if self.last_error:
          formatted_prompt = formatted_prompt + \
              f"Also, keep in mind this failing request when answering: {self.last_error}"

        answer  = qa.run(formatted_prompt)

        return str(answer)

    def get_api_docs_from_url(self, documentation_urls, question)-> str:
        """gets the api docs from a url"""
        import hashlib
        url_hash = hashlib.sha256(
            ("").join(documentation_urls).encode('utf-8')).hexdigest()

        chroma_directory = f"/tmp/{url_hash}"
        embeddings = OpenAIEmbeddings()

        if self.llm.__class__.__name__ != "OpenAIChat":
            self.llm.max_tokens = 2000

        if isinstance(documentation_urls, list):
            urls = documentation_urls
        else:
            urls = [documentation_urls]
        
        documents = []
        for url in urls:
            response = requests.get(
                url, headers={"User-Agent": "Mozilla/5.0"})
            if response.status_code == 200:
                content = response.content
                file_type = magic.from_buffer(
                    content, mime=True).split("/")[1]

                with tempfile.NamedTemporaryFile(suffix=f".{file_type}", delete=False) as f:
                    f.write(content)
                    loader = UnstructuredFileLoader(f.name)
                    document = loader.load()[0]
                    documents.append(document)

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=chroma_directory)

        prompt = DOCUMENTATION_PROMPT

        qa = VectorDBQA.from_chain_type(llm=self.llm, 
                                        k=1,
                                        chain_type="stuff", 
                                        vectorstore=vectorstore)

        formatted_prompt = prompt.format(question=question)            

        if self.last_error:
          formatted_prompt = formatted_prompt + \
              f"Also, keep in mind this failing request when answering: {self.last_error}"

        answer  = qa.run(formatted_prompt)
        


        return str(answer)

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """calls the tool"""
          
        question = inputs[self.question_key]

        api_docs = self.api_docs(question)

        if api_docs.strip() == 'ERROR':
            raise Exception(
                "Sorry, it doesn't look like the API can be used to answer your question.")

        prompt_parameters = self.parameters.copy()
        
        if 'open_api_url' in prompt_parameters:
            del prompt_parameters['open_api_url']
        if 'documentation_urls' in prompt_parameters:
            del prompt_parameters['documentation_urls']
        if 'last_error' in prompt_parameters:
            del prompt_parameters['last_error']
        if 'last_successful_request' in prompt_parameters:
            del prompt_parameters['last_successful_request']

        # format the prompt parameters to be displayed in the prompt, one per line
        prompt_parameters_modified = "\n".join(
            [f"The {k} is {v}" for k, v in prompt_parameters.items()])

        formatted_last_error = ""
        if 'last_error' in self.parameters:
            formatted_last_error = f"""Here's an example of WHAT NOT TO DO: {self.parameters['last_error']}
DO NOT generate the same JSON object as above. Try changing some of the parameters! Don't use the same request parameters as above!"""

        last_successful_request_as_string = ""
        if 'last_successful_request' in self.parameters:
            self.last_successful_request = self.parameters['last_successful_request']
            last_successful_request_as_string = f"""This JSON object DID WORK for the question {question}: {json.dumps(self.last_successful_request, indent=4)}"""

        api_request_params_chain = LLMChain(
            llm=self.llm,
            prompt=LLM_API_PROMPT,
            verbose=self.verbose
        )

        request_params_json = api_request_params_chain.predict(
            question=question,
            api_docs=api_docs,
            last_error=formatted_last_error,
            last_successful_request=last_successful_request_as_string,
            prompt_parameters=prompt_parameters_modified
        )

        try:
            if "ERROR" in request_params_json:
                raise Exception(
                    "Sorry, couldn't generate the API request for your question.")
            request_params_dict = json.loads(request_params_json)
        except Exception as e:
            self.callback_manager.on_text(
                api_docs, color="yellow", end="\n", verbose=self.verbose
            )
            self.callback_manager.on_text(
                request_params_json, color="red", end="\n", verbose=self.verbose
            )
            self.last_error = f"""This request DID NOT WORK for the question {question}: {request_params_json}"""
            raise Exception(e)

        self.callback_manager.on_text(
            request_params_dict, color="yellow", end="\n", verbose=self.verbose
        )

        if not request_params_dict['url']:
            raise Exception(
                "Sorry, couldn't generate the API request URL for your question.")
        if self.parameters.get('api_endpoint', "https") not in request_params_dict['url']:
            raise Exception(
                "Sorry, the API request URL doesn't match the configured API endpoint.")

        response = requests.request(**request_params_dict)

        if response.status_code not in [200, 201]:
            self.last_error = f"""This JSON object DID NOT WORK for the question {question}: {json.dumps(request_params_dict, indent=4)}"""

            self.callback_manager.on_text(
                self.last_error, color="red", end="\n", verbose=self.verbose
            )

            if self.retries > 0:
                self.retries -= 1
                return self._call(inputs)
            else:
                raise Exception(
                    f"Sorry, the API call failed with status code {response.status_code}.\nRequest params: {json.dumps(request_params_dict)}")

        self.last_successful_request = json.dumps(request_params_json)
        output = response.text

        self.callback_manager.on_text(
            output, color="green", end="\n", verbose=self.verbose
        )

        return {self.output_key: output, "last_successful_request": self.last_successful_request}

    @ property
    def _chain_type(self) -> str:
        return "llm_api_chain"
