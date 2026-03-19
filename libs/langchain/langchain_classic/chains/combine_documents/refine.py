"""Combine documents by doing a first pass and then refining on more documents."""

from __future__ import annotations

from typing import Any

from langchain_core._api import deprecated
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.prompts import BasePromptTemplate, format_document
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import ConfigDict, Field, model_validator

from langchain_classic.chains.combine_documents.base import (
    BaseCombineDocumentsChain,
)
from langchain_classic.chains.llm import LLMChain


def _get_default_document_prompt() -> PromptTemplate:
    return PromptTemplate(input_variables=["page_content"], template="{page_content}")


@deprecated(
    since="0.3.1",
    removal="1.0",
    message=(
        "This class is deprecated. Please see the migration guide here for "
        "a recommended replacement: "
        "https://python.langchain.com/docs/versions/migrating_chains/refine_docs_chain/"
    ),
)
class RefineDocumentsChain(BaseCombineDocumentsChain):
    """Combine documents by doing a first pass and then refining on more documents.

    This algorithm first calls `initial_llm_chain` on the first document, passing
    that first document in with the variable name `document_variable_name`, and
    produces a new variable with the variable name `initial_response_name`.

    Then, it loops over every remaining document. This is called the "refine" step.
    It calls `refine_llm_chain`,
    passing in that document with the variable name `document_variable_name`
    as well as the previous response with the variable name `initial_response_name`.

    Example:
        ```python
        from langchain_classic.chains import RefineDocumentsChain, LLMChain
        from langchain_core.prompts import PromptTemplate
        from langchain_openai import OpenAI

        # This controls how each document will be formatted. Specifically,
        # it will be passed to `format_document` - see that function for more
        # details.
        document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}"
        )
        document_variable_name = "context"
        model = OpenAI()
        # The prompt here should take as an input variable the
        # `document_variable_name`
        prompt = PromptTemplate.from_template("Summarize this content: {context}")
        initial_llm_chain = LLMChain(llm=model, prompt=prompt)
        initial_response_name = "prev_response"
        # The prompt here should take as an input variable the
        # `document_variable_name` as well as `initial_response_name`
        prompt_refine = PromptTemplate.from_template(
            "Here's your first summary: {prev_response}. "
            "Now add to it based on the following context: {context}"
        )
        refine_llm_chain = LLMChain(llm=model, prompt=prompt_refine)
        chain = RefineDocumentsChain(
            initial_llm_chain=initial_llm_chain,
            refine_llm_chain=refine_llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name,
            initial_response_name=initial_response_name,
        )
        ```
    """

    initial_llm_chain: LLMChain
    """LLM chain to use on initial document."""
    refine_llm_chain: LLMChain
    """LLM chain to use when refining."""
    document_variable_name: str
    """The variable name in the initial_llm_chain to put the documents in.
    If only one variable in the initial_llm_chain, this need not be provided."""
    initial_response_name: str
    """The variable name to format the initial response in when refining."""
    document_prompt: BasePromptTemplate = Field(
        default_factory=_get_default_document_prompt,
    )
    """Prompt to use to format each document, gets passed to `format_document`."""
    return_intermediate_steps: bool = False
    """Return the results of the refine steps in the output."""

    @property
    def output_keys(self) -> list[str]:
        """Expect input key."""
        _output_keys = super().output_keys
        if self.return_intermediate_steps:
            _output_keys = [*_output_keys, "intermediate_steps"]
        return _output_keys

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def get_return_intermediate_steps(cls, values: dict) -> Any:
        """For backwards compatibility."""
        if "return_refine_steps" in values:
            values["return_intermediate_steps"] = values["return_refine_steps"]
            del values["return_refine_steps"]
        return values

    @model_validator(mode="before")
    @classmethod
    def get_default_document_variable_name(cls, values: dict) -> Any:
        """Get default document variable name, if not provided."""
        if "initial_llm_chain" not in values:
            msg = "initial_llm_chain must be provided"
            raise ValueError(msg)

        llm_chain_variables = values["initial_llm_chain"].prompt.input_variables
        if "document_variable_name" not in values:
            if len(llm_chain_variables) == 1:
                values["document_variable_name"] = llm_chain_variables[0]
            else:
                msg = (
                    "document_variable_name must be provided if there are "
                    "multiple llm_chain input_variables"
                )
                raise ValueError(msg)
        elif values["document_variable_name"] not in llm_chain_variables:
            msg = (
                f"document_variable_name {values['document_variable_name']} was "
                f"not found in llm_chain input_variables: {llm_chain_variables}"
            )
            raise ValueError(msg)
        return values


"""Why i am suggestiing this :-
See your refine logic currently do this
if i have 5 chunks it will send one chunk at a time then second chunk 2nd time now the problem is that this cost tokens and time also 
so what i am suggesting this is my simpler version:-
1.see although my suggestion is based on this code
#RETRIEVAL
def retrieval(DB_PATH,emb=HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5",model_kwargs={"device":"cpu"},encode_kwargs = {"normalize_embeddings": True},query_instruction="Represent this sentence for searching relevant passages: "), texts=None, query=None, temperature=None, model=None, secret_key=None, Upload=None, public=None):
    if DB_PATH:
      if os.path.exists(DB_PATH):
          vector_db=FAISS.load_local(
              DB_PATH,
              emb,
              allow_dangerous_deserialization=True
          )
          print("2")
          no_of_chunks= vector_db.index.ntotal
          dynamic_k= max(1,int(no_of_chunks*0.70))
          retriever=vector_db.similarity_search(query,k=dynamic_k)
          return preref(text=retriever,question=query,temperature=temperature,model=model)
      
      else:
          text_splitter=RecursiveCharacterTextSplitter(
              chunk_size=1000,
              chunk_overlap=200
          )
          chunks=text_splitter.split_text(texts)
          
          vector_db=FAISS.from_texts(chunks,emb)
          vector_db.save_local(DB_PATH)

          if Upload:
              index_file_faiss_var= open(f"{DB_PATH}/index.faiss","rb")
              index_file_pkl_var= open(f"{DB_PATH}/index.pkl","rb")
              
              if secret_key:
                  try:
                      index1faiss = (
                          supabase.storage
                          .from_("NPMRagWebVectorDB")
                          .upload(
                              file=index_file_faiss_var,
                              path=f"{secret_key}/{DB_PATH}/index.faiss",
                              file_options={"upsert": "false"}
                          )
                      )
                      print(index1faiss)
                  except:
                      return "Sorry Some problems in uploading your Documents in Database, reupload the documents"
                  
                  try:
                      index2pkl= (
                          supabase.storage
                          .from_("NPMRagWebVectorDB")
                          .upload(
                              file=index_file_pkl_var,
                              path=f"{secret_key}/{DB_PATH}/index.pkl",
                              file_options={"upsert": "false"}
                          )
                      )
                      print(index2pkl)
                  except:
                      file_first_faiss_removal = (
                          supabase.storage
                          .from_("NPMRagWebVectorDB")
                          .remove([f"{secret_key}/{DB_PATH}/index.faiss"])
                      )
                      print(file_first_faiss_removal)
                      
                      return "Sory some problem in uploading your Documents in Database, reupload the documents"
                      
              elif public:
                  try:
                      index_public_faiss=(
                          supabase.storage
                          .from_("NPMRagWebVectorDB")
                          .upload(
                              file=index_file_faiss_var,
                              path=f"public/{DB_PATH}/index.faiss",
                              file_options={"upsert": "false"}
                          )
                      )
                      print(index_public_faiss)
                  except:
                      return "Sory some problem in uploading your Documents in Database, reupload the documents"

                  try:
                      index_public_pkl=(
                          supabase.storage
                          .from_("NPMRagWebVectorDB")
                          .upload(
                              file=index_file_pkl_var,
                              path=f"public/{DB_PATH}/index.pkl",
                              file_options={"upsert": "false"}
                          )
                      )
                      print(index_public_pkl)
                  except:
                      file_first_faiss_removal = (
                          supabase.storage
                          .from_("NPMRagWebVectorDB")
                          .remove([f"public/{DB_PATH}/index.faiss"])
                      )
                      print(file_first_faiss_removal)
                      return "Sory some problem in uploading your Documents in Database, reupload the documents"
                      
              else:
                  return "Sorry please pass at least Secret_key or Public param in order to save your document in Database for persistent memory."
                  
          else:
              no_of_chunks_d= vector_db.index.ntotal
              dynamic_k_d= max(1,(no_of_chunks_d*0.70))
              retriever=vector_db.similarity_search(query,k=dynamic_k_d)
              return preref(text=retriever,question=query,temperature=temperature,model=model)
    else:
      return "Sorry but you have to provide query and DB_PATH also in order to retrieve from Vectorised DataBase"

  
#REFINE INITIALISATION
def preref(text,question, temperature, model, **kwargs):
  ref=refine(
      text=text,
      question=question,
      temperature=temperature,
      model=model
  )
  result=ref.refinef()
  return result


#REFINE
class refine:
  def __init__(self,text,question, temperature, model):
    self.text=text
    self.question=question
    self.temperature=temperature
    self.model=model

  def refinef(self):
    texts=self.text
    question=self.question
    temperature=self.temperature
    model=self.model
    answers=[]
    no=len(texts)
    no_of_loop=0
    if no > 2:
        chunks_send=no/3
        for i in range(chunks_send):
            context=texts[i*3:(i+1)*3]
            final_context="\n---\n".join([doc.page_content for doc in context])"""
            #prompt=f"""Use the following information to answer the question: 
            #Text: {final_context}
            #Existing Answer: {answers}
            #Question: {question}
            #""" 
            """
            llm=Ollama(
                model=model,
                temperature=temperature
            )
            response=llm.invoke(prompt)
            
            if answers:
                answers.remove(answers[0])
                answers.append(response)
            else:
                answers.append(response)
            
            if not no_of_loop==chunks_send:
                no_of_loop+=1
            else:
                pass
        return response
    elif no == 2 or no < 2:
      context_texts=texts[0:2]
      final_context_e="\n---\n".join([doc.page_content for doc in context_texts])"""
      #prompt=f"""Use the following information to answer the question: 
      #Text: {final_context_e}
      #Existing Answer: {answers}
      #Question: {question}
      #"""
"""
      llm=Ollama(
          model=model,
          temperature=temperature
      )
      response_e=llm.invoke(prompt)
      return response

HERE you will see that i am using Ollama under npmai so the avergae content limit is 4k ok so i am  selecting 3 chunks to send at a time now i am selectioing 3 beacuase chunk
size is 1000 ok but this is a issue where we have to understand user llm and chunk size and also chunk overlap although i gave you a simple idea of how you can do this
"""

    
    def combine_docs(
        self,
        docs: list[Document],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> tuple[str, dict]:
        """Combine by mapping first chain over the first batch, 
        then refining over subsequent batches.
        """
        # 1. Define Batch Size (Default to 3 for speed, can be passed via kwargs)
        batch_size = kwargs.get("batch_size", 3)

        # 2. Construct Initial Inputs (Index 0 is the starting point)
        # Note: Initial chain still takes the first document to start the 'Existing Answer'
        inputs = self._construct_initial_inputs(docs, **kwargs)
        res = self.initial_llm_chain.predict(callbacks=callbacks, **inputs)
        refine_steps = [res]

        # 3. Batch Loop: Start from index 1 and jump by 'batch_size'
        for i in range(1, len(docs), batch_size):
            # Slice the list to get the next batch of documents
            batch = docs[i : i + batch_size]
            
            # Combine the batch into a single string using LangChain's formatter
            combined_page_content = "\n\n".join(
                [format_document(doc, self.document_prompt) for doc in batch]
            )
            
            # Create a temporary Document to pass into the refine logic
            # This keeps the 'construct_refine_inputs' helper working perfectly
            combined_doc = Document(page_content=combined_page_content)

            # 4. Refine Step
            base_inputs = self._construct_refine_inputs(combined_doc, res)
            inputs = {**base_inputs, **kwargs}
            res = self.refine_llm_chain.predict(callbacks=callbacks, **inputs)
            refine_steps.append(res)

        return self._construct_result(refine_steps, res)



    async def acombine_docs(
        self,
        docs: list[Document],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> tuple[str, dict]:
        """Combine by mapping a first chain over all, then stuffing into a final chain.

        Args:
            docs: List of documents to combine
            callbacks: Callbacks to be passed through
            **kwargs: additional parameters to be passed to LLM calls (like other
                input variables besides the documents)

        Returns:
            The first element returned is the single string output. The second
            element returned is a dictionary of other keys to return.
        """
        inputs = self._construct_initial_inputs(docs, **kwargs)
        res = await self.initial_llm_chain.apredict(callbacks=callbacks, **inputs)
        refine_steps = [res]
        for doc in docs[1:]:
            base_inputs = self._construct_refine_inputs(doc, res)
            inputs = {**base_inputs, **kwargs}
            res = await self.refine_llm_chain.apredict(callbacks=callbacks, **inputs)
            refine_steps.append(res)
        return self._construct_result(refine_steps, res)

    def _construct_result(self, refine_steps: list[str], res: str) -> tuple[str, dict]:
        if self.return_intermediate_steps:
            extra_return_dict = {"intermediate_steps": refine_steps}
        else:
            extra_return_dict = {}
        return res, extra_return_dict

    def _construct_refine_inputs(self, doc: Document, res: str) -> dict[str, Any]:
        return {
            self.document_variable_name: format_document(doc, self.document_prompt),
            self.initial_response_name: res,
        }

    def _construct_initial_inputs(
        self,
        docs: list[Document],
        **kwargs: Any,
    ) -> dict[str, Any]:
        base_info = {"page_content": docs[0].page_content}
        base_info.update(docs[0].metadata)
        document_info = {k: base_info[k] for k in self.document_prompt.input_variables}
        base_inputs: dict = {
            self.document_variable_name: self.document_prompt.format(**document_info),
        }
        return {**base_inputs, **kwargs}

    @property
    def _chain_type(self) -> str:
        return "refine_documents_chain"
