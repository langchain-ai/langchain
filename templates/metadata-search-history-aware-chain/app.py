from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from extract import extract_list
from openai import OpenAI
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

import os, json, random
from dotenv import load_dotenv
import gradio as gr


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_index = os.getenv("INDEX")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

openai_client = OpenAI(api_key=openai_api_key)

practice_list = ', '.join(extract_list())

description = "Extract mediator's practice field that user want to search. Available mediator practice fields are " + practice_list

metadata_list = ['fullname', 'mediator profile on mediate.com', 'mediator Biography', 'mediator state']
metadata_value = ['Name', "Profile", "Biography", "State"]

class MediatorRetriever(BaseRetriever):
    def getMetadata(self, message):
        tools = [
            {
                "type": "function", 
                "function": {
                    "name": "get_info",
                    "description": "Extract the information of mediator.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mediator country": {
                                "type": "string",
                                "description": "Extract mediator's country that user want to search."
                                },
                            "mediator city": {
                                "type": "string",
                                "description": "Extract mediator's city that user want to search."
                                },
                            "mediator state": {
                                "type": "string",
                                "description": "Extract mediator's state that user want to search. If both mediator city and mediator state are possible, please extract as mediator state."
                                },
                            "mediator areas of practice": {
                                "type": "array",
                                "description": description,
                                "enum": extract_list()
                                },    
                            }
                        },
                    }
                }
        ]

        response = openai_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                    {"role": "system", "content": f"You are a professional mediation field searcher. Your role is to extract information about mediator from user's message."},
                    {"role": "user", "content": message}
                ],
            tools=tools
        )

        print("Message =>", response.choices[0])

        try:
            data = response.choices[0].message.tool_calls[0].function.arguments
            search_status = True
        
        except:
            search_status = False
            data = response.choices[0].message.content

        return {"search_status": search_status, "data": data}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Sync implementations for retriever."""

        matching_documents = []
        message = ""
        data = self.getMetadata(query)
        search_status = data['search_status']

        if search_status == True:
            metadata = json.loads(data['data'])
            print(metadata)
            try:
                if 'mediator areas of practice' in metadata:
                    practice_data = metadata['mediator areas of practice']
                    del metadata['mediator areas of practice']
                elif 'mediator practice field' in metadata:
                    practice_data = metadata['mediator practice field']
                    del metadata['mediator practice field']
                
            except:
                practice_data = ""

            if "mediator city" in metadata:
                try:
                    del metadata['mediator state']
                    del metadata['mediator country']
                except:
                    pass

            print("metadata =>", metadata)     

            tools = [
                    {
                        "type": "function", 
                        "function": {
                            "name": "mediator_search",
                            "description": "Extract how many mediators user want to search.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "mediator": {
                                        "type": "number",
                                        "description": "The number of mediators that user want to search. If user ask a list of mediators, it means user want to search 3 mediators. If user's message don't have information about the number of mediators, you have to respond with 1.",
                                        "default": 1
                                    }
                                },
                                "required": ["mediator"]
                            }
                        }
                    }
                ]
            
            response = openai_client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    messages=[
                            {"role": "system", "content": "Please extract how many mediators users want to search."},
                            {"role": "user", "content": query}
                        ],
                        tools=tools,
            )
            try:
                number_str = response.choices[0].message.tool_calls[0].function.arguments
                mediator_num = json.loads(number_str)['mediator']
            except:
                mediator_num = 1

            print(mediator_num)

            template = """"""
            # prompt = "You are a professional mediator information analyzer. You have to write the reason why following mediators are matched to human's message. You shouldn't write mediator's information again. You should't write the mediators in context are the excellent choice or ideal candidate. You have to analyze the mediators at once.  Please respond with no more than 300 characters. "
            prompt = "You are a professional mediator information analyze. You have to analyze the follwing mediators based on human's message. You shouldn't write mediator's information again. You should't write the mediators in context are the excellent choice or ideal candidate. You have to analyze the mediators at once.  Please respond with no more than 300 characters. "
            end = """Context: {context}
                Chat history: {chat_history}
                Human: {human_input}
                Your Response as Chatbot:"""
            
            template += prompt + end

            prompt = PromptTemplate(
                input_variables=["chat_history", "human_input", "context"], 
                template=template
            )
            # print(message)

            pc = Pinecone(api_key=pinecone_api_key)

            embeddings = OpenAIEmbeddings(api_key=openai_api_key)
            
            index = pc.Index(pinecone_index)

            results = index.query(
                vector=embeddings.embed_query(query),
                top_k=748,
                filter=metadata,
                include_metadata=True
            )

            print("num of result =>", len(results['matches']))
            
            new_data = []
            for result in results['matches']:
                data = {}
                for metadata in metadata_list:      
                    data[metadata] = result['metadata'][metadata]
                
                if practice_data in result['metadata']['mediator areas of practice']:
                    new_data.append(data)

            print(len(new_data))
            random.shuffle(new_data)

            if len(new_data) != 0:
                if practice_data != "" and mediator_num == 1:
                    message += f"I have located a mediator who specializes in {practice_data}.  Here are their details:\n\n"
                elif practice_data != "" and mediator_num > 1:
                    message += f"I have located mediators who specialize in {practice_data}.  Here are their details:\n\n"
                elif practice_data == "" and mediator_num == 1:
                    message += f"I have located a mediator.  Here are their details:\n\n"
                elif practice_data == "" and mediator_num > 1:
                    message += f"I have located mediators.  Here are their details:\n\n"

            for index, new_datum in enumerate(new_data):
                if index < mediator_num:
                    content = ""

                    for metadata_index, metadata in enumerate(metadata_list):
                        content += f"<b>{metadata_value[metadata_index]}</b>: {new_datum[metadata]} \n"
                        message += f"<b>{metadata_value[metadata_index]}</b>: {new_datum[metadata]} \n"

                    message += "\n\n"
                    new_doc = Document(page_content=content)
                    matching_documents.append(new_doc)
                else:
                    break
            
            chat_openai = ChatOpenAI(model='gpt-4-1106-preview', 
                    openai_api_key=openai_api_key)
            
            memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

            chain = load_qa_chain(chat_openai, chain_type="stuff",  prompt=prompt, memory=memory)

            output = chain({"input_documents": matching_documents, "human_input": query}, return_only_outputs=False)
            
            message += f"Why appropriate: {output['output_text']}"
        else:
            message += data['data']
        
        return_data = {"documents": matching_documents, "message": message}

        return return_data
    
retriever = MediatorRetriever()

from langchain_community.chat_models import ChatOpenAI
from langchain.chains import create_history_aware_retriever
from langchain import hub
from langchain_core.messages import HumanMessage

rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
llm = ChatOpenAI(model="gpt-4-1106-preview", api_key=openai_api_key)

chat_retriever_chain = create_history_aware_retriever(
    llm, retriever, rephrase_prompt
)

chat_history = []

def search(query, history):
    data = chat_retriever_chain.invoke({"input": query, "chat_history": chat_history})

    chat_history.extend([HumanMessage(content=query), data['documents']])

    return data['message']

chatbot = gr.Chatbot(avatar_images=["user.png", "bot.jpg"], height=600)

demo = gr.ChatInterface(fn=search, title="Mediate.com Chatbot Prototype", multimodal=False, retry_btn=None, clear_btn=None, undo_btn=None, chatbot=chatbot)

if __name__ == "__main__":
    demo.launch(debug=True)