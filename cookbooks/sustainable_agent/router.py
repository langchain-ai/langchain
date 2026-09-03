"""
Semantic Router Module

This module analyzes user input and decides to which specialized expert 
(agent or lightweight/heavyweight model) the task should be routed.
"""
from typing import Literal
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

class RouteDecision(BaseModel):
    """Data schema for the Router's decision."""
    route: Literal["general_chat", "research", "coding"] = Field(
        ...,
        description="The selected route based on user input."
    )

def get_routing_prompt() -> PromptTemplate:
    """
    Contains the core rules the AI will use when making a routing decision.
    Operates based on context and chat history rules defined by the user.
    """
    template = """You are an expert router (Master Router). Your task is to read the user's input and chat context, then route it to the most appropriate department.
    DO NOT make superficial decisions based merely on keywords; understand the intent by seriously considering the context and the user's previous messages (chat history).
    
    RULES:
    1. CODING (coding): If the user wants to write code, create files, resolve a bug/error message, or build a software structure, route to this department.
    2. GENERAL CHAT (general_chat): If the user is just asking about logic, brainstorming, or having a casual dialogue, route to this department.
    3. RESEARCH (research): If the user's question requires up-to-date information, external data retrieval, or specific research, route to this department.
    
    Chat History:
    {chat_history}
    
    User Input: {user_input}
    """
    return PromptTemplate.from_template(template)
