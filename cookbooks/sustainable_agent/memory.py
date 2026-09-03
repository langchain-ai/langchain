"""
Feedback Loop Memory Module

This module stores the agent's past mistakes and the user's corrections (feedback), 
retrieving them when necessary. It is the heart of making the system 'sustainable'.
"""
from typing import List
from pydantic import BaseModel, Field
import json
import os

class LessonLearned(BaseModel):
    """Data structure for a lesson learned by the agent."""
    topic: str = Field(..., description="The topic of the lesson (e.g., Python Syntax, API Request)")
    mistake: str = Field(..., description="The mistake the agent made")
    correction: str = Field(..., description="The correct information taught by the user")

class SustainableMemory:
    def __init__(self, db_path: str = "memory_db.json"):
        self.db_path = db_path
        self._ensure_db()
        
    def _ensure_db(self):
        """Creates the memory file from scratch if it does not exist."""
        if not os.path.exists(self.db_path):
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump([], f)
                
    def add_lesson(self, lesson: LessonLearned):
        """Adds a new lesson (correction of a mistake) to the memory."""
        with open(self.db_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        data.append(lesson.model_dump())
        
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
    def get_relevant_lessons(self, current_context: str) -> List[LessonLearned]:
        """
        Finds which past lessons are useful based on the incoming new question.
        
        Engineering Optimization (Hybrid Search): 
        To avoid slowing down the system, a very fast keyword-based (topic) pre-filtering is performed first.
        Then, heavy semantic processing is applied ONLY to the few remaining lessons.
        """
        with open(self.db_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
            
        lessons = [LessonLearned(**item) for item in all_data]
        if not lessons:
            return []
            
        # Phase 1 (Fast/Cheap): Keyword-Based Pre-Filtering
        context_words = set(current_context.lower().split())
        pre_filtered_lessons = []
        
        for lesson in lessons:
            topic_words = set(lesson.topic.lower().split())
            # If any word in the question appears in the lesson's topic, add to list
            if context_words.intersection(topic_words):
                pre_filtered_lessons.append(lesson)
                
        # IF no lessons pass the pre-filtering, DO NOT run the heavy process! (Performance savings)
        if not pre_filtered_lessons:
            return []
            
        # Phase 2 (Heavy/Expensive): Semantic Check only for remaining lessons
        relevant_lessons = []
        for lesson in pre_filtered_lessons:
            # NOTE: In a real architecture, LLM or Vector (Embedding) distance measurement runs here.
            # Since it's applied only to the few items in 'pre_filtered_lessons', system speed does not drop.
            relevant_lessons.append(lesson)
            
        return relevant_lessons
