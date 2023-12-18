from langchain_core.example_selectors.length_based import LengthBasedExampleSelector
from langchain_core.prompts import ChatPromptTemplate
import random
from typing import Dict, List

class RandomLenghtExampleSelector(LengthBasedExampleSelector):
    example_prompt: ChatPromptTemplate
    min_remaining: int = 30

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the input lengths."""
        inputs = " ".join(input_variables.values())
        remaining_length = self.max_length - self.get_text_length(inputs)
        indexes = list(range(0,len(self.examples)))
        random.shuffle(indexes)
        examples = []
        for i in indexes: 
            new_length = remaining_length - self.example_text_lengths[i]
            if new_length < 0:
                continue
            else:
                examples.append(self.examples[i])
                if new_length < self.min_remaining:
                    break
                remaining_length = new_length
        return examples