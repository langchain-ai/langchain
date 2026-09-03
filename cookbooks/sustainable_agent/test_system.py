from memory import SustainableMemory, LessonLearned
from agent import SustainableAgent

def run_test():
    print("1. Initializing Memory...")
    memory = SustainableMemory()
    
    # Intentionally add a bug fix (lesson) to the system
    print("2. Adding the correction of a past mistake to memory...")
    lesson = LessonLearned(
        topic="Python File Reading",
        mistake="Forgetting to close the file by only opening with 'open(file)'.",
        correction="You must always use the 'with open(file) as f:' block. Never forget to close the file!"
    )
    memory.add_lesson(lesson)
    
    print("\n3. Initializing Orchestrator and asking a new question...")
    agent = SustainableAgent()
    
    # The user asks a new question (contains words 'file' and 'Python', triggering memory!)
    # Also contains 'code' and 'bug', which should route to the "coding" room based on rules!
    chat_history = "User: Hello!\nAgent: Hello, how can I help you?"
    user_input = "Could you write code that reads a file with Python? Please make sure there are no bugs."
    
    print("User's Question:", user_input)
    agent.process_request(chat_history=chat_history, user_input=user_input)

if __name__ == "__main__":
    run_test()
