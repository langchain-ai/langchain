from langchain_core.messages import HumanMessage, AIMessage
from unittest.mock import MagicMock
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables.tiered_router import TieredSemanticRouter

# 1. Setup Mocks to act as our "Small" and "Large" models
small_model = MagicMock(spec=BaseChatModel)
small_model.invoke.return_value = AIMessage(content="[GPT-4o-mini]: 2+2 is 4.")

large_model = MagicMock(spec=BaseChatModel)
large_model.invoke.return_value = AIMessage(content="[GPT-4o]: Microservices offer scalability but increase operational complexity...")

# 2. Initialize the Router
router = TieredSemanticRouter(
    primary=small_model, 
    fallback=large_model, 
    threshold=0.6
)

def run_demo(user_input: str):
    print(f"\n{'='*60}")
    print(f"USER INPUT: {user_input}")
    
    # Calculate score manually for demo visibility
    score = router._get_complexity_score(user_input)
    print(f"COMPLEXITY SCORE: {score:.2f} (Threshold: {router.threshold})")
    
    # Run the router
    response = router.invoke(user_input)
    
    if score <= router.threshold:
        print(f"DECISION: ✅ ROUTED TO PRIMARY (CHEAP)")
    else:
        print(f"DECISION: 🔥 PROMOTED TO FALLBACK (EXPENSIVE)")
        
    print(f"FINAL RESPONSE: {response.content}")

# --- THE COMPARISON ---

# Test A: Simple Message
run_demo("What is 2+2?")

# Test B: Complex Message (Longer + keyword 'analyze')
run_demo("""
Analyze the architectural trade-offs between microservices and monoliths 
for a high-traffic e-commerce application. Focus on database consistency, 
latency, and deployment overhead.
""")