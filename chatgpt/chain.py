import requests
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from creds import GEMINI_API_KEY

# --- IMPORTANT: Replace with your actual Gemini API Key ---
# If you have a creds.py file, ensure it's in the same directory or adjust the import path.
# For example: from creds import GEMINI_API_KEY

# --- Gemini model setup ---
gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY
)

# --- API calling tool functions ---
def call_local_api(name: str, payload: dict) -> str:
    """
    Calls a local API endpoint with the given name and payload.
    Handles JSON encoding/decoding and error reporting.
    """
    api_url = f"http://127.0.0.1:8000/{name}"
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        return json.dumps(response.json())
    except requests.exceptions.RequestException as e:
        # Catch specific requests exceptions for network or HTTP errors
        return f"API request error for {name}: {str(e)}"
    except json.JSONDecodeError:
        # Handle cases where the response is not valid JSON
        return f"API response error for {name}: Invalid JSON received."
    except Exception as e:
        # Catch any other unexpected errors
        return f"An unexpected error occurred calling {name} API: {str(e)}"

def match_ingredients_tool(args) -> str:
    """
    Matches a list of ingredient names to known food IDs using a local API.
    Expects a JSON string or dictionary with an 'ingredients' key.
    """
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return "Invalid input format for match_ingredients. Expected JSON with 'ingredients' field."

    ingredients = args.get("ingredients", [])
    if not isinstance(ingredients, list):
        return "Invalid 'ingredients' format. Expected a list of strings."
    if not ingredients:
        return "No ingredients provided for matching."

    return call_local_api("match_ingredients", {"ingredients": ingredients})

def get_nutrients_tool(args) -> str:
    """
    Returns summed nutrient data for a list of foods with quantities using a local API.
    Expects a JSON string or dictionary with a 'foods' key.
    """
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return "Invalid input format for get_nutrients. Expected JSON with 'foods' field."

    foods = args.get("foods", [])
    if not isinstance(foods, list):
        return "Invalid 'foods' format. Expected a list of food objects."
    if not foods:
        return "No foods provided for nutrient calculation."

    # Basic validation for food items structure
    for food_item in foods:
        if not isinstance(food_item, dict) or "food_id" not in food_item or "quantity_g" not in food_item:
            return "Invalid food item structure. Each food must have 'food_id' (str) and 'quantity_g' (number)."

    return call_local_api("get_nutrients", {"foods": foods})

# --- LangChain tool wrappers ---
match_tool = Tool(
    name="match_ingredients",
    func=match_ingredients_tool,
    description=(
        "Matches a list of ingredient names (e.g., 'almond', 'rice') to known food IDs in the database. "
        "Input: {\"ingredients\": [list of ingredient names]}. "
        "Output: list of food IDs and names."
    )
)

nutrient_tool = Tool(
    name="get_nutrients",
    func=get_nutrients_tool,
    description=(
        "Return summed nutrient data for a list of foods with quantities. "
        "Input: {\"foods\": [{\"food_id\": str, \"quantity_g\": number}]}. "
        "Output: total calories, protein, fat, carbs."
    )
)

# --- LangGraph agent ---
# Create a ReAct agent using the Gemini model and the defined tools.
# FIXED: Use create_react_agent directly since it already returns a CompiledStateGraph
agent_executor = create_react_agent(gemini, [match_tool, nutrient_tool])

# --- Run agent query ---
query = """
Create a meal plan for 2000 kcal, 150g protein.
You MUST use match_ingredients to get food IDs for every ingredient you propose.
You MUST use get_nutrients to compute total nutrients for the meal plan.
Never estimate yourself.
"""

print("Invoking agent with query...")
try:
    # FIXED: Use proper message format with tuple ('user', 'content') instead of HumanMessage objects
    response = agent_executor.invoke({"messages": [("user", query)]})
    print("
    print(response)
except Exception as e:
    print(f"

