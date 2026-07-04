from langchain_core.messages import HumanMessage  
from langchain_core.language_models._utils import _normalize_messages  

def test_mime_preservation():
    print("Testing MIME type preservation...")
    msg = HumanMessage(content=[  
        {  
            "type": "file",  
            "file": {  
                "filename": "test.csv",  
                "file_data": "data:text/csv;base64,aGVsbG8=",  
            },  
        },  
    ])  
    # This is the logic that you fixed
    normalized = _normalize_messages([msg])[0].content[0]
    mime = normalized.get('mime_type')
    print(f"Result: {mime}")
    
    if mime == "text/csv":
        print("✅ SUCCESS: MIME type preserved!")
    else:
        print(f"❌ FAILED: Expected text/csv, but got {mime}")

def test_openai_guard():
    print("\nTesting OpenAI Guard...")
    from langchain_core.messages.block_translators.openai import convert_to_openai_data_block
    
    block = {
        "type": "file",
        "mime_type": "text/csv",
        "base64": "aGVsbG8=",
        "filename": "test.csv"
    }
    
    try:
        convert_to_openai_data_block(block, api="chat/completions")
        print("❌ FAILED: Should have raised ValueError for CSV in Chat Completions")
    except ValueError as e:
        print(f"✅ SUCCESS: Guard caught invalid file: {e}")

if __name__ == "__main__":
    test_mime_preservation()
    test_openai_guard()