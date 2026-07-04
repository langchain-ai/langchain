from langchain_core.messages import HumanMessage  
from langchain_core.language_models._utils import _normalize_messages  
  
msg = HumanMessage(content=[  
    {  
        "type": "file",  
        "file": {  
            "filename": "test.csv",  
            "file_data": "data:text/csv;base64,aGVsbG8=",  
        },  
    },  
])  
# This is the line that currently returns 'application/pdf' incorrectly
print(f"MIME Type: {_normalize_messages([msg])[0].content[0]['mime_type']}")