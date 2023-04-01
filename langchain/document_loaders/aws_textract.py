import os
from io import BytesIO
from typing import Any, List, Optional
from PIL import Image
from langchain.docstore.document import Document
from langchain.utils import get_from_dict_or_env




class AwsTextractExtraction:


    def __init__(self,aws_region_name,aws_secret_key,aws_access_key,file_path):
   

          self.aws_region_name  =  aws_region_name
          self.aws_secret_key=  aws_secret_key
          self.aws_access_key =  aws_access_key
          self.file_path = file_path
          try:
            import boto3

          except ImportError:
                raise ValueError(
                    "Could not import aws boto3 package. "
                    "Please install it with `pip install boto3`."
                )



    def get_text_from_image(self)-> List[Document]:
         output=[] 
         page_no=0
         
         session = boto3.Session(
            region_name=self.aws_region_name,
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key
            )
         textract_client =  session.client('textract')
         Image=  Image.open(self.file_path)
         buf = BytesIO()
         Image.save(buf, format='PNG')
         image_bytes = buf.getvalue()
        

         response=  textract_client.detect_document_text(Document={'Bytes': image_bytes})
         detected_txt =''
         for item in response["Blocks"]:
              if  item['BlockType'] == 'LINE':
                   detected_txt  += item['Text'] + '\n'
         
         metadata = {"source": self.file_path,"page":page_no}

         output.append(Document(page_content=detected_txt  , metadata=metadata))  
         return output

   

