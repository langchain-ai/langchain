import boto3
import os
from io import BytesIO
from PIL import Image
from langchain.docstore.document import Document
from langchain.utils import get_from_dict_or_env




class AwsTextractExtraction:


    def __init__(self, aws_secret_key, aws_access_key, aws_region_name,file_path):
   

          self.aws_region_name  =  get_from_dict_or_env( key="aws_region_name", env_key="AWS_REGION")
          self.aws_secret_key=  get_from_dict_or_env( key="aws_secret_key", env_key="AWS_SECRET_KEY")
          self.aws_access_key =  get_from_dict_or_env( key="aws_access_key", env_key="AWS_ACCESS_KEY")
          self.file_path = file_path



    def get_text_from_pdf(self)-> List[str]:
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

   

