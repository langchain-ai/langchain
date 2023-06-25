from typing import Any, Iterator, Mapping, Optional
import requests
from bs4 import BeautifulSoup

from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document


class GrobidParser():
    """Loads a PDF with pypdf and chunks at character level."""

    def lazy_parse(self,file_path,xml_data,segment_sentences) -> Iterator[Document]:
      soup = BeautifulSoup(xml_data, 'xml')
      sections = soup.find_all('div')
      title = soup.find_all('title')[0].text
      chunks=[]
      for section in sections:
        sect=section.find('head')
        if sect!=None:
          for i, paragraph in enumerate(section.find_all('p')):
            chunk_bboxes=[]
            paragraph_text=[]
            for i,sentence in enumerate(paragraph.find_all('s')):
                paragraph_text.append(sentence.text)
                sbboxes=[]
                for bbox in sentence.get("coords").split(";"):
                  box=bbox.split(",")
                  sbboxes.append({'page':box[0],'x':box[1],'y':box[2],'h':box[3],'w':box[4]})
                chunk_bboxes.append(sbboxes)
                if segment_sentences==True:
                  fpage, lpage = sbboxes[0]['page'], sbboxes[-1]['page']
                  sentence_dict= {'text':sentence.text,
                          'para':str(i),
                          'bboxes':[sbboxes],
                          'section_title':sect.text,
                          'section_number':sect.get('n'),
                          'pages':(fpage,lpage),
                        }
                  chunks.append(sentence_dict)
            if segment_sentences!=True:
                fpage, lpage = chunk_bboxes[0][0]['page'], chunk_bboxes[-1][-1]['page']
                paragraph_dict= {'text':"".join(paragraph_text),
                          'para':str(i),
                          'bboxes':chunk_bboxes,
                          'section_title':sect.text,
                          'section_number':sect.get('n'),
                          'pages':(fpage,lpage),
                        }
                chunks.append(paragraph_dict)


      yield from [
        Document(page_content=chunk['text'],
                metadata=dict(
                  {'text':str(chunk['text']),
                  'para':str(chunk['para']),
                  'bboxes':str(chunk['bboxes']),
                  'pages':str(chunk['pages']),
                  'section_title':str(chunk['section_title']),
                  'section_number':str(chunk['section_number']),
                  'paper_title':str(title),
                  'file_path':str(file_path),
                  })
                )
                for chunk in chunks
      ]
