import importlib
import os
import uuid
from typing import List

import pinecone
import pytest

from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.rockset import Rockset
