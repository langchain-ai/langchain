# -----------------------------------------------------------------------------
# Copyright (c) 2023 - , Oracle and/or its affiliates.
# -----------------------------------------------------------------------------
# Authors:
#   Sudhir Kumar (sudhirkk)
# 
# -----------------------------------------------------------------------------
# test_oracleds.py
# -----------------------------------------------------------------------------
import sys
import json
import configparser
import oracledb

from langchain_community.document_loaders.oracleai import (
    OracleDocLoader,
    OracleTextSplitter,
)
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.utilities.oracleai import OracleSummary

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores.oraclevs import (
    drop_table_purge,
    _table_exists,
)

import pytest

uname = ""
passwd = ""
# uname = "LANGCHAINUSER"
# passwd = "langchainuser"
v_dsn = ""

### Test loader #####
@pytest.mark.requires("oracledb")
def test_loader_test() -> None:

    try:
        # oracle connection
        connection = oracledb.connect(user=uname, password=passwd, dsn=v_dsn)
        cursor = connection.cursor()

        if _table_exists(connection, "LANGCHAIN_DEMO"):
            drop_table_purge(connection, "LANGCHAIN_DEMO")

        cursor.execute("CREATE TABLE langchain_demo(id number, text varchar2(25))")

        rows = [ (1, "First" ), (2, "Second" ),
         (3, "Third" ), (4, "Fourth" ),
         (5, "Fifth" ), (6, "Sixth" ),
         (7, "Seventh" ) ]

        cursor.executemany("insert into LANGCHAIN_DEMO(id, text) values (:1, :2)", rows)

        connection.commit()

        # local file, local directory, database  column
        loader_params = { "owner" : uname, "tablename":"LANGCHAIN_DEMO", "colname":"TEXT"}

        # instantiate
        loader = OracleDocLoader(conn = connection, params = loader_params)

        # load 
        docs = loader.load()

        # verify 
        print(f"Number of docs loaded: {len(docs)}")
        if _table_exists(connection, "LANGCHAIN_DEMO"):
            drop_table_purge(connection, "LANGCHAIN_DEMO")

    except Exception as e:
        sys.exit(1)

    try:
        # expectation : ORA-00942
        loader_params = {"owner" : uname, "tablename":"COUNTRIES1", "colname":"COUNTRY_NAME"}

        # instantiate
        loader = OracleDocLoader(conn = connection, params = loader_params)

        # load
        docs = loader.load()

    except Exception as e:
        pass

    try:
        # expectation : file "SUDHIR" doesn't exist.
        loader_params = {"file" : "SUDHIR"}

        # instantiate
        loader = OracleDocLoader(conn = connection, params = loader_params)

        # load
        docs = loader.load()

    except Exception as e:
        pass

    try:
        # expectation : path "SUDHIR" doesn't exist.
        loader_params = {"dir" : "SUDHIR"}

        # instantiate
        loader = OracleDocLoader(conn = connection, params = loader_params)

        # load
        docs = loader.load()

    except Exception as e:
        pass

### Test splitter ####
@pytest.mark.requires("oracledb")
def test_splitter_test() -> None:
    try:
        # oracle connection
        connection = oracledb.connect(user=uname, password=passwd, dsn=v_dsn)
        doc = 'Langchain is a wonderful framework to load, split, chunk and embed your data!!'

        # by words , max = 1000
        splitter_params = {"by":"words","max":"1000","overlap":"200","split":"custom", "custom_list":["," ],"extended":"true","normalize":"all"}

        # instantiate
        splitter = OracleTextSplitter(conn=connection, params=splitter_params)

        # generate chunks
        chunks = splitter.split_text(doc)

        # verify 
        print(f"Nummber of chunks: {len(chunks)}")

        # by chars , max = 4000
        splitter_params = {"by":"chars","max":"4000","overlap":"800","split":"NEWLINE","normalize":"all"}

        # instantiate
        splitter = OracleTextSplitter(conn=connection, params=splitter_params)

        # generate chunks
        chunks = splitter.split_text(doc)

        # verify 
        print(f"Nummber of chunks: {len(chunks)}")

        # by words , max = 10
        splitter_params = {"by":"words","max":"10","overlap":"2","split":"SENTENCE"}

        # instantiate
        splitter = OracleTextSplitter(conn=connection, params=splitter_params)

        # generate chunks
        chunks = splitter.split_text(doc)

        # verify 
        print(f"Nummber of chunks: {len(chunks)}")

        # by chars , max = 50
        splitter_params = {"by":"chars","max":"50","overlap":"10","split":"SPACE","normalize":"all"}

        # instantiate
        splitter = OracleTextSplitter(conn=connection, params=splitter_params)

        # generate chunks
        chunks = splitter.split_text(doc)

        # verify 
        print(f"Nummber of chunks: {len(chunks)}")

    except Exception as e:
        sys.exit(1)

    try:
        # ORA-20003: invalid value xyz for BY parameter
        splitter_params = {"by":"xyz"}

        # instantiate
        splitter = OracleTextSplitter(conn=connection, params=splitter_params)

        # generate chunks
        chunks = splitter.split_text(doc)
    except Exception as e:
        pass        

    try:
        # Expection: ORA-30584: invalid text chunking MAXIMUM - '10'
        splitter_params = {"by":"chars","max":"10","overlap":"2","split":"SPACE","normalize":"all"}

        # instantiate
        splitter = OracleTextSplitter(conn=connection, params=splitter_params)

        # generate chunks
        chunks = splitter.split_text(doc)

    except Exception as e:
        pass

    try:
        # Expection: ORA-30584: invalid text chunking MAXIMUM - '5'
        splitter_params = {"by":"words","max":"5","overlap":"2","split":"SPACE","normalize":"all"}

        # instantiate
        splitter = OracleTextSplitter(conn=connection, params=splitter_params)

        # generate chunks
        chunks = splitter.split_text(doc)

    except Exception as e:
        pass

    try:
        # Expection: ORA-30586: invalid text chunking SPLIT BY - SENTENCE
        splitter_params = {"by":"words","max":"50","overlap":"2","split":"SENTENCE","normalize":"all"}

        # instantiate
        splitter = OracleTextSplitter(conn=connection, params=splitter_params)

        # generate chunks
        chunks = splitter.split_text(doc)

    except Exception as e:
        pass


#### Test summary ####
@pytest.mark.requires("oracledb")
def test_summary_test() -> None:
    try:
        # oracle connection
        connection = oracledb.connect(user=uname, password=passwd, dsn=v_dsn)

       # provider : Database, glevel : Paragraph
        summary_params = {"provider": "database","glevel": "paragraph","numParagraphs": 2,"language": "english"}

        # summary 
        summary = OracleSummary(conn = connection, params = summary_params)

        doc = 'It was 7 minutes after midnight. The dog was lying on the grass in the middle of the lawn in front of Mrs Shears house. Its eyes were closed. It looked as if it was running on its side, the way dogs run when they think they are chasing a cat in a dream. But the dog was not running or asleep. The dog was dead. There was a garden fork sticking out of the dog. The points of the fork must have gone all the way through the dog and into the ground because the fork had not fallen over. I decided that the dog was probably killed with the fork because I could not see any other wounds in the dog and I do not think you would stick a garden fork into a dog after it had died for some other reason, like cancer for example, or a road accident. But I could not be certain about this'
        summaries = summary.get_summary(doc)
        
        # Verify 
        print(f"Number of summaries: {len(summaries)}")

        # provider : Database, glevel : Sentence
        summary_params = {"provider": "database","glevel": "Sentence"}

        # summary 
        summary = OracleSummary(conn = connection, params = summary_params)
        summaries = summary.get_summary(doc)
        
        # Verify 
        print(f"Number of summaries: {len(summaries)}")

        # provider : Database, glevel : P
        summary_params = {"provider": "database","glevel": "P"}

        # summary 
        summary = OracleSummary(conn = connection, params = summary_params)
        summaries = summary.get_summary(doc)
        
        # Verify 
        print(f"Number of summaries: {len(summaries)}")

        # provider : Database, glevel : S 
        summary_params = {"provider": "database","glevel": "S","numParagraphs": 16,"language": "english"}

        # summary 
        summary = OracleSummary(conn = connection, params = summary_params)
        summaries = summary.get_summary(doc)
        
        # Verify 
        print(f"Number of summaries: {len(summaries)}")

        # provider : Database, glevel : S, doc = ' '
        summary_params = {"provider": "database","glevel": "S","numParagraphs": 2}

        # summary 
        summary = OracleSummary(conn = connection, params = summary_params)

        doc = ' '
        summaries = summary.get_summary(doc)

        # Verify 
        print(f"Number of summaries: {len(summaries)}")

    except Exception as e:
        sys.exit(1)

    try:
       # Expection : DRG-11002: missing value for PROVIDER
        summary_params = {"provider": "database1","glevel": "S"}

        # summary 
        summary = OracleSummary(conn = connection, params = summary_params)
        summaries = summary.get_summary(doc)

    except Exception as e:
        pass

    try:
       # Expectation : DRG-11425: gist level SUDHIR is invalid,
       #               DRG-11427: valid gist level values are S, P
        summary_params = {"provider": "database","glevel": "SUDHIR"}

        # summary 
        summary = OracleSummary(conn = connection, params = summary_params)
        summaries = summary.get_summary(doc)

    except Exception as e:
        pass

    try:
       # Expectation : DRG-11441: gist numParagraphs -2 is invalid
        summary_params = {"provider": "database","glevel": "S","numParagraphs": -2}

        # summary 
        summary = OracleSummary(conn = connection, params = summary_params)
        summaries = summary.get_summary(doc)

    except Exception as e:
        pass