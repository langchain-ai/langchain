"""Test Oracle AI Vector Search functionality."""
# import required modules
import sys
import oracledb
import pandas as pd
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.oraclevs import _table_exists
from langchain_community.vectorstores.oraclevs import _create_table
from langchain_community.vectorstores.oraclevs import create_index
from langchain_community.vectorstores.oraclevs import drop_table_purge
from langchain_community.vectorstores.oraclevs import drop_index_if_exists
import threading
import sys 
import pytest

username = 'vector'
password = 'vector'
dsn = '152.67.235.198:1521/orclpdb1'

############################
####### table_exists #######
############################
@pytest.mark.requires("oracledb")
def test_table_exists_test() -> None:

    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
        print("Connection successful!")
    except Exception as e:
        print("Connection failed!")
        sys.exit(1)
    # 1. Existing Table:(all capital letters)
    # expectation:True
    print(_table_exists(connection,"V$TRANSACTION"))

    # 2. Existing Table:(all small letters)
    # expectation:True
    print(_table_exists(connection,"v$transaction"))

    # 3. Non-Existing Table
    # expectation:false
    print(_table_exists(connection,"Hello"))

    # 4. Invalid Table Name
    # Expectation:ORA-00903: invalid table name
    try:
        print(_table_exists(connection,'123'))
    except Exception as e:
        print(e)

    # 5. Empty String 
    # Expectation:ORA-00903: invalid table name
    try:
        print(_table_exists(connection,''))
    except Exception as e:
        print(e)

    # 6. Special Character
    # Expectation:ORA-00911: #: invalid character after FROM
    try:
        print(_table_exists(connection,'##4'))
    except Exception as e:
        print(e)

    # 7. Table name length > 128
    # Expectation:ORA-00972: The identifier XXXXXXXXXX...XXXXXXXXXX... 
    # exceeds the maximum length of 128 bytes.
    try:
        print(_table_exists(connection,'x'*129))
    except Exception as e:
        print(e)

    # 8. <Schema_Name.Table_Name>
    # Expectation:True
    _create_table(connection,'TB1',65535)

    # 9. Toggle Case (like TaBlE)
    # Expectation:True
    print(_table_exists(connection,'Tb1'))
    drop_table_purge(connection,'TB1')

    # 10. Table_Name→ "हिन्दी"
    # Expectation:True
    _create_table(connection,'"हिन्दी"',545)
    print(_table_exists(connection,'"हिन्दी"'))
    drop_table_purge(connection,'"हिन्दी"')


############################
####### create_table #######
############################

@pytest.mark.requires("oracledb")
def test_create_table_test() -> None:

    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
        print("Connection successful!")
    except Exception as e:
        print("Connection failed!")
        sys.exit(1)

    # 1. New table - HELLO
    #    Dimension - 100
    # Expectation:table is created
    _create_table(connection,'HELLO',100)
    
    # 2. Existing table name
    #    HELLO
    #    Dimension - 110
    # Expectation:Nothing happens
    _create_table(connection,'HELLO',110)
    drop_table_purge(connection,'HELLO')
    
    # 3. New Table - 123
    #    Dimension - 100
    # Expectation:ORA-00903: invalid table name
    try:
        _create_table(connection,'123',100)
        drop_table_purge(connection,'123')
    except Exception as e:
        print(e)
    
    # 4. New Table - Hello123
    #    Dimension - 65535
    # Expectation:table is created
    _create_table(connection,'Hello123',65535)
    drop_table_purge(connection,'Hello123')
    
    # 5. New Table - T1
    #    Dimension - 65536
    # Expectation:ORA-51801: VECTOR column type specification 
    # has an unsupported dimension count ('65536').
    try:
        _create_table(connection,'T1',65536)
        drop_table_purge(connection,'T1')
    except Exception as e:
        print(e)
    
    # 6. New Table - T1
    #    Dimension - 0
    # Expectation:ORA-51801: VECTOR column type specification has 
    # an unsupported dimension count (0).
    try:
        _create_table(connection,'T1',0)
        drop_table_purge(connection,'T1')
    except Exception as e:
        print(e)
    
    # 7. New Table - T1
    #    Dimension - -1
    # Expectation:ORA-51801: VECTOR column type specification has 
    # an unsupported dimension count ('-').
    try:
        _create_table(connection,'T1',-1)
        drop_table_purge(connection,'T1')
    except Exception as e:
        print(e)
    
    # 8. New Table - T1
    #    Dimension - 2.2
    # Expectation:ORA-02017: integer value required
    try:
        _create_table(connection,'T1',2.2)
        drop_table_purge(connection,'T1')
    except Exception as e:
        print(e)
    
    # 9. New Table - T1
    #    Dimension - '2'
    _create_table(connection,'T1','2')
    drop_table_purge(connection,'T1')
    
    # 10. New Table - T2
    #     Dimension - T2
    # Expectation:table is created
    try:
        _create_table(connection,'T2','T2')
        drop_table_purge(connection,'T2')
    except Exception as e:
        print(e)
    
    # 11. New Table - T2
    #     Dimension - '1000'
    # Expectation:table is created
    _create_table(connection,'T2',int('1000'))
    drop_table_purge(connection,'T2')
    
    # 12. New Table - T3
    #     Dimension - 100 passed as a variable
    # Expectation:table is created
    val=100
    _create_table(connection,'T3',val)
    drop_table_purge(connection,'T3')
    
    # 13. 
    # Expectation:ORA-00922: missing or invalid option
    val="""H
    ello"""
    try:
        _create_table(connection,val,545)
        drop_table_purge(connection,val)
    except Exception as e:
        print(e)
    
    # 14. New Table - हिन्दी
    #     Dimension - 545
    # Expectation:table is created
    _create_table(connection,'"हिन्दी"',545)
    drop_table_purge(connection,'"हिन्दी"')
    
    # 15. <schema_name.table_name>
    # Expectation:failure - user does not exist
    try:
        _create_table(connection,"U1.TB4",128)
        drop_table_purge(connection,'U1.TB4')
    except Exception as e:
        print(e)
    
    # 16. 
    # Expectation:table is created  
    _create_table(connection,'"T5"',128)
    drop_table_purge(connection,'"T5"')
    
    # 17. Toggle Case
    # Expectation:table creation fails
    try:
        _create_table(connection,'TaBlE',128)
        drop_table_purge(connection,'TaBlE')
    except Exception as e:
        print(e)
    
    # 18. table_name as empty_string
    # Expectation: ORA-00903: invalid table name
    try:
        _create_table(connection,'',128)
        drop_table_purge(connection,'')
        _create_table(connection,'""',128)
        drop_table_purge(connection,'""')
    except Exception as e:
        print(e)
    
    # 19. Dimension='*'
    # Expectation:table is created
    _create_table(connection,'T7','*')
    drop_table_purge(connection,'T7')
    
    # 20. 'NULL' passed as table_name
    # Expectation: ORA-00903: invalid table name
    try:
        _create_table(connection,'NULL','*')
        drop_table_purge(connection,'NULL')
    except Exception as e:
        print(e)
    
    # 21. Reserved keywords as table_name
    # Expectation: ORA-00903: invalid table name
    try:
        _create_table(connection,'ALTER','*')
        drop_table_purge(connection,'ALTER')
    except Exception as e:
        print(e)
    
    # 22. Arithmetic Operations in dimension parameter
    # Expectation:table is created
    n=1
    _create_table(connection,'T10',n+500)
    drop_table_purge(connection,'T10')
    
    # 23. String Operations in table_name&dimension parameter
    # Expectation:table is created
    _create_table(connection,'YaSh'.replace('aS','ok'),500)
    drop_table_purge(connection,'YaSh'.replace('aS','ok'))
    

##################################
####### create_hnsw_index #######
##################################

@pytest.mark.requires("oracledb")
def test_create_hnsw_index_test() -> None:

    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
        print("Connection successful!")
    except Exception as e:
        print("Connection failed!")
        sys.exit(1)
    # 1. Table_name - TB1
    #    New Index
    #    distance_strategy - DistanceStrategy.Dot_product
    # Expectation:Index created
    model1 = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-mpnet-base-v2")
    vs=OracleVS(connection,model1,'TB1',DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(connection,vs)

    # 2. Creating same index again
    #    Table_name - TB1
    # Expectation:Nothing happens
    create_index(connection,vs)
    drop_index_if_exists(connection,'HNSW')

    # 3. Create index with following parameters:
    #    idx_name - hnsw_idx2
    #    idx_type - HNSW
    # Expectation:Index created
    create_index(connection,vs,params={"idx_name": "hnsw_idx2", "idx_type": "HNSW"})
    drop_index_if_exists(connection,'hnsw_idx2')

    # 4. Table Name - TB1
    #    idx_name - "हिन्दी"
    #    idx_type - HNSW
    # Expectation:Index created
    create_index(connection,vs,params={"idx_name": '"हिन्दी"', "idx_type": "HNSW"})
    drop_index_if_exists(connection,'"हिन्दी"')

    # 5. idx_name passed empty
    # Expectation:ORA-01741: illegal zero-length identifier
    create_index(connection,vs,params={"idx_name": '""', "idx_type": "HNSW"})
    drop_index_if_exists(connection,'""')

    # 6. idx_type left empty
    # Expectation:Index created
    create_index(connection,vs,params={"idx_name": 'Hello', "idx_type": ""})
    drop_index_if_exists(connection,'Hello')

    # 7. Index_name is enclosed in ""
    # Expectation:Index created
    create_index(connection,vs,params={"idx_name": '"Hello"', "idx_type": ""})
    drop_index_if_exists(connection,'"Hello"')

    # 8. multiple values to same parameter 
    # Expectation:Index created
    create_index(connection,vs,params={"idx_name": 'Hello', "idx_name":'Hello1', "idx_type":'HNSW'})
    drop_index_if_exists(connection,'Hello')
    drop_index_if_exists(connection,'Hello1')

    # 9. efconstruction passed as parameter but not neighbours
    # Expectation:Index created
    create_index(connection,vs,params={"idx_name":"idx11","efConstruction":100,"idx_type":'HNSW'})
    drop_index_if_exists(connection,'idx11')

    # 10. efconstruction passed as parameter as well as neighbours
    # (for this idx_type parameter is also necessary)
    # Expectation:Index created
    create_index(connection,vs,params={"idx_name":"idx11","efConstruction":100,"neighbors":80,"idx_type":'HNSW'})
    drop_index_if_exists(connection,'idx11')

    # 11. Limit of Values for(integer values):
    #     parallel
    #     efConstruction
    #     Neighbors
    #     Accuracy
    #     0<Accuracy<=100
    #     0<Neighbour<=2048
    #     0<efConstruction<=65535
    #     0<parallel<=255
    # Expectation:Index created
    create_index(connection,vs,params={"idx_name":"idx11","efConstruction":65535,"neighbors":2048,"idx_type":'HNSW',"parallel":255})
    drop_index_if_exists(connection,'idx11')
    create_index(connection,vs,params={"idx_name":"idx11","efConstruction":0,"neighbors":2048,"idx_type":'HNSW',"parallel":255})
    drop_index_if_exists(connection,'idx11')
    create_index(connection,vs,params={"idx_name":"idx11","efConstruction":100,"neighbors":0,"idx_type":'HNSW',"parallel":255})
    drop_index_if_exists(connection,'idx11')
    create_index(connection,vs,params={"idx_name":"idx11","efConstruction":100,"neighbors":100,"idx_type":'HNSW',"parallel":0})
    drop_index_if_exists(connection,'idx11')
    create_index(connection,vs,params={"idx_name":"idx11","efConstruction":10,"neighbors":100,"idx_type":'HNSW',"parallel":10,"accuracy":120})
    drop_index_if_exists(connection,'idx11')
    # with negative values/out-of-bound values for all 4 of them, we get the same errors
    # Expectation:Index not created
    create_index(connection,vs,params={"idx_name":"idx11","efConstruction":200,"neighbors":100,"idx_type":'HNSW',"parallel":"hello","accuracy":10})
    drop_index_if_exists(connection,'idx11')

    # 12. Table_name as <schema_name.table_name>
    # Expectation:Index created
    create_index(connection,vs,params={"idx_name":"idx11","efConstruction":200,"neighbors":100,"idx_type":'HNSW',"parallel":8,"accuracy":10})
    drop_index_if_exists(connection,'idx11')

    # 13. index_name as <schema_name.index_name>
    # Expectation:Index created
    create_index(connection,vs,params={"idx_name":"U1.idx11","efConstruction":200,"neighbors":100,"idx_type":'HNSW',"parallel":8,"accuracy":10})
    drop_index_if_exists(connection,'U1.idx11')

    # 14. Index_name size >129
    # Expectation:Index not created
    create_index(connection,vs,params={"idx_name":"x"*129,"idx_type":'HNSW'})
    drop_index_if_exists(connection,"x"*129)

    # 15. Index_name size 128
    # Expectation:Index created
    create_index(connection,vs,params={"idx_name":"x"*128,"idx_type":'HNSW'})
    drop_index_if_exists(connection,"x"*128)
    drop_table_purge(connection,'TB1')


##################################
####### index_exists #############
##################################

@pytest.mark.requires("oracledb")
def test_index_exists_test() -> None:

    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
        print("Connection successful!")
    except Exception as e:
        print("Connection failed!")
        sys.exit(1)
    model1=HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-mpnet-base-v2")
    # 1. Existing Index:(all capital letters)
    # Expectation:true
    vs=OracleVS(connection,model1,'TB1',DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(connection,vs,params={"idx_name":"idx11","idx_type":'HNSW'})
    print(index_exists(connection,"IDX11"))

    # 2. Existing Table:(all small letters)
    # Expectation:true
    print(index_exists(connection,"idx11"))

    # 3. Non-Existing Index
    # Expectation:False
    print(index_exists(connection,"Hello"))

    # 4. Invalid Index Name
    # Expectation:Error
    print(index_exists(connection,'123'))

    # 5. Empty String
    # Expectation:Error
    print(index_exists(connection,''))
    print(index_exists(connection,""))

    # 6. Special Character
    # Expectation:Error
    print(index_exists(connection,'##4'))

    # 7. Index name length > 128
    # Expectation:Error
    print(index_exists(connection,'x'*129))

    # 8. <Schema_Name.Index_Name>
    # Expectation:true
    print(index_exists(connection,'U1.IDX11'))

    # 9. Toggle Case (like iDx11)
    # Expectation:true
    print(index_exists(connection,'IdX11'))

    # 10. Index_Name→ "हिन्दी"
    # Expectation:true
    drop_index_if_exists(connection,'idx11')
    create_index(connection,vs,params={"idx_name": '"हिन्दी"', "idx_type": "HNSW"})
    print(index_exists(connection,'"हिन्दी"'))
    drop_table_purge(connection,'TB1')

##################################
####### add_texts ################
##################################

@pytest.mark.requires("oracledb")
def test_add_texts_test() -> None:

    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
        print("Connection successful!")
    except Exception as e:
        print("Connection failed!")
        sys.exit(1)
    # 1. Add 2 records to table
    # Expectation:Successfull
    texts = ["Rohan", "Shailendra"]
    metadata = [{'id': "100", 'link': "Document Example Test 1"}, {'id': "101", 'link': "Document Example Test 2"}]
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs_obj=OracleVS(connection,model,'TB1',DistanceStrategy.EUCLIDEAN_DISTANCE)
    vs_obj.add_texts(texts,metadata)

    # 2. Add record but metadata is not there
    # Expectation:An exception occurred :: Either specify an 'ids' list or 
    # 'metadatas' with an 'id' attribute for each element.
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs_obj=OracleVS(connection,model,'TB2',DistanceStrategy.EUCLIDEAN_DISTANCE)
    texts = ["Sri Ram","Krishna"]
    vs_obj.add_texts(texts)

    # 3. Add record with ids option
    #    ids are passed as int
    #    ids are passed as string
    #    ids are passed as empty string
    #    ids are passed as multi-line string
    #    ids are passed as "<string>"
    # Expectations:
    # An exception occurred :: argument of type 'int' is not iterable
    # Successfull
    # Successfull
    # Successfull
    # Successfull
    ids=[114,124] 
    vs_obj.add_texts(texts,ids)

    ids=["114","124"]
    vs_obj.add_texts(texts,ids=ids)

    ids=["","134"]
    vs_obj.add_texts(texts,ids=ids)

    ids=["""Good afternoon
    my friends""","India"]
    vs_obj.add_texts(texts,ids=ids)

    ids=['"Good afternoon"','"India"']
    vs_obj.add_texts(texts,ids=ids)

    # 4. Add records with ids and metadatas
    # Expectation:Successfull
    texts = ["Sri Ram 6","Krishna 6"]
    ids=['1','2']
    metadata = [{'id': "102", 'link': "Document Example",'stream':'Science'}, {'id': "104", 'link': "Document Example 45"}]
    vs_obj.add_texts(texts,metadata,ids=ids)

    # 5. Add 10000 records
    # Expectation:Successfull
    texts = ["Sri Ram{0}".format(i) for i in range(1,10000)]
    ids=["Hello{0}".format(i) for i in range(1,10000)]
    vs_obj.add_texts(texts,ids=ids)

    # 6. Add 2 different record concurrently
    # Expectation:Successfull
    def add(val:str):
        model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vs_obj=OracleVS(connection,model,'TB3',DistanceStrategy.EUCLIDEAN_DISTANCE)
        texts=[val]
        ids=texts
        vs_obj.add_texts(texts,ids=ids)
    thread_1 = threading.Thread(target=add('Sri Ram'))
    thread_2 = threading.Thread(target=add('Sri Krishna'))
    thread_1.start()
    thread_2.start()
    thread_1.join()
    thread_2.join()

    # 7. Add 2 same record concurrently
    # Expectation:Successfull, For one of the insert,get primary key violation error
    def add(val:str):
        model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vs_obj=OracleVS(connection,model,'TB4',DistanceStrategy.EUCLIDEAN_DISTANCE)
        texts=[val]
        ids=texts
        vs_obj.add_texts(texts,ids=ids)
    thread_1 = threading.Thread(target=add('Sri Ram'))
    thread_2 = threading.Thread(target=add('Sri Ram'))
    thread_1.start()
    thread_2.start()
    thread_1.join()
    thread_2.join()

    # 8. 
    # Expectation:AttributeError: 'NoneType' object has no attribute 'replace'
    texts = [None]
    ids=['None']
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs_obj=OracleVS(connection,model,'TB5',DistanceStrategy.EUCLIDEAN_DISTANCE)
    vs_obj.add_texts(texts,ids=ids)

    # 9.
    # Expectation:Successfull
    texts = ["None"]
    ids=["None"]
    vs_obj.add_texts(texts,'hello',ids=ids)

    # 10. Add 10000 records
    # Expectation:Successfull
    for i in range(1,10000):
        texts = ["Yash{0}".format(i)]
        ids=["1234{0}".format(i)]
        vs_obj.add_texts(texts,ids=ids)

    # 11. create object with table name of type <schema_name.table_name>
    # Expectation:Successfull
    vs_obj=OracleVS(connection,model,'U1.TB6',DistanceStrategy.DOT_PRODUCT)
    for i in range(1,10):
        texts = ["Yash{0}".format(i)]
        ids=["1234{0}".format(i)]
        vs_obj.add_texts(texts,ids=ids)


##################################
####### embed_documents(text) ####
##################################
@pytest.mark.requires("oracledb")
def test_embed_documents_test() -> None:

    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
        print("Connection successful!")
    except Exception as e:
        print("Connection failed!")
        sys.exit(1)
    # 1. String Example-'Sri Ram'
    # Expectation:Vector Printed
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs_obj=OracleVS(connection,model,'TB7',DistanceStrategy.EUCLIDEAN_DISTANCE)
    print(vs_obj.embed_documents('Sri Ram'))
    
    # 2. Number
    # expectation:TypeError: 'int' object is not iterable
    print(vs_obj.embed_documents(123))
    
    # 3. Empty string
    # expectation: []
    print(vs_obj.embed_documents(''))
    
    # 4. List
    # Expectation:Vector Printed
    print(vs_obj.embed_documents(['hello','yash']))
    
    # 5. Dictionary
    # Expectation:AttributeError: 'dict' object has no attribute 'replace'
    print(vs_obj.embed_documents([{'hello':'ok','yash':'ok'},{'hello':'ok1','yash':'ok1'}]))


##################################
####### embed_query(text) ########
##################################
@pytest.mark.requires("oracledb")
def test_embed_query_test() -> None:

    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
        print("Connection successful!")
    except Exception as e:
        print("Connection failed!")
        sys.exit(1)
    # 1. String
    # Expectation:Vector printed
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs_obj=OracleVS(connection,model,'TB8',DistanceStrategy.EUCLIDEAN_DISTANCE)
    print(vs_obj.embed_query('Sri Ram'))
    
    # 2. Number
    # Expectation:TypeError: 'int' object is not iterable
    print(vs_obj.embed_query(123))
    
    # 3. Empty string
    # Expectation:[]
    print(vs_obj.embed_query(''))
    
    # 4. List
    # Expectation:AttributeError: 'list' object has no attribute 'replace'
    print(vs_obj.embed_query(['hello','yash']))
    
    # 5. Dictionary
    # Expectation:AttributeError: 'dict' object has no attribute 'replace'
    print(vs_obj.embed_query([{'hello':'ok','yash':'ok'},{'hello':'ok1','yash':'ok1'}]))


##################################
####### create_index #############
##################################
@pytest.mark.requires("oracledb")
def test_create_index_test() -> None:

    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
        print("Connection successful!")
    except Exception as e:
        print("Connection failed!")
        sys.exit(1)
    # 1. No optional parameters passed
    # Expectation:Successfull
    model1 = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-mpnet-base-v2")
    vs=OracleVS(connection,model1,'TB9',DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(connection,vs)
    drop_index_if_exists(connection,'HNSW')

    # 2. ivf index
    # Expectation:Successfull
    create_index(connection,vs,{'idx_type':'IVF','idx_name':'IVF'})
    drop_index_if_exists(connection,'IVF')

    # 3. ivf index with neighbour_part passed as parameter
    # Expectation:Successfull
    create_index(connection,vs,{'idx_type':'IVF','neighbor_part':10})
    drop_index_if_exists(connection,'IVF')

    # 4. ivf index with neighbour_part and accuracy passed as parameter
    # Expectation:Successfull
    create_index(connection,vs,{'idx_type':'IVF','neighbor_part':10,'accuracy':90})
    drop_index_if_exists(connection,'IVF')

    # 5. ivf index with neighbour_part and parallel passed as parameter
    # Expectation:Successfull
    create_index(connection,vs,{'idx_type':'IVF','neighbor_part':10,'parallel':90})
    drop_index_if_exists(connection,'IVF')

    # 6. ivf index and then perform dml(insert)
    # Expectation:Successfull
    create_index(connection,vs,{'idx_type':'IVF','idx_name':'IVF'})
    texts = ["Sri Ram","Krishna"]
    metadatas=[{'id':"hello"},{'id':"105"}]
    vs.add_texts(texts)
    # perform delete
    vs.delete(["hello"])
    drop_index_if_exists(connection,'IVF')

    # 7. ivf index with neighbour_part,parallel and accuracy passed as parameter
    # Expectation:Successfull
    create_index(connection,vs,{'idx_type':'IVF','neighbor_part':10,'parallel':90,'accuracy':99})
    drop_index_if_exists(connection,'IVF')


##################################
####### perform_search ###########
##################################
@pytest.mark.requires("oracledb")
def test_perform_search_test() -> None:

    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
        print("Connection successful!")
    except Exception as e:
        print("Connection failed!")
        sys.exit(1)
    model1 = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-mpnet-base-v2")
    vs_1=OracleVS(connection,model1,'TB10',DistanceStrategy.EUCLIDEAN_DISTANCE)
    vs_2=OracleVS(connection,model1,'TB11',DistanceStrategy.DOT_PRODUCT)
    vs_3=OracleVS(connection,model1,'TB12',DistanceStrategy.MAX_INNER_PRODUCT)
    vs_4=OracleVS(connection,model1,'TB13',DistanceStrategy.EUCLIDEAN_DISTANCE)
    vs_5=OracleVS(connection,model1,'TB14',DistanceStrategy.DOT_PRODUCT)
    vs_6=OracleVS(connection,model1,'TB15',DistanceStrategy.MAX_INNER_PRODUCT)

    # vector store lists:
    vs_list=[vs_1,vs_2,vs_3,vs_4,vs_5,vs_6]

    for i,vs in enumerate(vs_list,start=1):
        # insert data
        texts=['Yash','Varanasi','Yashaswi','Mumbai','BengaluruYash']
        metadatas=[{'id':"hello"},{'id':"105"},{'id':"106"},{'id':"yash"},{'id':"108"}]

        vs.add_texts(texts,metadatas)

        # create index
        if i == 1 or i == 2 or i == 3:
            create_index(connection,vs,{'idx_type':'HNSW','idx_name':'IDX1{i}'})
        else:
            create_index(connection,vs,{'idx_type':'IVF','idx_name':'IDX1{i}'})

        # perform search
        query="YashB"

        filter={"id":"106","id":"108","id":"yash"}

        # similarity_searh without filter
        print(vs.similarity_search(query,2))

        # similarity_searh with filter
        print(vs.similarity_search(query,2,filter=filter))

        # Similarity search with relevance score
        print(vs.similarity_search_with_relevance_score(query, 2))

        # Similarity search with relevance score with filter
        print(vs.similarity_search_with_relevance_score(query, 2, filter=filter))

        # Max marginal relevance search
        print(vs.max_marginal_relevance_search(query, 2, fetch_k=20, lambda_mult=0.5))

        # Max marginal relevance search with filter
        print(vs.max_marginal_relevance_search(query, 2, fetch_k=20, lambda_mult=0.5, filter=filter))