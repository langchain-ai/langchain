"""Test Oracle AI Vector Search functionality."""

# import required modules
import sys
import threading

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.oraclevs import (
    OracleVS,
    _create_table,
    _index_exists,
    _table_exists,
    create_index,
    drop_index_if_exists,
    drop_table_purge,
)
from langchain_community.vectorstores.utils import DistanceStrategy

username = ""
password = ""
dsn = ""


############################
####### table_exists #######
############################
def test_table_exists_test() -> None:
    try:
        import oracledb
    except ImportError:
        return

    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception:
        sys.exit(1)
    # 1. Existing Table:(all capital letters)
    # expectation:True
    _table_exists(connection, "V$TRANSACTION")

    # 2. Existing Table:(all small letters)
    # expectation:True
    _table_exists(connection, "v$transaction")

    # 3. Non-Existing Table
    # expectation:false
    _table_exists(connection, "Hello")

    # 4. Invalid Table Name
    # Expectation:ORA-00903: invalid table name
    try:
        _table_exists(connection, "123")
    except Exception:
        pass

    # 5. Empty String
    # Expectation:ORA-00903: invalid table name
    try:
        _table_exists(connection, "")
    except Exception:
        pass

    # 6. Special Character
    # Expectation:ORA-00911: #: invalid character after FROM
    try:
        _table_exists(connection, "##4")
    except Exception:
        pass

    # 7. Table name length > 128
    # Expectation:ORA-00972: The identifier XXXXXXXXXX...XXXXXXXXXX...
    # exceeds the maximum length of 128 bytes.
    try:
        _table_exists(connection, "x" * 129)
    except Exception:
        pass

    # 8. <Schema_Name.Table_Name>
    # Expectation:True
    _create_table(connection, "TB1", 65535)

    # 9. Toggle Case (like TaBlE)
    # Expectation:True
    _table_exists(connection, "Tb1")
    drop_table_purge(connection, "TB1")

    # 10. Table_Name→ "हिन्दी"
    # Expectation:True
    _create_table(connection, '"हिन्दी"', 545)
    _table_exists(connection, '"हिन्दी"')
    drop_table_purge(connection, '"हिन्दी"')


############################
####### create_table #######
############################


def test_create_table_test() -> None:
    try:
        import oracledb
    except ImportError:
        return

    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception:
        sys.exit(1)

    # 1. New table - HELLO
    #    Dimension - 100
    # Expectation:table is created
    _create_table(connection, "HELLO", 100)

    # 2. Existing table name
    #    HELLO
    #    Dimension - 110
    # Expectation:Nothing happens
    _create_table(connection, "HELLO", 110)
    drop_table_purge(connection, "HELLO")

    # 3. New Table - 123
    #    Dimension - 100
    # Expectation:ORA-00903: invalid table name
    try:
        _create_table(connection, "123", 100)
        drop_table_purge(connection, "123")
    except Exception:
        pass

    # 4. New Table - Hello123
    #    Dimension - 65535
    # Expectation:table is created
    _create_table(connection, "Hello123", 65535)
    drop_table_purge(connection, "Hello123")

    # 5. New Table - T1
    #    Dimension - 65536
    # Expectation:ORA-51801: VECTOR column type specification
    # has an unsupported dimension count ('65536').
    try:
        _create_table(connection, "T1", 65536)
        drop_table_purge(connection, "T1")
    except Exception:
        pass

    # 6. New Table - T1
    #    Dimension - 0
    # Expectation:ORA-51801: VECTOR column type specification has
    # an unsupported dimension count (0).
    try:
        _create_table(connection, "T1", 0)
        drop_table_purge(connection, "T1")
    except Exception:
        pass

    # 7. New Table - T1
    #    Dimension - -1
    # Expectation:ORA-51801: VECTOR column type specification has
    # an unsupported dimension count ('-').
    try:
        _create_table(connection, "T1", -1)
        drop_table_purge(connection, "T1")
    except Exception:
        pass

    # 8. New Table - T2
    #     Dimension - '1000'
    # Expectation:table is created
    _create_table(connection, "T2", int("1000"))
    drop_table_purge(connection, "T2")

    # 9. New Table - T3
    #     Dimension - 100 passed as a variable
    # Expectation:table is created
    val = 100
    _create_table(connection, "T3", val)
    drop_table_purge(connection, "T3")

    # 10.
    # Expectation:ORA-00922: missing or invalid option
    val2 = """H
    ello"""
    try:
        _create_table(connection, val2, 545)
        drop_table_purge(connection, val2)
    except Exception:
        pass

    # 11. New Table - हिन्दी
    #     Dimension - 545
    # Expectation:table is created
    _create_table(connection, '"हिन्दी"', 545)
    drop_table_purge(connection, '"हिन्दी"')

    # 12. <schema_name.table_name>
    # Expectation:failure - user does not exist
    try:
        _create_table(connection, "U1.TB4", 128)
        drop_table_purge(connection, "U1.TB4")
    except Exception:
        pass

    # 13.
    # Expectation:table is created
    _create_table(connection, '"T5"', 128)
    drop_table_purge(connection, '"T5"')

    # 14. Toggle Case
    # Expectation:table creation fails
    try:
        _create_table(connection, "TaBlE", 128)
        drop_table_purge(connection, "TaBlE")
    except Exception:
        pass

    # 15. table_name as empty_string
    # Expectation: ORA-00903: invalid table name
    try:
        _create_table(connection, "", 128)
        drop_table_purge(connection, "")
        _create_table(connection, '""', 128)
        drop_table_purge(connection, '""')
    except Exception:
        pass

    # 16. Arithmetic Operations in dimension parameter
    # Expectation:table is created
    n = 1
    _create_table(connection, "T10", n + 500)
    drop_table_purge(connection, "T10")

    # 17. String Operations in table_name&dimension parameter
    # Expectation:table is created
    _create_table(connection, "YaSh".replace("aS", "ok"), 500)
    drop_table_purge(connection, "YaSh".replace("aS", "ok"))


##################################
####### create_hnsw_index #######
##################################


def test_create_hnsw_index_test() -> None:
    try:
        import oracledb
    except ImportError:
        return

    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception:
        sys.exit(1)
    # 1. Table_name - TB1
    #    New Index
    #    distance_strategy - DistanceStrategy.Dot_product
    # Expectation:Index created
    model1 = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )
    vs = OracleVS(connection, model1, "TB1", DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(connection, vs)

    # 2. Creating same index again
    #    Table_name - TB1
    # Expectation:Nothing happens
    try:
        create_index(connection, vs)
        drop_index_if_exists(connection, "HNSW")
    except Exception:
        pass
    drop_table_purge(connection, "TB1")

    # 3. Create index with following parameters:
    #    idx_name - hnsw_idx2
    #    idx_type - HNSW
    # Expectation:Index created
    vs = OracleVS(connection, model1, "TB2", DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(connection, vs, params={"idx_name": "hnsw_idx2", "idx_type": "HNSW"})
    drop_index_if_exists(connection, "hnsw_idx2")
    drop_table_purge(connection, "TB2")

    # 4. Table Name - TB1
    #    idx_name - "हिन्दी"
    #    idx_type - HNSW
    # Expectation:Index created
    try:
        vs = OracleVS(connection, model1, "TB3", DistanceStrategy.EUCLIDEAN_DISTANCE)
        create_index(connection, vs, params={"idx_name": '"हिन्दी"', "idx_type": "HNSW"})
        drop_index_if_exists(connection, '"हिन्दी"')
    except Exception:
        pass
    drop_table_purge(connection, "TB3")

    # 5. idx_name passed empty
    # Expectation:ORA-01741: illegal zero-length identifier
    try:
        vs = OracleVS(connection, model1, "TB4", DistanceStrategy.EUCLIDEAN_DISTANCE)
        create_index(connection, vs, params={"idx_name": '""', "idx_type": "HNSW"})
        drop_index_if_exists(connection, '""')
    except Exception:
        pass
    drop_table_purge(connection, "TB4")

    # 6. idx_type left empty
    # Expectation:Index created
    try:
        vs = OracleVS(connection, model1, "TB5", DistanceStrategy.EUCLIDEAN_DISTANCE)
        create_index(connection, vs, params={"idx_name": "Hello", "idx_type": ""})
        drop_index_if_exists(connection, "Hello")
    except Exception:
        pass
    drop_table_purge(connection, "TB5")

    # 7. efconstruction passed as parameter but not neighbours
    # Expectation:Index created
    vs = OracleVS(connection, model1, "TB7", DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(
        connection,
        vs,
        params={"idx_name": "idx11", "efConstruction": 100, "idx_type": "HNSW"},
    )
    drop_index_if_exists(connection, "idx11")
    drop_table_purge(connection, "TB7")

    # 8. efconstruction passed as parameter as well as neighbours
    # (for this idx_type parameter is also necessary)
    # Expectation:Index created
    vs = OracleVS(connection, model1, "TB8", DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(
        connection,
        vs,
        params={
            "idx_name": "idx11",
            "efConstruction": 100,
            "neighbors": 80,
            "idx_type": "HNSW",
        },
    )
    drop_index_if_exists(connection, "idx11")
    drop_table_purge(connection, "TB8")

    #  9. Limit of Values for(integer values):
    #     parallel
    #     efConstruction
    #     Neighbors
    #     Accuracy
    #     0<Accuracy<=100
    #     0<Neighbour<=2048
    #     0<efConstruction<=65535
    #     0<parallel<=255
    # Expectation:Index created
    drop_table_purge(connection, "TB9")
    vs = OracleVS(connection, model1, "TB9", DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(
        connection,
        vs,
        params={
            "idx_name": "idx11",
            "efConstruction": 65535,
            "neighbors": 2048,
            "idx_type": "HNSW",
            "parallel": 255,
        },
    )
    drop_index_if_exists(connection, "idx11")
    drop_table_purge(connection, "TB9")
    # index not created:
    try:
        vs = OracleVS(connection, model1, "TB10", DistanceStrategy.EUCLIDEAN_DISTANCE)
        create_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 0,
                "neighbors": 2048,
                "idx_type": "HNSW",
                "parallel": 255,
            },
        )
        drop_index_if_exists(connection, "idx11")
    except Exception:
        pass
    # index not created:
    try:
        vs = OracleVS(connection, model1, "TB11", DistanceStrategy.EUCLIDEAN_DISTANCE)
        create_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 100,
                "neighbors": 0,
                "idx_type": "HNSW",
                "parallel": 255,
            },
        )
        drop_index_if_exists(connection, "idx11")
    except Exception:
        pass
    # index not created
    try:
        vs = OracleVS(connection, model1, "TB12", DistanceStrategy.EUCLIDEAN_DISTANCE)
        create_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 100,
                "neighbors": 100,
                "idx_type": "HNSW",
                "parallel": 0,
            },
        )
        drop_index_if_exists(connection, "idx11")
    except Exception:
        pass
    # index not created
    try:
        vs = OracleVS(connection, model1, "TB13", DistanceStrategy.EUCLIDEAN_DISTANCE)
        create_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 10,
                "neighbors": 100,
                "idx_type": "HNSW",
                "parallel": 10,
                "accuracy": 120,
            },
        )
        drop_index_if_exists(connection, "idx11")
    except Exception:
        pass
    # with negative values/out-of-bound values for all 4 of them, we get the same errors
    # Expectation:Index not created
    try:
        vs = OracleVS(connection, model1, "TB14", DistanceStrategy.EUCLIDEAN_DISTANCE)
        create_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 200,
                "neighbors": 100,
                "idx_type": "HNSW",
                "parallel": "hello",
                "accuracy": 10,
            },
        )
        drop_index_if_exists(connection, "idx11")
    except Exception:
        pass
    drop_table_purge(connection, "TB10")
    drop_table_purge(connection, "TB11")
    drop_table_purge(connection, "TB12")
    drop_table_purge(connection, "TB13")
    drop_table_purge(connection, "TB14")

    # 10. Table_name as <schema_name.table_name>
    # Expectation:Index created
    vs = OracleVS(connection, model1, "TB15", DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(
        connection,
        vs,
        params={
            "idx_name": "idx11",
            "efConstruction": 200,
            "neighbors": 100,
            "idx_type": "HNSW",
            "parallel": 8,
            "accuracy": 10,
        },
    )
    drop_index_if_exists(connection, "idx11")
    drop_table_purge(connection, "TB15")

    # 11. index_name as <schema_name.index_name>
    # Expectation:U1 not present
    try:
        vs = OracleVS(
            connection, model1, "U1.TB16", DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        create_index(
            connection,
            vs,
            params={
                "idx_name": "U1.idx11",
                "efConstruction": 200,
                "neighbors": 100,
                "idx_type": "HNSW",
                "parallel": 8,
                "accuracy": 10,
            },
        )
        drop_index_if_exists(connection, "U1.idx11")
        drop_table_purge(connection, "TB16")
    except Exception:
        pass

    # 12. Index_name size >129
    # Expectation:Index not created
    try:
        vs = OracleVS(connection, model1, "TB17", DistanceStrategy.EUCLIDEAN_DISTANCE)
        create_index(connection, vs, params={"idx_name": "x" * 129, "idx_type": "HNSW"})
        drop_index_if_exists(connection, "x" * 129)
    except Exception:
        pass
    drop_table_purge(connection, "TB17")

    # 13. Index_name size 128
    # Expectation:Index created
    vs = OracleVS(connection, model1, "TB18", DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(connection, vs, params={"idx_name": "x" * 128, "idx_type": "HNSW"})
    drop_index_if_exists(connection, "x" * 128)
    drop_table_purge(connection, "TB18")


##################################
####### index_exists #############
##################################


def test_index_exists_test() -> None:
    try:
        import oracledb
    except ImportError:
        return

    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception:
        sys.exit(1)
    model1 = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )
    # 1. Existing Index:(all capital letters)
    # Expectation:true
    vs = OracleVS(connection, model1, "TB1", DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(connection, vs, params={"idx_name": "idx11", "idx_type": "HNSW"})
    _index_exists(connection, "IDX11")

    # 2. Existing Table:(all small letters)
    # Expectation:true
    _index_exists(connection, "idx11")

    # 3. Non-Existing Index
    # Expectation:False
    _index_exists(connection, "Hello")

    # 4. Invalid Index Name
    # Expectation:Error
    try:
        _index_exists(connection, "123")
    except Exception:
        pass

    # 5. Empty String
    # Expectation:Error
    try:
        _index_exists(connection, "")
    except Exception:
        pass
    try:
        _index_exists(connection, "")
    except Exception:
        pass

    # 6. Special Character
    # Expectation:Error
    try:
        _index_exists(connection, "##4")
    except Exception:
        pass

    # 7. Index name length > 128
    # Expectation:Error
    try:
        _index_exists(connection, "x" * 129)
    except Exception:
        pass

    # 8. <Schema_Name.Index_Name>
    # Expectation:true
    _index_exists(connection, "U1.IDX11")

    # 9. Toggle Case (like iDx11)
    # Expectation:true
    _index_exists(connection, "IdX11")

    # 10. Index_Name→ "हिन्दी"
    # Expectation:true
    drop_index_if_exists(connection, "idx11")
    try:
        create_index(connection, vs, params={"idx_name": '"हिन्दी"', "idx_type": "HNSW"})
        _index_exists(connection, '"हिन्दी"')
    except Exception:
        pass
    drop_table_purge(connection, "TB1")


##################################
####### add_texts ################
##################################


def test_add_texts_test() -> None:
    try:
        import oracledb
    except ImportError:
        return

    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception:
        sys.exit(1)
    # 1. Add 2 records to table
    # Expectation:Successful
    texts = ["Rohan", "Shailendra"]
    metadata = [
        {"id": "100", "link": "Document Example Test 1"},
        {"id": "101", "link": "Document Example Test 2"},
    ]
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs_obj = OracleVS(connection, model, "TB1", DistanceStrategy.EUCLIDEAN_DISTANCE)
    vs_obj.add_texts(texts, metadata)
    drop_table_purge(connection, "TB1")

    # 2. Add record but metadata is not there
    # Expectation:An exception occurred :: Either specify an 'ids' list or
    # 'metadatas' with an 'id' attribute for each element.
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs_obj = OracleVS(connection, model, "TB2", DistanceStrategy.EUCLIDEAN_DISTANCE)
    texts2 = ["Sri Ram", "Krishna"]
    vs_obj.add_texts(texts2)
    drop_table_purge(connection, "TB2")

    # 3. Add record with ids option
    #    ids are passed as string
    #    ids are passed as empty string
    #    ids are passed as multi-line string
    #    ids are passed as "<string>"
    # Expectations:
    # Successful
    # Successful
    # Successful
    # Successful

    vs_obj = OracleVS(connection, model, "TB4", DistanceStrategy.EUCLIDEAN_DISTANCE)
    ids3 = ["114", "124"]
    vs_obj.add_texts(texts2, ids=ids3)
    drop_table_purge(connection, "TB4")

    vs_obj = OracleVS(connection, model, "TB5", DistanceStrategy.EUCLIDEAN_DISTANCE)
    ids4 = ["", "134"]
    vs_obj.add_texts(texts2, ids=ids4)
    drop_table_purge(connection, "TB5")

    vs_obj = OracleVS(connection, model, "TB6", DistanceStrategy.EUCLIDEAN_DISTANCE)
    ids5 = [
        """Good afternoon
    my friends""",
        "India",
    ]
    vs_obj.add_texts(texts2, ids=ids5)
    drop_table_purge(connection, "TB6")

    vs_obj = OracleVS(connection, model, "TB7", DistanceStrategy.EUCLIDEAN_DISTANCE)
    ids6 = ['"Good afternoon"', '"India"']
    vs_obj.add_texts(texts2, ids=ids6)
    drop_table_purge(connection, "TB7")

    # 4. Add records with ids and metadatas
    # Expectation:Successful
    vs_obj = OracleVS(connection, model, "TB8", DistanceStrategy.EUCLIDEAN_DISTANCE)
    texts3 = ["Sri Ram 6", "Krishna 6"]
    ids7 = ["1", "2"]
    metadata = [
        {"id": "102", "link": "Document Example", "stream": "Science"},
        {"id": "104", "link": "Document Example 45"},
    ]
    vs_obj.add_texts(texts3, metadata, ids=ids7)
    drop_table_purge(connection, "TB8")

    # 5. Add 10000 records
    # Expectation:Successful
    vs_obj = OracleVS(connection, model, "TB9", DistanceStrategy.EUCLIDEAN_DISTANCE)
    texts4 = ["Sri Ram{0}".format(i) for i in range(1, 10000)]
    ids8 = ["Hello{0}".format(i) for i in range(1, 10000)]
    vs_obj.add_texts(texts4, ids=ids8)
    drop_table_purge(connection, "TB9")

    # 6. Add 2 different record concurrently
    # Expectation:Successful
    def add(val: str) -> None:
        model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        vs_obj = OracleVS(
            connection, model, "TB10", DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        texts5 = [val]
        ids9 = texts5
        vs_obj.add_texts(texts5, ids=ids9)

    thread_1 = threading.Thread(target=add, args=("Sri Ram"))
    thread_2 = threading.Thread(target=add, args=("Sri Krishna"))
    thread_1.start()
    thread_2.start()
    thread_1.join()
    thread_2.join()
    drop_table_purge(connection, "TB10")

    # 7. Add 2 same record concurrently
    # Expectation:Successful, For one of the insert,get primary key violation error
    def add1(val: str) -> None:
        model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        vs_obj = OracleVS(
            connection, model, "TB11", DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        texts = [val]
        ids10 = texts
        vs_obj.add_texts(texts, ids=ids10)

    try:
        thread_1 = threading.Thread(target=add1, args=("Sri Ram"))
        thread_2 = threading.Thread(target=add1, args=("Sri Ram"))
        thread_1.start()
        thread_2.start()
        thread_1.join()
        thread_2.join()
    except Exception:
        pass
    drop_table_purge(connection, "TB11")

    # 8. create object with table name of type <schema_name.table_name>
    # Expectation:U1 does not exist
    try:
        vs_obj = OracleVS(connection, model, "U1.TB14", DistanceStrategy.DOT_PRODUCT)
        for i in range(1, 10):
            texts7 = ["Yash{0}".format(i)]
            ids13 = ["1234{0}".format(i)]
            vs_obj.add_texts(texts7, ids=ids13)
        drop_table_purge(connection, "TB14")
    except Exception:
        pass


##################################
####### embed_documents(text) ####
##################################
def test_embed_documents_test() -> None:
    try:
        import oracledb
    except ImportError:
        return

    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception:
        sys.exit(1)
    # 1. String Example-'Sri Ram'
    # Expectation:Vector Printed
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs_obj = OracleVS(connection, model, "TB7", DistanceStrategy.EUCLIDEAN_DISTANCE)

    # 4. List
    # Expectation:Vector Printed
    vs_obj._embed_documents(["hello", "yash"])
    drop_table_purge(connection, "TB7")


##################################
####### embed_query(text) ########
##################################
def test_embed_query_test() -> None:
    try:
        import oracledb
    except ImportError:
        return

    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception:
        sys.exit(1)
    # 1. String
    # Expectation:Vector printed
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs_obj = OracleVS(connection, model, "TB8", DistanceStrategy.EUCLIDEAN_DISTANCE)
    vs_obj._embed_query("Sri Ram")
    drop_table_purge(connection, "TB8")

    # 3. Empty string
    # Expectation:[]
    vs_obj._embed_query("")


##################################
####### create_index #############
##################################
def test_create_index_test() -> None:
    try:
        import oracledb
    except ImportError:
        return

    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception:
        sys.exit(1)
    # 1. No optional parameters passed
    # Expectation:Successful
    model1 = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )
    vs = OracleVS(connection, model1, "TB1", DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(connection, vs)
    drop_index_if_exists(connection, "HNSW")
    drop_table_purge(connection, "TB1")

    # 2. ivf index
    # Expectation:Successful
    vs = OracleVS(connection, model1, "TB2", DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(connection, vs, {"idx_type": "IVF", "idx_name": "IVF"})
    drop_index_if_exists(connection, "IVF")
    drop_table_purge(connection, "TB2")

    # 3. ivf index with neighbour_part passed as parameter
    # Expectation:Successful
    vs = OracleVS(connection, model1, "TB3", DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(connection, vs, {"idx_type": "IVF", "neighbor_part": 10})
    drop_index_if_exists(connection, "IVF")
    drop_table_purge(connection, "TB3")

    # 4. ivf index with neighbour_part and accuracy passed as parameter
    # Expectation:Successful
    vs = OracleVS(connection, model1, "TB4", DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(
        connection, vs, {"idx_type": "IVF", "neighbor_part": 10, "accuracy": 90}
    )
    drop_index_if_exists(connection, "IVF")
    drop_table_purge(connection, "TB4")

    # 5. ivf index with neighbour_part and parallel passed as parameter
    # Expectation:Successful
    vs = OracleVS(connection, model1, "TB5", DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(
        connection, vs, {"idx_type": "IVF", "neighbor_part": 10, "parallel": 90}
    )
    drop_index_if_exists(connection, "IVF")
    drop_table_purge(connection, "TB5")

    # 6. ivf index and then perform dml(insert)
    # Expectation:Successful
    vs = OracleVS(connection, model1, "TB6", DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(connection, vs, {"idx_type": "IVF", "idx_name": "IVF"})
    texts = ["Sri Ram", "Krishna"]
    vs.add_texts(texts)
    # perform delete
    vs.delete(["hello"])
    drop_index_if_exists(connection, "IVF")
    drop_table_purge(connection, "TB6")

    # 7. ivf index with neighbour_part,parallel and accuracy passed as parameter
    # Expectation:Successful
    vs = OracleVS(connection, model1, "TB7", DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(
        connection,
        vs,
        {"idx_type": "IVF", "neighbor_part": 10, "parallel": 90, "accuracy": 99},
    )
    drop_index_if_exists(connection, "IVF")
    drop_table_purge(connection, "TB7")


##################################
####### perform_search ###########
##################################
def test_perform_search_test() -> None:
    try:
        import oracledb
    except ImportError:
        return

    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception:
        sys.exit(1)
    model1 = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )
    vs_1 = OracleVS(connection, model1, "TB10", DistanceStrategy.EUCLIDEAN_DISTANCE)
    vs_2 = OracleVS(connection, model1, "TB11", DistanceStrategy.DOT_PRODUCT)
    vs_3 = OracleVS(connection, model1, "TB12", DistanceStrategy.COSINE)
    vs_4 = OracleVS(connection, model1, "TB13", DistanceStrategy.EUCLIDEAN_DISTANCE)
    vs_5 = OracleVS(connection, model1, "TB14", DistanceStrategy.DOT_PRODUCT)
    vs_6 = OracleVS(connection, model1, "TB15", DistanceStrategy.COSINE)

    # vector store lists:
    vs_list = [vs_1, vs_2, vs_3, vs_4, vs_5, vs_6]

    for i, vs in enumerate(vs_list, start=1):
        # insert data
        texts = ["Yash", "Varanasi", "Yashaswi", "Mumbai", "BengaluruYash"]
        metadatas = [
            {"id": "hello"},
            {"id": "105"},
            {"id": "106"},
            {"id": "yash"},
            {"id": "108"},
        ]

        vs.add_texts(texts, metadatas)

        # create index
        if i == 1 or i == 2 or i == 3:
            create_index(connection, vs, {"idx_type": "HNSW", "idx_name": f"IDX1{i}"})
        else:
            create_index(connection, vs, {"idx_type": "IVF", "idx_name": f"IDX1{i}"})

        # perform search
        query = "YashB"

        filter = {"id": ["106", "108", "yash"]}

        # similarity_searh without filter
        vs.similarity_search(query, 2)

        # similarity_searh with filter
        vs.similarity_search(query, 2, filter=filter)

        # Similarity search with relevance score
        vs.similarity_search_with_score(query, 2)

        # Similarity search with relevance score with filter
        vs.similarity_search_with_score(query, 2, filter=filter)

        # Max marginal relevance search
        vs.max_marginal_relevance_search(query, 2, fetch_k=20, lambda_mult=0.5)

        # Max marginal relevance search with filter
        vs.max_marginal_relevance_search(
            query, 2, fetch_k=20, lambda_mult=0.5, filter=filter
        )

    drop_table_purge(connection, "TB10")
    drop_table_purge(connection, "TB11")
    drop_table_purge(connection, "TB12")
    drop_table_purge(connection, "TB13")
    drop_table_purge(connection, "TB14")
    drop_table_purge(connection, "TB15")
