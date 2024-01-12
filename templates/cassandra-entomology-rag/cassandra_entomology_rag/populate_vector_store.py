import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def populate(vector_store):
    # is the store empty? find out with a probe search
    hits = vector_store.similarity_search_by_vector(
        embedding=[0.001] * 1536,
        k=1,
    )
    #
    if len(hits) == 0:
        # this seems a first run:
        # must populate the vector store
        src_file_name = os.path.join(BASE_DIR, "..", "sources.txt")
        lines = [
            line.strip()
            for line in open(src_file_name).readlines()
            if line.strip()
            if line[0] != "#"
        ]
        # deterministic IDs to prevent duplicates on multiple runs
        ids = ["_".join(line.split(" ")[:2]).lower().replace(":", "") for line in lines]
        #
        vector_store.add_texts(texts=lines, ids=ids)
        return len(lines)
    else:
        return 0
