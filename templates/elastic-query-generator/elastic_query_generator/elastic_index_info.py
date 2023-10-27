from typing import List


def _list_indices(database, include_indices=None, ignore_indices=None) -> List[str]:
    all_indices = [
        index["index"] for index in database.cat.indices(format="json")
    ]

    if include_indices:
        all_indices = [i for i in all_indices if i in include_indices]
    if ignore_indices:
        all_indices = [i for i in all_indices if i not in ignore_indices]

    return all_indices

def get_indices_infos(database, sample_documents_in_index_info=5) -> str:
    indices = _list_indices(database)
    mappings = database.indices.get_mapping(index=",".join(indices))
    if sample_documents_in_index_info > 0:
        for k, v in mappings.items():
            hits = database.search(
                index=k,
                query={"match_all": {}},
                size=sample_documents_in_index_info,
            )["hits"]["hits"]
            hits = [str(hit["_source"]) for hit in hits]
            mappings[k]["mappings"] = str(v) + "\n\n/*\n" + "\n".join(hits) + "\n*/"
    return "\n\n".join(
        [
            "Mapping for index {}:\n{}".format(index, mappings[index]["mappings"])
            for index in mappings
        ]
    )