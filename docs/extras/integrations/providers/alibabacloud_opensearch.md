# Alibaba Cloud Opensearch

[Alibaba Cloud Opensearch](https://www.alibabacloud.com/product/opensearch) OpenSearch is a one-stop platform to develop intelligent search services. OpenSearch was built based on the large-scale distributed search engine developed by Alibaba. OpenSearch serves more than 500 business cases in Alibaba Group and thousands of Alibaba Cloud customers. OpenSearch helps develop search services in different search scenarios, including e-commerce, O2O, multimedia, the content industry, communities and forums, and big data query in enterprises.

OpenSearch helps you develop high quality, maintenance-free, and high performance intelligent search services to provide your users with high search efficiency and accuracy.

 OpenSearch provides the vector search feature. In specific scenarios, especially test question search and image search scenarios, you can use the vector search feature together with the multimodal search feature to improve the accuracy of search results. This topic describes the syntax and usage notes of vector indexes.

## Purchase an instance and configure it

- Purchase OpenSearch Vector Search Edition from [Alibaba Cloud](https://opensearch.console.aliyun.com) and configure the instance according to the help [documentation](https://help.aliyun.com/document_detail/463198.html?spm=a2c4g.465092.0.0.2cd15002hdwavO).
  
## Alibaba Cloud Opensearch Vector Store Wrappers
supported functions:
- `add_texts`
- `add_documents`
- `from_texts`
- `from_documents`
- `similarity_search`
- `asimilarity_search`
- `similarity_search_by_vector`
- `asimilarity_search_by_vector`
- `similarity_search_with_relevance_scores`

For a more detailed walk through of the Alibaba Cloud OpenSearch wrapper, see [this notebook](../modules/indexes/vectorstores/examples/alibabacloud_opensearch.ipynb)

If you encounter any problems during use, please feel free to contact [xingshaomin.xsm@alibaba-inc.com](xingshaomin.xsm@alibaba-inc.com) , and we will do our best to provide you with assistance and support.

