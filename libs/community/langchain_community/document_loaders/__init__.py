"""**Document Loaders**  are classes to load Documents.

**Document Loaders** are usually used to load a lot of Documents in a single run.

**Class hierarchy:**

.. code-block::

    BaseLoader --> <name>Loader  # Examples: TextLoader, UnstructuredFileLoader

**Main helpers:**

.. code-block::

    Document, <name>TextSplitter
"""
import importlib
from typing import Any

_module_lookup = {
    "AZLyricsLoader": "langchain_community.document_loaders.azlyrics",
    "AcreomLoader": "langchain_community.document_loaders.acreom",
    "AirbyteCDKLoader": "langchain_community.document_loaders.airbyte",
    "AirbyteGongLoader": "langchain_community.document_loaders.airbyte",
    "AirbyteHubspotLoader": "langchain_community.document_loaders.airbyte",
    "AirbyteJSONLoader": "langchain_community.document_loaders.airbyte_json",
    "AirbyteSalesforceLoader": "langchain_community.document_loaders.airbyte",
    "AirbyteShopifyLoader": "langchain_community.document_loaders.airbyte",
    "AirbyteStripeLoader": "langchain_community.document_loaders.airbyte",
    "AirbyteTypeformLoader": "langchain_community.document_loaders.airbyte",
    "AirbyteZendeskSupportLoader": "langchain_community.document_loaders.airbyte",
    "AirtableLoader": "langchain_community.document_loaders.airtable",
    "AmazonTextractPDFLoader": "langchain_community.document_loaders.pdf",
    "ApifyDatasetLoader": "langchain_community.document_loaders.apify_dataset",
    "ArcGISLoader": "langchain_community.document_loaders.arcgis_loader",
    "ArxivLoader": "langchain_community.document_loaders.arxiv",
    "AssemblyAIAudioLoaderById": "langchain_community.document_loaders.assemblyai",
    "AssemblyAIAudioTranscriptLoader": "langchain_community.document_loaders.assemblyai",  # noqa: E501
    "AstraDBLoader": "langchain_community.document_loaders.astradb",
    "AsyncChromiumLoader": "langchain_community.document_loaders.chromium",
    "AsyncHtmlLoader": "langchain_community.document_loaders.async_html",
    "AthenaLoader": "langchain_community.document_loaders.athena",
    "AzureAIDataLoader": "langchain_community.document_loaders.azure_ai_data",
    "AzureAIDocumentIntelligenceLoader": "langchain_community.document_loaders.doc_intelligence",  # noqa: E501
    "AzureBlobStorageContainerLoader": "langchain_community.document_loaders.azure_blob_storage_container",  # noqa: E501
    "AzureBlobStorageFileLoader": "langchain_community.document_loaders.azure_blob_storage_file",  # noqa: E501
    "BSHTMLLoader": "langchain_community.document_loaders.html_bs",
    "BibtexLoader": "langchain_community.document_loaders.bibtex",
    "BigQueryLoader": "langchain_community.document_loaders.bigquery",
    "BiliBiliLoader": "langchain_community.document_loaders.bilibili",
    "BlackboardLoader": "langchain_community.document_loaders.blackboard",
    "Blob": "langchain_community.document_loaders.blob_loaders",
    "BlobLoader": "langchain_community.document_loaders.blob_loaders",
    "BlockchainDocumentLoader": "langchain_community.document_loaders.blockchain",
    "BraveSearchLoader": "langchain_community.document_loaders.brave_search",
    "BrowserlessLoader": "langchain_community.document_loaders.browserless",
    "CSVLoader": "langchain_community.document_loaders.csv_loader",
    "CassandraLoader": "langchain_community.document_loaders.cassandra",
    "ChatGPTLoader": "langchain_community.document_loaders.chatgpt",
    "CoNLLULoader": "langchain_community.document_loaders.conllu",
    "CollegeConfidentialLoader": "langchain_community.document_loaders.college_confidential",  # noqa: E501
    "ConcurrentLoader": "langchain_community.document_loaders.concurrent",
    "ConfluenceLoader": "langchain_community.document_loaders.confluence",
    "CouchbaseLoader": "langchain_community.document_loaders.couchbase",
    "CubeSemanticLoader": "langchain_community.document_loaders.cube_semantic",
    "DataFrameLoader": "langchain_community.document_loaders.dataframe",
    "DatadogLogsLoader": "langchain_community.document_loaders.datadog_logs",
    "DiffbotLoader": "langchain_community.document_loaders.diffbot",
    "DirectoryLoader": "langchain_community.document_loaders.directory",
    "DiscordChatLoader": "langchain_community.document_loaders.discord",
    "DocugamiLoader": "langchain_community.document_loaders.docugami",
    "DocusaurusLoader": "langchain_community.document_loaders.docusaurus",
    "Docx2txtLoader": "langchain_community.document_loaders.word_document",
    "DropboxLoader": "langchain_community.document_loaders.dropbox",
    "DuckDBLoader": "langchain_community.document_loaders.duckdb_loader",
    "EtherscanLoader": "langchain_community.document_loaders.etherscan",
    "EverNoteLoader": "langchain_community.document_loaders.evernote",
    "FacebookChatLoader": "langchain_community.document_loaders.facebook_chat",
    "FaunaLoader": "langchain_community.document_loaders.fauna",
    "FigmaFileLoader": "langchain_community.document_loaders.figma",
    "FileSystemBlobLoader": "langchain_community.document_loaders.blob_loaders",
    "GCSDirectoryLoader": "langchain_community.document_loaders.gcs_directory",
    "GCSFileLoader": "langchain_community.document_loaders.gcs_file",
    "GeoDataFrameLoader": "langchain_community.document_loaders.geodataframe",
    "GitHubIssuesLoader": "langchain_community.document_loaders.github",
    "GitLoader": "langchain_community.document_loaders.git",
    "GitbookLoader": "langchain_community.document_loaders.gitbook",
    "GithubFileLoader": "langchain_community.document_loaders.github",
    "GoogleApiClient": "langchain_community.document_loaders.youtube",
    "GoogleApiYoutubeLoader": "langchain_community.document_loaders.youtube",
    "GoogleDriveLoader": "langchain_community.document_loaders.googledrive",
    "GoogleSpeechToTextLoader": "langchain_community.document_loaders.google_speech_to_text",  # noqa: E501
    "GutenbergLoader": "langchain_community.document_loaders.gutenberg",
    "HNLoader": "langchain_community.document_loaders.hn",
    "HuggingFaceDatasetLoader": "langchain_community.document_loaders.hugging_face_dataset",  # noqa: E501
    "HuggingFaceModelLoader": "langchain_community.document_loaders.hugging_face_model",
    "IFixitLoader": "langchain_community.document_loaders.ifixit",
    "IMSDbLoader": "langchain_community.document_loaders.imsdb",
    "ImageCaptionLoader": "langchain_community.document_loaders.image_captions",
    "IuguLoader": "langchain_community.document_loaders.iugu",
    "JSONLoader": "langchain_community.document_loaders.json_loader",
    "JoplinLoader": "langchain_community.document_loaders.joplin",
    "LakeFSLoader": "langchain_community.document_loaders.lakefs",
    "LarkSuiteDocLoader": "langchain_community.document_loaders.larksuite",
    "LLMSherpaFileLoader": "langchain_community.document_loaders.llmsherpa",
    "MHTMLLoader": "langchain_community.document_loaders.mhtml",
    "MWDumpLoader": "langchain_community.document_loaders.mediawikidump",
    "MastodonTootsLoader": "langchain_community.document_loaders.mastodon",
    "MathpixPDFLoader": "langchain_community.document_loaders.pdf",
    "MaxComputeLoader": "langchain_community.document_loaders.max_compute",
    "MergedDataLoader": "langchain_community.document_loaders.merge",
    "ModernTreasuryLoader": "langchain_community.document_loaders.modern_treasury",
    "MongodbLoader": "langchain_community.document_loaders.mongodb",
    "NewsURLLoader": "langchain_community.document_loaders.news",
    "NotebookLoader": "langchain_community.document_loaders.notebook",
    "NotionDBLoader": "langchain_community.document_loaders.notiondb",
    "NotionDirectoryLoader": "langchain_community.document_loaders.notion",
    "OBSDirectoryLoader": "langchain_community.document_loaders.obs_directory",
    "OBSFileLoader": "langchain_community.document_loaders.obs_file",
    "ObsidianLoader": "langchain_community.document_loaders.obsidian",
    "OneDriveFileLoader": "langchain_community.document_loaders.onedrive_file",
    "OneDriveLoader": "langchain_community.document_loaders.onedrive",
    "OnlinePDFLoader": "langchain_community.document_loaders.pdf",
    "OpenCityDataLoader": "langchain_community.document_loaders.open_city_data",
    "OracleAutonomousDatabaseLoader": "langchain_community.document_loaders.oracleadb_loader",  # noqa: E501
    "OutlookMessageLoader": "langchain_community.document_loaders.email",
    "PDFMinerLoader": "langchain_community.document_loaders.pdf",
    "PDFMinerPDFasHTMLLoader": "langchain_community.document_loaders.pdf",
    "PDFPlumberLoader": "langchain_community.document_loaders.pdf",
    "PagedPDFSplitter": "langchain_community.document_loaders.pdf",
    "PebbloSafeLoader": "langchain_community.document_loaders.pebblo",
    "PlaywrightURLLoader": "langchain_community.document_loaders.url_playwright",
    "PolarsDataFrameLoader": "langchain_community.document_loaders.polars_dataframe",
    "PsychicLoader": "langchain_community.document_loaders.psychic",
    "PubMedLoader": "langchain_community.document_loaders.pubmed",
    "PyMuPDFLoader": "langchain_community.document_loaders.pdf",
    "PyPDFDirectoryLoader": "langchain_community.document_loaders.pdf",
    "PyPDFLoader": "langchain_community.document_loaders.pdf",
    "PyPDFium2Loader": "langchain_community.document_loaders.pdf",
    "PySparkDataFrameLoader": "langchain_community.document_loaders.pyspark_dataframe",
    "PythonLoader": "langchain_community.document_loaders.python",
    "RSSFeedLoader": "langchain_community.document_loaders.rss",
    "ReadTheDocsLoader": "langchain_community.document_loaders.readthedocs",
    "RecursiveUrlLoader": "langchain_community.document_loaders.recursive_url_loader",
    "RedditPostsLoader": "langchain_community.document_loaders.reddit",
    "RoamLoader": "langchain_community.document_loaders.roam",
    "RocksetLoader": "langchain_community.document_loaders.rocksetdb",
    "S3DirectoryLoader": "langchain_community.document_loaders.s3_directory",
    "S3FileLoader": "langchain_community.document_loaders.s3_file",
    "SQLDatabaseLoader": "langchain_community.document_loaders.sql_database",
    "SRTLoader": "langchain_community.document_loaders.srt",
    "SeleniumURLLoader": "langchain_community.document_loaders.url_selenium",
    "SharePointLoader": "langchain_community.document_loaders.sharepoint",
    "SitemapLoader": "langchain_community.document_loaders.sitemap",
    "SlackDirectoryLoader": "langchain_community.document_loaders.slack_directory",
    "SnowflakeLoader": "langchain_community.document_loaders.snowflake_loader",
    "SpreedlyLoader": "langchain_community.document_loaders.spreedly",
    "StripeLoader": "langchain_community.document_loaders.stripe",
    "SurrealDBLoader": "langchain_community.document_loaders.surrealdb",
    "TelegramChatApiLoader": "langchain_community.document_loaders.telegram",
    "TelegramChatFileLoader": "langchain_community.document_loaders.telegram",
    "TelegramChatLoader": "langchain_community.document_loaders.telegram",
    "TencentCOSDirectoryLoader": "langchain_community.document_loaders.tencent_cos_directory",  # noqa: E501
    "TencentCOSFileLoader": "langchain_community.document_loaders.tencent_cos_file",
    "TensorflowDatasetLoader": "langchain_community.document_loaders.tensorflow_datasets",  # noqa: E501
    "TextLoader": "langchain_community.document_loaders.text",
    "TiDBLoader": "langchain_community.document_loaders.tidb",
    "ToMarkdownLoader": "langchain_community.document_loaders.tomarkdown",
    "TomlLoader": "langchain_community.document_loaders.toml",
    "TrelloLoader": "langchain_community.document_loaders.trello",
    "TwitterTweetLoader": "langchain_community.document_loaders.twitter",
    "UnstructuredAPIFileIOLoader": "langchain_community.document_loaders.unstructured",
    "UnstructuredAPIFileLoader": "langchain_community.document_loaders.unstructured",
    "UnstructuredCHMLoader": "langchain_community.document_loaders.chm",
    "UnstructuredCSVLoader": "langchain_community.document_loaders.csv_loader",
    "UnstructuredEPubLoader": "langchain_community.document_loaders.epub",
    "UnstructuredEmailLoader": "langchain_community.document_loaders.email",
    "UnstructuredExcelLoader": "langchain_community.document_loaders.excel",
    "UnstructuredFileIOLoader": "langchain_community.document_loaders.unstructured",
    "UnstructuredFileLoader": "langchain_community.document_loaders.unstructured",
    "UnstructuredHTMLLoader": "langchain_community.document_loaders.html",
    "UnstructuredImageLoader": "langchain_community.document_loaders.image",
    "UnstructuredMarkdownLoader": "langchain_community.document_loaders.markdown",
    "UnstructuredODTLoader": "langchain_community.document_loaders.odt",
    "UnstructuredOrgModeLoader": "langchain_community.document_loaders.org_mode",
    "UnstructuredPDFLoader": "langchain_community.document_loaders.pdf",
    "UnstructuredPowerPointLoader": "langchain_community.document_loaders.powerpoint",
    "UnstructuredRSTLoader": "langchain_community.document_loaders.rst",
    "UnstructuredRTFLoader": "langchain_community.document_loaders.rtf",
    "UnstructuredTSVLoader": "langchain_community.document_loaders.tsv",
    "UnstructuredURLLoader": "langchain_community.document_loaders.url",
    "UnstructuredWordDocumentLoader": "langchain_community.document_loaders.word_document",  # noqa: E501
    "UnstructuredXMLLoader": "langchain_community.document_loaders.xml",
    "VsdxLoader": "langchain_community.document_loaders.vsdx",
    "WeatherDataLoader": "langchain_community.document_loaders.weather",
    "WebBaseLoader": "langchain_community.document_loaders.web_base",
    "WhatsAppChatLoader": "langchain_community.document_loaders.whatsapp_chat",
    "WikipediaLoader": "langchain_community.document_loaders.wikipedia",
    "XorbitsLoader": "langchain_community.document_loaders.xorbits",
    "YoutubeAudioLoader": "langchain_community.document_loaders.blob_loaders",
    "YoutubeLoader": "langchain_community.document_loaders.youtube",
    "YuqueLoader": "langchain_community.document_loaders.yuque",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_module_lookup.keys())
