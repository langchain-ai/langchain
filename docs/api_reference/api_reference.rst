.. _api_reference:

=============
API Reference
=============

:mod:`langchain.agents`: Agents
================================

.. automodule:: langchain.agents
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: agents
    :template: class.rst

    agents.agent.Agent
    agents.agent.AgentExecutor
    agents.agent.AgentOutputParser
    agents.agent.BaseMultiActionAgent
    agents.agent.BaseSingleActionAgent
    agents.agent.ExceptionTool
    agents.agent.LLMSingleActionAgent
    agents.agent_toolkits.azure_cognitive_services.toolkit.AzureCognitiveServicesToolkit
    agents.agent_toolkits.base.BaseToolkit
    agents.agent_toolkits.file_management.toolkit.FileManagementToolkit
    agents.agent_toolkits.gmail.toolkit.GmailToolkit
    agents.agent_toolkits.jira.toolkit.JiraToolkit
    agents.agent_toolkits.json.toolkit.JsonToolkit
    agents.agent_toolkits.nla.tool.NLATool
    agents.agent_toolkits.nla.toolkit.NLAToolkit
    agents.agent_toolkits.office365.toolkit.O365Toolkit
    agents.agent_toolkits.openapi.planner.RequestsDeleteToolWithParsing
    agents.agent_toolkits.openapi.planner.RequestsGetToolWithParsing
    agents.agent_toolkits.openapi.planner.RequestsPatchToolWithParsing
    agents.agent_toolkits.openapi.planner.RequestsPostToolWithParsing
    agents.agent_toolkits.openapi.toolkit.OpenAPIToolkit
    agents.agent_toolkits.openapi.toolkit.RequestsToolkit
    agents.agent_toolkits.playwright.toolkit.PlayWrightBrowserToolkit
    agents.agent_toolkits.powerbi.toolkit.PowerBIToolkit
    agents.agent_toolkits.spark_sql.toolkit.SparkSQLToolkit
    agents.agent_toolkits.sql.toolkit.SQLDatabaseToolkit
    agents.agent_toolkits.vectorstore.toolkit.VectorStoreInfo
    agents.agent_toolkits.vectorstore.toolkit.VectorStoreRouterToolkit
    agents.agent_toolkits.vectorstore.toolkit.VectorStoreToolkit
    agents.agent_toolkits.zapier.toolkit.ZapierToolkit
    agents.agent_types.AgentType
    agents.chat.base.ChatAgent
    agents.chat.output_parser.ChatOutputParser
    agents.conversational.base.ConversationalAgent
    agents.conversational.output_parser.ConvoOutputParser
    agents.conversational_chat.base.ConversationalChatAgent
    agents.conversational_chat.output_parser.ConvoOutputParser
    agents.mrkl.base.ChainConfig
    agents.mrkl.base.MRKLChain
    agents.mrkl.base.ZeroShotAgent
    agents.mrkl.output_parser.MRKLOutputParser
    agents.openai_functions_agent.base.OpenAIFunctionsAgent
    agents.openai_functions_multi_agent.base.OpenAIMultiFunctionsAgent
    agents.react.base.ReActChain
    agents.react.base.ReActDocstoreAgent
    agents.react.base.ReActTextWorldAgent
    agents.react.output_parser.ReActOutputParser
    agents.schema.AgentScratchPadChatPromptTemplate
    agents.self_ask_with_search.base.SelfAskWithSearchAgent
    agents.self_ask_with_search.base.SelfAskWithSearchChain
    agents.self_ask_with_search.output_parser.SelfAskOutputParser
    agents.structured_chat.base.StructuredChatAgent
    agents.structured_chat.output_parser.StructuredChatOutputParser
    agents.structured_chat.output_parser.StructuredChatOutputParserWithRetries
    agents.tools.InvalidTool

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: agents

    agents.agent_toolkits.csv.base.create_csv_agent
    agents.agent_toolkits.json.base.create_json_agent
    agents.agent_toolkits.openapi.base.create_openapi_agent
    agents.agent_toolkits.openapi.planner.create_openapi_agent
    agents.agent_toolkits.openapi.spec.dereference_refs
    agents.agent_toolkits.openapi.spec.reduce_openapi_spec
    agents.agent_toolkits.pandas.base.create_pandas_dataframe_agent
    agents.agent_toolkits.powerbi.base.create_pbi_agent
    agents.agent_toolkits.powerbi.chat_base.create_pbi_chat_agent
    agents.agent_toolkits.python.base.create_python_agent
    agents.agent_toolkits.spark.base.create_spark_dataframe_agent
    agents.agent_toolkits.spark_sql.base.create_spark_sql_agent
    agents.agent_toolkits.sql.base.create_sql_agent
    agents.agent_toolkits.vectorstore.base.create_vectorstore_agent
    agents.agent_toolkits.vectorstore.base.create_vectorstore_router_agent
    agents.initialize.initialize_agent
    agents.load_tools.get_all_tool_names
    agents.load_tools.load_huggingface_tool
    agents.load_tools.load_tools
    agents.loading.load_agent
    agents.loading.load_agent_from_config
    agents.utils.validate_tools_single_input

:mod:`langchain.base_language`: Base Language
==============================================

.. automodule:: langchain.base_language
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: base_language
    :template: class.rst

    base_language.BaseLanguageModel

:mod:`langchain.cache`: Cache
==============================

.. automodule:: langchain.cache
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: cache
    :template: class.rst

    cache.BaseCache
    cache.FullLLMCache
    cache.GPTCache
    cache.InMemoryCache
    cache.MomentoCache
    cache.RedisCache
    cache.RedisSemanticCache
    cache.SQLAlchemyCache
    cache.SQLiteCache

:mod:`langchain.callbacks`: Callbacks
======================================

.. automodule:: langchain.callbacks
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: callbacks
    :template: class.rst

    callbacks.aim_callback.AimCallbackHandler
    callbacks.argilla_callback.ArgillaCallbackHandler
    callbacks.arize_callback.ArizeCallbackHandler
    callbacks.base.AsyncCallbackHandler
    callbacks.base.BaseCallbackHandler
    callbacks.base.BaseCallbackManager
    callbacks.clearml_callback.ClearMLCallbackHandler
    callbacks.comet_ml_callback.CometCallbackHandler
    callbacks.file.FileCallbackHandler
    callbacks.human.HumanApprovalCallbackHandler
    callbacks.human.HumanRejectedException
    callbacks.infino_callback.InfinoCallbackHandler
    callbacks.manager.AsyncCallbackManager
    callbacks.manager.AsyncCallbackManagerForChainRun
    callbacks.manager.AsyncCallbackManagerForLLMRun
    callbacks.manager.AsyncCallbackManagerForToolRun
    callbacks.manager.AsyncRunManager
    callbacks.manager.BaseRunManager
    callbacks.manager.CallbackManager
    callbacks.manager.CallbackManagerForChainRun
    callbacks.manager.CallbackManagerForLLMRun
    callbacks.manager.CallbackManagerForToolRun
    callbacks.manager.RunManager
    callbacks.mlflow_callback.MlflowCallbackHandler
    callbacks.openai_info.OpenAICallbackHandler
    callbacks.stdout.StdOutCallbackHandler
    callbacks.streaming_aiter.AsyncIteratorCallbackHandler
    callbacks.streaming_aiter_final_only.AsyncFinalIteratorCallbackHandler
    callbacks.streaming_stdout.StreamingStdOutCallbackHandler
    callbacks.streaming_stdout_final_only.FinalStreamingStdOutCallbackHandler
    callbacks.streamlit.mutable_expander.ChildRecord
    callbacks.streamlit.mutable_expander.ChildType
    callbacks.streamlit.streamlit_callback_handler.LLMThoughtState
    callbacks.streamlit.streamlit_callback_handler.StreamlitCallbackHandler
    callbacks.streamlit.streamlit_callback_handler.ToolRecord
    callbacks.tracers.base.BaseTracer
    callbacks.tracers.base.TracerException
    callbacks.tracers.evaluation.EvaluatorCallbackHandler
    callbacks.tracers.langchain.LangChainTracer
    callbacks.tracers.langchain_v1.LangChainTracerV1
    callbacks.tracers.run_collector.RunCollectorCallbackHandler
    callbacks.tracers.schemas.BaseRun
    callbacks.tracers.schemas.ChainRun
    callbacks.tracers.schemas.LLMRun
    callbacks.tracers.schemas.Run
    callbacks.tracers.schemas.ToolRun
    callbacks.tracers.schemas.TracerSession
    callbacks.tracers.schemas.TracerSessionBase
    callbacks.tracers.schemas.TracerSessionV1
    callbacks.tracers.schemas.TracerSessionV1Base
    callbacks.tracers.schemas.TracerSessionV1Create
    callbacks.tracers.stdout.ConsoleCallbackHandler
    callbacks.tracers.wandb.WandbRunArgs
    callbacks.tracers.wandb.WandbTracer
    callbacks.wandb_callback.WandbCallbackHandler
    callbacks.whylabs_callback.WhyLabsCallbackHandler

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: callbacks

    callbacks.aim_callback.import_aim
    callbacks.clearml_callback.import_clearml
    callbacks.comet_ml_callback.import_comet_ml
    callbacks.infino_callback.import_infino
    callbacks.manager.env_var_is_set
    callbacks.manager.get_openai_callback
    callbacks.manager.trace_as_chain_group
    callbacks.manager.tracing_enabled
    callbacks.manager.tracing_v2_enabled
    callbacks.manager.wandb_tracing_enabled
    callbacks.mlflow_callback.analyze_text
    callbacks.mlflow_callback.construct_html_from_prompt_and_generation
    callbacks.mlflow_callback.import_mlflow
    callbacks.openai_info.get_openai_token_cost_for_model
    callbacks.openai_info.standardize_model_name
    callbacks.streamlit.__init__.StreamlitCallbackHandler
    callbacks.tracers.langchain.log_error_once
    callbacks.tracers.langchain.wait_for_all_tracers
    callbacks.tracers.langchain_v1.get_headers
    callbacks.tracers.stdout.elapsed
    callbacks.tracers.stdout.try_json_stringify
    callbacks.utils.flatten_dict
    callbacks.utils.hash_string
    callbacks.utils.import_pandas
    callbacks.utils.import_spacy
    callbacks.utils.import_textstat
    callbacks.utils.load_json
    callbacks.wandb_callback.analyze_text
    callbacks.wandb_callback.construct_html_from_prompt_and_generation
    callbacks.wandb_callback.import_wandb
    callbacks.wandb_callback.load_json_to_dict
    callbacks.whylabs_callback.import_langkit

:mod:`langchain.chains`: Chains
================================

.. automodule:: langchain.chains
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: chains
    :template: class.rst

    chains.api.base.APIChain
    chains.api.openapi.chain.OpenAPIEndpointChain
    chains.api.openapi.requests_chain.APIRequesterChain
    chains.api.openapi.requests_chain.APIRequesterOutputParser
    chains.api.openapi.response_chain.APIResponderChain
    chains.api.openapi.response_chain.APIResponderOutputParser
    chains.base.Chain
    chains.combine_documents.base.AnalyzeDocumentChain
    chains.combine_documents.base.BaseCombineDocumentsChain
    chains.combine_documents.map_reduce.CombineDocsProtocol
    chains.combine_documents.map_reduce.MapReduceDocumentsChain
    chains.combine_documents.map_rerank.MapRerankDocumentsChain
    chains.combine_documents.refine.RefineDocumentsChain
    chains.combine_documents.stuff.StuffDocumentsChain
    chains.constitutional_ai.base.ConstitutionalChain
    chains.constitutional_ai.models.ConstitutionalPrinciple
    chains.conversation.base.ConversationChain
    chains.conversational_retrieval.base.BaseConversationalRetrievalChain
    chains.conversational_retrieval.base.ChatVectorDBChain
    chains.conversational_retrieval.base.ConversationalRetrievalChain
    chains.flare.base.FlareChain
    chains.flare.base.QuestionGeneratorChain
    chains.flare.prompts.FinishedOutputParser
    chains.graph_qa.base.GraphQAChain
    chains.graph_qa.cypher.GraphCypherQAChain
    chains.graph_qa.kuzu.KuzuQAChain
    chains.graph_qa.nebulagraph.NebulaGraphQAChain
    chains.hyde.base.HypotheticalDocumentEmbedder
    chains.llm.LLMChain
    chains.llm_bash.base.LLMBashChain
    chains.llm_bash.prompt.BashOutputParser
    chains.llm_checker.base.LLMCheckerChain
    chains.llm_math.base.LLMMathChain
    chains.llm_requests.LLMRequestsChain
    chains.llm_summarization_checker.base.LLMSummarizationCheckerChain
    chains.mapreduce.MapReduceChain
    chains.moderation.OpenAIModerationChain
    chains.natbot.base.NatBotChain
    chains.natbot.crawler.ElementInViewPort
    chains.openai_functions.citation_fuzzy_match.FactWithEvidence
    chains.openai_functions.citation_fuzzy_match.QuestionAnswer
    chains.openai_functions.openapi.SimpleRequestChain
    chains.openai_functions.qa_with_structure.AnswerWithSources
    chains.pal.base.PALChain
    chains.prompt_selector.BasePromptSelector
    chains.prompt_selector.ConditionalPromptSelector
    chains.qa_generation.base.QAGenerationChain
    chains.qa_with_sources.base.BaseQAWithSourcesChain
    chains.qa_with_sources.base.QAWithSourcesChain
    chains.qa_with_sources.loading.LoadingCallable
    chains.qa_with_sources.retrieval.RetrievalQAWithSourcesChain
    chains.qa_with_sources.vector_db.VectorDBQAWithSourcesChain
    chains.query_constructor.base.StructuredQueryOutputParser
    chains.query_constructor.ir.Comparator
    chains.query_constructor.ir.Comparison
    chains.query_constructor.ir.Expr
    chains.query_constructor.ir.FilterDirective
    chains.query_constructor.ir.Operation
    chains.query_constructor.ir.Operator
    chains.query_constructor.ir.StructuredQuery
    chains.query_constructor.ir.Visitor
    chains.query_constructor.parser.QueryTransformer
    chains.query_constructor.schema.AttributeInfo
    chains.question_answering.__init__.LoadingCallable
    chains.retrieval_qa.base.BaseRetrievalQA
    chains.retrieval_qa.base.RetrievalQA
    chains.retrieval_qa.base.VectorDBQA
    chains.router.base.MultiRouteChain
    chains.router.base.Route
    chains.router.base.RouterChain
    chains.router.embedding_router.EmbeddingRouterChain
    chains.router.llm_router.LLMRouterChain
    chains.router.llm_router.RouterOutputParser
    chains.router.multi_prompt.MultiPromptChain
    chains.router.multi_retrieval_qa.MultiRetrievalQAChain
    chains.sequential.SequentialChain
    chains.sequential.SimpleSequentialChain
    chains.sql_database.base.SQLDatabaseChain
    chains.sql_database.base.SQLDatabaseSequentialChain
    chains.summarize.__init__.LoadingCallable
    chains.transform.TransformChain

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: chains

    chains.combine_documents.base.format_document
    chains.graph_qa.cypher.extract_cypher
    chains.loading.load_chain
    chains.loading.load_chain_from_config
    chains.openai_functions.citation_fuzzy_match.create_citation_fuzzy_match_chain
    chains.openai_functions.extraction.create_extraction_chain
    chains.openai_functions.extraction.create_extraction_chain_pydantic
    chains.openai_functions.openapi.get_openapi_chain
    chains.openai_functions.openapi.openapi_spec_to_openai_fn
    chains.openai_functions.qa_with_structure.create_qa_with_sources_chain
    chains.openai_functions.qa_with_structure.create_qa_with_structure_chain
    chains.openai_functions.tagging.create_tagging_chain
    chains.openai_functions.tagging.create_tagging_chain_pydantic
    chains.openai_functions.utils.get_llm_kwargs
    chains.pal.math_prompt.solution
    chains.pal.math_prompt.solution
    chains.pal.math_prompt.solution
    chains.pal.math_prompt.solution
    chains.pal.math_prompt.solution
    chains.pal.math_prompt.solution
    chains.pal.math_prompt.solution
    chains.pal.math_prompt.solution
    chains.prompt_selector.is_chat_model
    chains.prompt_selector.is_llm
    chains.qa_with_sources.loading.load_qa_with_sources_chain
    chains.query_constructor.base.load_query_constructor_chain
    chains.query_constructor.parser.get_parser
    chains.question_answering.__init__.load_qa_chain
    chains.summarize.__init__.load_summarize_chain

:mod:`langchain.chat_models`: Chat Models
==========================================

.. automodule:: langchain.chat_models
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: chat_models
    :template: class.rst

    chat_models.anthropic.ChatAnthropic
    chat_models.azure_openai.AzureChatOpenAI
    chat_models.base.BaseChatModel
    chat_models.base.SimpleChatModel
    chat_models.fake.FakeListChatModel
    chat_models.google_palm.ChatGooglePalm
    chat_models.google_palm.ChatGooglePalmError
    chat_models.openai.ChatOpenAI
    chat_models.promptlayer_openai.PromptLayerChatOpenAI
    chat_models.vertexai.ChatVertexAI

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: chat_models

    chat_models.google_palm.chat_with_retry

:mod:`langchain.client`: Client
================================

.. automodule:: langchain.client
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: client
    :template: class.rst

    client.runner_utils.InputFormatError

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: client

    client.runner_utils.run_llm
    client.runner_utils.run_llm_or_chain
    client.runner_utils.run_on_dataset
    client.runner_utils.run_on_examples

:mod:`langchain.docstore`: Docstore
====================================

.. automodule:: langchain.docstore
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: docstore
    :template: class.rst

    docstore.arbitrary_fn.DocstoreFn
    docstore.base.AddableMixin
    docstore.base.Docstore
    docstore.in_memory.InMemoryDocstore
    docstore.wikipedia.Wikipedia

:mod:`langchain.document_loaders`: Document Loaders
====================================================

.. automodule:: langchain.document_loaders
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: document_loaders
    :template: class.rst

    document_loaders.acreom.AcreomLoader
    document_loaders.airbyte_json.AirbyteJSONLoader
    document_loaders.airtable.AirtableLoader
    document_loaders.apify_dataset.ApifyDatasetLoader
    document_loaders.arxiv.ArxivLoader
    document_loaders.azlyrics.AZLyricsLoader
    document_loaders.azure_blob_storage_container.AzureBlobStorageContainerLoader
    document_loaders.azure_blob_storage_file.AzureBlobStorageFileLoader
    document_loaders.base.BaseBlobParser
    document_loaders.base.BaseLoader
    document_loaders.bibtex.BibtexLoader
    document_loaders.bigquery.BigQueryLoader
    document_loaders.bilibili.BiliBiliLoader
    document_loaders.blackboard.BlackboardLoader
    document_loaders.blob_loaders.file_system.FileSystemBlobLoader
    document_loaders.blob_loaders.schema.Blob
    document_loaders.blob_loaders.schema.BlobLoader
    document_loaders.blob_loaders.youtube_audio.YoutubeAudioLoader
    document_loaders.blockchain.BlockchainDocumentLoader
    document_loaders.blockchain.BlockchainType
    document_loaders.chatgpt.ChatGPTLoader
    document_loaders.college_confidential.CollegeConfidentialLoader
    document_loaders.confluence.ConfluenceLoader
    document_loaders.confluence.ContentFormat
    document_loaders.conllu.CoNLLULoader
    document_loaders.csv_loader.CSVLoader
    document_loaders.csv_loader.UnstructuredCSVLoader
    document_loaders.dataframe.DataFrameLoader
    document_loaders.diffbot.DiffbotLoader
    document_loaders.directory.DirectoryLoader
    document_loaders.discord.DiscordChatLoader
    document_loaders.docugami.DocugamiLoader
    document_loaders.duckdb_loader.DuckDBLoader
    document_loaders.email.OutlookMessageLoader
    document_loaders.email.UnstructuredEmailLoader
    document_loaders.embaas.BaseEmbaasLoader
    document_loaders.embaas.EmbaasBlobLoader
    document_loaders.embaas.EmbaasDocumentExtractionParameters
    document_loaders.embaas.EmbaasDocumentExtractionPayload
    document_loaders.embaas.EmbaasLoader
    document_loaders.epub.UnstructuredEPubLoader
    document_loaders.evernote.EverNoteLoader
    document_loaders.excel.UnstructuredExcelLoader
    document_loaders.facebook_chat.FacebookChatLoader
    document_loaders.fauna.FaunaLoader
    document_loaders.figma.FigmaFileLoader
    document_loaders.gcs_directory.GCSDirectoryLoader
    document_loaders.gcs_file.GCSFileLoader
    document_loaders.generic.GenericLoader
    document_loaders.git.GitLoader
    document_loaders.gitbook.GitbookLoader
    document_loaders.github.BaseGitHubLoader
    document_loaders.github.GitHubIssuesLoader
    document_loaders.googledrive.GoogleDriveLoader
    document_loaders.gutenberg.GutenbergLoader
    document_loaders.helpers.FileEncoding
    document_loaders.hn.HNLoader
    document_loaders.html.UnstructuredHTMLLoader
    document_loaders.html_bs.BSHTMLLoader
    document_loaders.hugging_face_dataset.HuggingFaceDatasetLoader
    document_loaders.ifixit.IFixitLoader
    document_loaders.image.UnstructuredImageLoader
    document_loaders.image_captions.ImageCaptionLoader
    document_loaders.imsdb.IMSDbLoader
    document_loaders.iugu.IuguLoader
    document_loaders.joplin.JoplinLoader
    document_loaders.json_loader.JSONLoader
    document_loaders.larksuite.LarkSuiteDocLoader
    document_loaders.markdown.UnstructuredMarkdownLoader
    document_loaders.mastodon.MastodonTootsLoader
    document_loaders.max_compute.MaxComputeLoader
    document_loaders.mediawikidump.MWDumpLoader
    document_loaders.merge.MergedDataLoader
    document_loaders.mhtml.MHTMLLoader
    document_loaders.modern_treasury.ModernTreasuryLoader
    document_loaders.notebook.NotebookLoader
    document_loaders.notion.NotionDirectoryLoader
    document_loaders.notiondb.NotionDBLoader
    document_loaders.obsidian.ObsidianLoader
    document_loaders.odt.UnstructuredODTLoader
    document_loaders.onedrive.OneDriveLoader
    document_loaders.onedrive_file.OneDriveFileLoader
    document_loaders.open_city_data.OpenCityDataLoader
    document_loaders.org_mode.UnstructuredOrgModeLoader
    document_loaders.parsers.audio.OpenAIWhisperParser
    document_loaders.parsers.generic.MimeTypeBasedParser
    document_loaders.parsers.grobid.GrobidParser
    document_loaders.parsers.grobid.ServerUnavailableException
    document_loaders.parsers.html.bs4.BS4HTMLParser
    document_loaders.parsers.language.code_segmenter.CodeSegmenter
    document_loaders.parsers.language.javascript.JavaScriptSegmenter
    document_loaders.parsers.language.language_parser.LanguageParser
    document_loaders.parsers.language.python.PythonSegmenter
    document_loaders.parsers.pdf.PDFMinerParser
    document_loaders.parsers.pdf.PDFPlumberParser
    document_loaders.parsers.pdf.PyMuPDFParser
    document_loaders.parsers.pdf.PyPDFParser
    document_loaders.parsers.pdf.PyPDFium2Parser
    document_loaders.parsers.txt.TextParser
    document_loaders.pdf.BasePDFLoader
    document_loaders.pdf.MathpixPDFLoader
    document_loaders.pdf.OnlinePDFLoader
    document_loaders.pdf.PDFMinerLoader
    document_loaders.pdf.PDFMinerPDFasHTMLLoader
    document_loaders.pdf.PDFPlumberLoader
    document_loaders.pdf.PyMuPDFLoader
    document_loaders.pdf.PyPDFDirectoryLoader
    document_loaders.pdf.PyPDFLoader
    document_loaders.pdf.PyPDFium2Loader
    document_loaders.pdf.UnstructuredPDFLoader
    document_loaders.powerpoint.UnstructuredPowerPointLoader
    document_loaders.psychic.PsychicLoader
    document_loaders.pyspark_dataframe.PySparkDataFrameLoader
    document_loaders.python.PythonLoader
    document_loaders.readthedocs.ReadTheDocsLoader
    document_loaders.recursive_url_loader.RecursiveUrlLoader
    document_loaders.reddit.RedditPostsLoader
    document_loaders.roam.RoamLoader
    document_loaders.rst.UnstructuredRSTLoader
    document_loaders.rtf.UnstructuredRTFLoader
    document_loaders.s3_directory.S3DirectoryLoader
    document_loaders.s3_file.S3FileLoader
    document_loaders.sitemap.SitemapLoader
    document_loaders.slack_directory.SlackDirectoryLoader
    document_loaders.snowflake_loader.SnowflakeLoader
    document_loaders.spreedly.SpreedlyLoader
    document_loaders.srt.SRTLoader
    document_loaders.stripe.StripeLoader
    document_loaders.telegram.TelegramChatApiLoader
    document_loaders.telegram.TelegramChatFileLoader
    document_loaders.tencent_cos_directory.TencentCOSDirectoryLoader
    document_loaders.tencent_cos_file.TencentCOSFileLoader
    document_loaders.text.TextLoader
    document_loaders.tomarkdown.ToMarkdownLoader
    document_loaders.toml.TomlLoader
    document_loaders.trello.TrelloLoader
    document_loaders.twitter.TwitterTweetLoader
    document_loaders.unstructured.UnstructuredAPIFileIOLoader
    document_loaders.unstructured.UnstructuredAPIFileLoader
    document_loaders.unstructured.UnstructuredBaseLoader
    document_loaders.unstructured.UnstructuredFileIOLoader
    document_loaders.unstructured.UnstructuredFileLoader
    document_loaders.url.UnstructuredURLLoader
    document_loaders.url_playwright.PlaywrightURLLoader
    document_loaders.url_selenium.SeleniumURLLoader
    document_loaders.weather.WeatherDataLoader
    document_loaders.web_base.WebBaseLoader
    document_loaders.whatsapp_chat.WhatsAppChatLoader
    document_loaders.wikipedia.WikipediaLoader
    document_loaders.word_document.Docx2txtLoader
    document_loaders.word_document.UnstructuredWordDocumentLoader
    document_loaders.xml.UnstructuredXMLLoader
    document_loaders.youtube.GoogleApiYoutubeLoader
    document_loaders.youtube.YoutubeLoader

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: document_loaders

    document_loaders.chatgpt.concatenate_rows
    document_loaders.facebook_chat.concatenate_rows
    document_loaders.helpers.detect_file_encodings
    document_loaders.notebook.concatenate_cells
    document_loaders.notebook.remove_newlines
    document_loaders.parsers.registry.get_parser
    document_loaders.telegram.concatenate_rows
    document_loaders.telegram.text_to_docs
    document_loaders.unstructured.get_elements_from_api
    document_loaders.unstructured.satisfies_min_unstructured_version
    document_loaders.unstructured.validate_unstructured_version
    document_loaders.whatsapp_chat.concatenate_rows

:mod:`langchain.document_transformers`: Document Transformers
==============================================================

.. automodule:: langchain.document_transformers
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: document_transformers
    :template: class.rst

    document_transformers.EmbeddingsRedundantFilter

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: document_transformers

    document_transformers.get_stateful_documents

:mod:`langchain.embeddings`: Embeddings
========================================

.. automodule:: langchain.embeddings
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: embeddings
    :template: class.rst

    embeddings.aleph_alpha.AlephAlphaAsymmetricSemanticEmbedding
    embeddings.aleph_alpha.AlephAlphaSymmetricSemanticEmbedding
    embeddings.base.Embeddings
    embeddings.bedrock.BedrockEmbeddings
    embeddings.cohere.CohereEmbeddings
    embeddings.dashscope.DashScopeEmbeddings
    embeddings.deepinfra.DeepInfraEmbeddings
    embeddings.elasticsearch.ElasticsearchEmbeddings
    embeddings.embaas.EmbaasEmbeddings
    embeddings.embaas.EmbaasEmbeddingsPayload
    embeddings.fake.FakeEmbeddings
    embeddings.google_palm.GooglePalmEmbeddings
    embeddings.huggingface.HuggingFaceEmbeddings
    embeddings.huggingface.HuggingFaceInstructEmbeddings
    embeddings.huggingface_hub.HuggingFaceHubEmbeddings
    embeddings.jina.JinaEmbeddings
    embeddings.llamacpp.LlamaCppEmbeddings
    embeddings.minimax.MiniMaxEmbeddings
    embeddings.modelscope_hub.ModelScopeEmbeddings
    embeddings.mosaicml.MosaicMLInstructorEmbeddings
    embeddings.octoai_embeddings.OctoAIEmbeddings
    embeddings.openai.OpenAIEmbeddings
    embeddings.sagemaker_endpoint.EmbeddingsContentHandler
    embeddings.sagemaker_endpoint.SagemakerEndpointEmbeddings
    embeddings.self_hosted.SelfHostedEmbeddings
    embeddings.self_hosted_hugging_face.SelfHostedHuggingFaceEmbeddings
    embeddings.self_hosted_hugging_face.SelfHostedHuggingFaceInstructEmbeddings
    embeddings.tensorflow_hub.TensorflowHubEmbeddings
    embeddings.vertexai.VertexAIEmbeddings

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: embeddings

    embeddings.dashscope.embed_with_retry
    embeddings.google_palm.embed_with_retry
    embeddings.minimax.embed_with_retry
    embeddings.openai.embed_with_retry
    embeddings.self_hosted_hugging_face.load_embedding_model

:mod:`langchain.env`: Env
==========================

.. automodule:: langchain.env
    :no-members:
    :no-inherited-members:

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: env

    env.get_runtime_environment

:mod:`langchain.evaluation`: Evaluation
========================================

.. automodule:: langchain.evaluation
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: evaluation
    :template: class.rst

    evaluation.agents.trajectory_eval_chain.TrajectoryEval
    evaluation.agents.trajectory_eval_chain.TrajectoryEvalChain
    evaluation.agents.trajectory_eval_chain.TrajectoryOutputParser
    evaluation.comparison.eval_chain.PairwiseStringEvalChain
    evaluation.comparison.eval_chain.PairwiseStringResultOutputParser
    evaluation.criteria.eval_chain.CriteriaEvalChain
    evaluation.criteria.eval_chain.CriteriaResultOutputParser
    evaluation.qa.eval_chain.ContextQAEvalChain
    evaluation.qa.eval_chain.CotQAEvalChain
    evaluation.qa.eval_chain.QAEvalChain
    evaluation.qa.generate_chain.QAGenerateChain
    evaluation.run_evaluators.base.RunEvaluatorChain
    evaluation.run_evaluators.base.RunEvaluatorOutputParser
    evaluation.run_evaluators.implementations.ChoicesOutputParser
    evaluation.run_evaluators.implementations.CriteriaOutputParser
    evaluation.run_evaluators.implementations.StringRunEvaluatorInputMapper
    evaluation.run_evaluators.implementations.TrajectoryEvalOutputParser
    evaluation.run_evaluators.implementations.TrajectoryInputMapper
    evaluation.schema.PairwiseStringEvaluator
    evaluation.schema.StringEvaluator

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: evaluation

    evaluation.loading.load_dataset
    evaluation.run_evaluators.implementations.get_criteria_evaluator
    evaluation.run_evaluators.implementations.get_qa_evaluator
    evaluation.run_evaluators.implementations.get_trajectory_evaluator

:mod:`langchain.example_generator`: Example Generator
======================================================

.. automodule:: langchain.example_generator
    :no-members:
    :no-inherited-members:

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: example_generator

    example_generator.generate_example

:mod:`langchain.experimental`: Experimental
============================================

.. automodule:: langchain.experimental
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: experimental
    :template: class.rst

    experimental.autonomous_agents.autogpt.memory.AutoGPTMemory
    experimental.autonomous_agents.autogpt.output_parser.AutoGPTAction
    experimental.autonomous_agents.autogpt.output_parser.AutoGPTOutputParser
    experimental.autonomous_agents.autogpt.output_parser.BaseAutoGPTOutputParser
    experimental.autonomous_agents.autogpt.prompt.AutoGPTPrompt
    experimental.autonomous_agents.baby_agi.baby_agi.BabyAGI
    experimental.autonomous_agents.baby_agi.task_creation.TaskCreationChain
    experimental.autonomous_agents.baby_agi.task_execution.TaskExecutionChain
    experimental.autonomous_agents.baby_agi.task_prioritization.TaskPrioritizationChain
    experimental.generative_agents.generative_agent.GenerativeAgent
    experimental.generative_agents.memory.GenerativeAgentMemory
    experimental.llms.jsonformer_decoder.JsonFormer
    experimental.llms.rellm_decoder.RELLM
    experimental.plan_and_execute.agent_executor.PlanAndExecute
    experimental.plan_and_execute.executors.base.BaseExecutor
    experimental.plan_and_execute.executors.base.ChainExecutor
    experimental.plan_and_execute.planners.base.BasePlanner
    experimental.plan_and_execute.planners.base.LLMPlanner
    experimental.plan_and_execute.planners.chat_planner.PlanningOutputParser
    experimental.plan_and_execute.schema.BaseStepContainer
    experimental.plan_and_execute.schema.ListStepContainer
    experimental.plan_and_execute.schema.Plan
    experimental.plan_and_execute.schema.PlanOutputParser
    experimental.plan_and_execute.schema.Step
    experimental.plan_and_execute.schema.StepResponse

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: experimental

    experimental.autonomous_agents.autogpt.output_parser.preprocess_json_input
    experimental.autonomous_agents.autogpt.prompt_generator.get_prompt
    experimental.llms.jsonformer_decoder.import_jsonformer
    experimental.llms.rellm_decoder.import_rellm
    experimental.plan_and_execute.executors.agent_executor.load_agent_executor
    experimental.plan_and_execute.planners.chat_planner.load_chat_planner

:mod:`langchain.formatting`: Formatting
========================================

.. automodule:: langchain.formatting
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: formatting
    :template: class.rst

    formatting.StrictFormatter

:mod:`langchain.graphs`: Graphs
================================

.. automodule:: langchain.graphs
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: graphs
    :template: class.rst

    graphs.networkx_graph.KnowledgeTriple

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: graphs

    graphs.networkx_graph.get_entities
    graphs.networkx_graph.parse_triples

:mod:`langchain.indexes`: Indexes
==================================

.. automodule:: langchain.indexes
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: indexes
    :template: class.rst

    indexes.graph.GraphIndexCreator
    indexes.vectorstore.VectorStoreIndexWrapper
    indexes.vectorstore.VectorstoreIndexCreator

:mod:`langchain.input`: Input
==============================

.. automodule:: langchain.input
    :no-members:
    :no-inherited-members:

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: input

    input.get_bolded_text
    input.get_color_mapping
    input.get_colored_text
    input.print_text

:mod:`langchain.llms`: LLMs
============================

.. automodule:: langchain.llms
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: llms
    :template: class.rst

    llms.ai21.AI21
    llms.ai21.AI21PenaltyData
    llms.aleph_alpha.AlephAlpha
    llms.amazon_api_gateway.AmazonAPIGateway
    llms.anthropic.Anthropic
    llms.anyscale.Anyscale
    llms.aviary.Aviary
    llms.azureml_endpoint.AzureMLEndpointClient
    llms.azureml_endpoint.AzureMLOnlineEndpoint
    llms.azureml_endpoint.DollyContentFormatter
    llms.azureml_endpoint.HFContentFormatter
    llms.azureml_endpoint.OSSContentFormatter
    llms.bananadev.Banana
    llms.base.BaseLLM
    llms.base.LLM
    llms.baseten.Baseten
    llms.beam.Beam
    llms.bedrock.Bedrock
    llms.cerebriumai.CerebriumAI
    llms.clarifai.Clarifai
    llms.cohere.Cohere
    llms.ctransformers.CTransformers
    llms.databricks.Databricks
    llms.deepinfra.DeepInfra
    llms.fake.FakeListLLM
    llms.forefrontai.ForefrontAI
    llms.google_palm.GooglePalm
    llms.gooseai.GooseAI
    llms.gpt4all.GPT4All
    llms.huggingface_endpoint.HuggingFaceEndpoint
    llms.huggingface_hub.HuggingFaceHub
    llms.huggingface_pipeline.HuggingFacePipeline
    llms.huggingface_text_gen_inference.HuggingFaceTextGenInference
    llms.human.HumanInputLLM
    llms.llamacpp.LlamaCpp
    llms.manifest.ManifestWrapper
    llms.modal.Modal
    llms.mosaicml.MosaicML
    llms.nlpcloud.NLPCloud
    llms.octoai_endpoint.OctoAIEndpoint
    llms.openai.AzureOpenAI
    llms.openai.BaseOpenAI
    llms.openai.OpenAI
    llms.openai.OpenAIChat
    llms.openllm.IdentifyingParams
    llms.openllm.OpenLLM
    llms.openlm.OpenLM
    llms.petals.Petals
    llms.pipelineai.PipelineAI
    llms.predictionguard.PredictionGuard
    llms.promptlayer_openai.PromptLayerOpenAI
    llms.promptlayer_openai.PromptLayerOpenAIChat
    llms.replicate.Replicate
    llms.rwkv.RWKV
    llms.sagemaker_endpoint.ContentHandlerBase
    llms.sagemaker_endpoint.LLMContentHandler
    llms.sagemaker_endpoint.SagemakerEndpoint
    llms.self_hosted.SelfHostedPipeline
    llms.self_hosted_hugging_face.SelfHostedHuggingFaceLLM
    llms.stochasticai.StochasticAI
    llms.textgen.TextGen
    llms.vertexai.VertexAI
    llms.writer.Writer

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: llms

    llms.aviary.get_completions
    llms.aviary.get_models
    llms.base.get_prompts
    llms.base.update_cache
    llms.cohere.completion_with_retry
    llms.databricks.get_default_api_token
    llms.databricks.get_default_host
    llms.databricks.get_repl_context
    llms.google_palm.generate_with_retry
    llms.loading.load_llm
    llms.loading.load_llm_from_config
    llms.openai.completion_with_retry
    llms.openai.update_token_usage
    llms.utils.enforce_stop_tokens
    llms.vertexai.is_codey_model

:mod:`langchain.load`: Load
============================

.. automodule:: langchain.load
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: load
    :template: class.rst

    load.serializable.BaseSerialized
    load.serializable.Serializable
    load.serializable.SerializedConstructor
    load.serializable.SerializedNotImplemented
    load.serializable.SerializedSecret

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: load

    load.dump.default
    load.dump.dumpd
    load.dump.dumps
    load.load.loads
    load.serializable.to_json_not_implemented

:mod:`langchain.math_utils`: Math Utils
========================================

.. automodule:: langchain.math_utils
    :no-members:
    :no-inherited-members:

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: math_utils

    math_utils.cosine_similarity
    math_utils.cosine_similarity_top_k

:mod:`langchain.memory`: Memory
================================

.. automodule:: langchain.memory
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: memory
    :template: class.rst

    memory.buffer.ConversationBufferMemory
    memory.buffer.ConversationStringBufferMemory
    memory.buffer_window.ConversationBufferWindowMemory
    memory.chat_memory.BaseChatMemory
    memory.chat_message_histories.cassandra.CassandraChatMessageHistory
    memory.chat_message_histories.cosmos_db.CosmosDBChatMessageHistory
    memory.chat_message_histories.dynamodb.DynamoDBChatMessageHistory
    memory.chat_message_histories.file.FileChatMessageHistory
    memory.chat_message_histories.firestore.FirestoreChatMessageHistory
    memory.chat_message_histories.in_memory.ChatMessageHistory
    memory.chat_message_histories.momento.MomentoChatMessageHistory
    memory.chat_message_histories.mongodb.MongoDBChatMessageHistory
    memory.chat_message_histories.postgres.PostgresChatMessageHistory
    memory.chat_message_histories.redis.RedisChatMessageHistory
    memory.chat_message_histories.sql.SQLChatMessageHistory
    memory.chat_message_histories.zep.ZepChatMessageHistory
    memory.combined.CombinedMemory
    memory.entity.BaseEntityStore
    memory.entity.ConversationEntityMemory
    memory.entity.InMemoryEntityStore
    memory.entity.RedisEntityStore
    memory.entity.SQLiteEntityStore
    memory.kg.ConversationKGMemory
    memory.motorhead_memory.MotorheadMemory
    memory.readonly.ReadOnlySharedMemory
    memory.simple.SimpleMemory
    memory.summary.ConversationSummaryMemory
    memory.summary.SummarizerMixin
    memory.summary_buffer.ConversationSummaryBufferMemory
    memory.token_buffer.ConversationTokenBufferMemory
    memory.vectorstore.VectorStoreRetrieverMemory

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: memory

    memory.chat_message_histories.sql.create_message_model
    memory.utils.get_prompt_input_key

:mod:`langchain.output_parsers`: Output Parsers
================================================

.. automodule:: langchain.output_parsers
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: output_parsers
    :template: class.rst

    output_parsers.boolean.BooleanOutputParser
    output_parsers.combining.CombiningOutputParser
    output_parsers.datetime.DatetimeOutputParser
    output_parsers.enum.EnumOutputParser
    output_parsers.fix.OutputFixingParser
    output_parsers.list.CommaSeparatedListOutputParser
    output_parsers.list.ListOutputParser
    output_parsers.openai_functions.JsonKeyOutputFunctionsParser
    output_parsers.openai_functions.JsonOutputFunctionsParser
    output_parsers.openai_functions.OutputFunctionsParser
    output_parsers.openai_functions.PydanticAttrOutputFunctionsParser
    output_parsers.openai_functions.PydanticOutputFunctionsParser
    output_parsers.pydantic.PydanticOutputParser
    output_parsers.rail_parser.GuardrailsOutputParser
    output_parsers.regex.RegexParser
    output_parsers.regex_dict.RegexDictParser
    output_parsers.retry.RetryOutputParser
    output_parsers.retry.RetryWithErrorOutputParser
    output_parsers.structured.ResponseSchema
    output_parsers.structured.StructuredOutputParser

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: output_parsers

    output_parsers.json.parse_and_check_json_markdown
    output_parsers.json.parse_json_markdown
    output_parsers.loading.load_output_parser

:mod:`langchain.prompts`: Prompts
==================================

.. automodule:: langchain.prompts
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: prompts
    :template: class.rst

    prompts.base.BasePromptTemplate
    prompts.base.StringPromptTemplate
    prompts.base.StringPromptValue
    prompts.chat.AIMessagePromptTemplate
    prompts.chat.BaseChatPromptTemplate
    prompts.chat.BaseMessagePromptTemplate
    prompts.chat.BaseStringMessagePromptTemplate
    prompts.chat.ChatMessagePromptTemplate
    prompts.chat.ChatPromptTemplate
    prompts.chat.ChatPromptValue
    prompts.chat.HumanMessagePromptTemplate
    prompts.chat.MessagesPlaceholder
    prompts.chat.SystemMessagePromptTemplate
    prompts.example_selector.base.BaseExampleSelector
    prompts.example_selector.length_based.LengthBasedExampleSelector
    prompts.example_selector.ngram_overlap.NGramOverlapExampleSelector
    prompts.example_selector.semantic_similarity.MaxMarginalRelevanceExampleSelector
    prompts.example_selector.semantic_similarity.SemanticSimilarityExampleSelector
    prompts.few_shot.FewShotPromptTemplate
    prompts.few_shot_with_templates.FewShotPromptWithTemplates
    prompts.pipeline.PipelinePromptTemplate
    prompts.prompt.PromptTemplate

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: prompts

    prompts.base.check_valid_template
    prompts.base.jinja2_formatter
    prompts.base.validate_jinja2
    prompts.example_selector.ngram_overlap.ngram_overlap_score
    prompts.example_selector.semantic_similarity.sorted_values
    prompts.loading.load_prompt
    prompts.loading.load_prompt_from_config

:mod:`langchain.requests`: Requests
====================================

.. automodule:: langchain.requests
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: requests
    :template: class.rst

    requests.Requests
    requests.TextRequestsWrapper

:mod:`langchain.retrievers`: Retrievers
========================================

.. automodule:: langchain.retrievers
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: retrievers
    :template: class.rst

    retrievers.arxiv.ArxivRetriever
    retrievers.azure_cognitive_search.AzureCognitiveSearchRetriever
    retrievers.chatgpt_plugin_retriever.ChatGPTPluginRetriever
    retrievers.contextual_compression.ContextualCompressionRetriever
    retrievers.chaindesk.ChaindeskRetriever
    retrievers.docarray.DocArrayRetriever
    retrievers.docarray.SearchType
    retrievers.document_compressors.base.BaseDocumentCompressor
    retrievers.document_compressors.base.DocumentCompressorPipeline
    retrievers.document_compressors.chain_extract.LLMChainExtractor
    retrievers.document_compressors.chain_extract.NoOutputParser
    retrievers.document_compressors.chain_filter.LLMChainFilter
    retrievers.document_compressors.cohere_rerank.CohereRerank
    retrievers.document_compressors.embeddings_filter.EmbeddingsFilter
    retrievers.elastic_search_bm25.ElasticSearchBM25Retriever
    retrievers.kendra.AdditionalResultAttribute
    retrievers.kendra.AdditionalResultAttributeValue
    retrievers.kendra.AmazonKendraRetriever
    retrievers.kendra.DocumentAttribute
    retrievers.kendra.DocumentAttributeValue
    retrievers.kendra.Highlight
    retrievers.kendra.QueryResult
    retrievers.kendra.QueryResultItem
    retrievers.kendra.RetrieveResult
    retrievers.kendra.RetrieveResultItem
    retrievers.kendra.TextWithHighLights
    retrievers.knn.KNNRetriever
    retrievers.llama_index.LlamaIndexGraphRetriever
    retrievers.llama_index.LlamaIndexRetriever
    retrievers.merger_retriever.MergerRetriever
    retrievers.metal.MetalRetriever
    retrievers.milvus.MilvusRetriever
    retrievers.multi_query.LineList
    retrievers.multi_query.LineListOutputParser
    retrievers.multi_query.MultiQueryRetriever
    retrievers.pinecone_hybrid_search.PineconeHybridSearchRetriever
    retrievers.pupmed.PubMedRetriever
    retrievers.remote_retriever.RemoteLangChainRetriever
    retrievers.self_query.base.SelfQueryRetriever
    retrievers.self_query.chroma.ChromaTranslator
    retrievers.self_query.myscale.MyScaleTranslator
    retrievers.self_query.pinecone.PineconeTranslator
    retrievers.self_query.qdrant.QdrantTranslator
    retrievers.self_query.weaviate.WeaviateTranslator
    retrievers.svm.SVMRetriever
    retrievers.tfidf.TFIDFRetriever
    retrievers.time_weighted_retriever.TimeWeightedVectorStoreRetriever
    retrievers.vespa_retriever.VespaRetriever
    retrievers.weaviate_hybrid_search.WeaviateHybridSearchRetriever
    retrievers.wikipedia.WikipediaRetriever
    retrievers.zep.ZepRetriever
    retrievers.zilliz.ZillizRetriever

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: retrievers

    retrievers.document_compressors.chain_extract.default_get_input
    retrievers.document_compressors.chain_filter.default_get_input
    retrievers.kendra.clean_excerpt
    retrievers.kendra.combined_text
    retrievers.knn.create_index
    retrievers.milvus.MilvusRetreiver
    retrievers.pinecone_hybrid_search.create_index
    retrievers.pinecone_hybrid_search.hash_text
    retrievers.self_query.myscale.DEFAULT_COMPOSER
    retrievers.self_query.myscale.FUNCTION_COMPOSER
    retrievers.svm.create_index
    retrievers.zilliz.ZillizRetreiver

:mod:`langchain.schema`: Schema
================================

.. automodule:: langchain.schema
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: schema
    :template: class.rst

    schema.AIMessage
    schema.AgentFinish
    schema.BaseChatMessageHistory
    schema.BaseDocumentTransformer
    schema.BaseLLMOutputParser
    schema.BaseMemory
    schema.BaseMessage
    schema.BaseOutputParser
    schema.BaseRetriever
    schema.ChatGeneration
    schema.ChatMessage
    schema.ChatResult
    schema.Document
    schema.FunctionMessage
    schema.Generation
    schema.HumanMessage
    schema.LLMResult
    schema.NoOpOutputParser
    schema.OutputParserException
    schema.PromptValue
    schema.RunInfo
    schema.SystemMessage

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: schema

    schema.get_buffer_string
    schema.messages_from_dict
    schema.messages_to_dict

:mod:`langchain.server`: Server
================================

.. automodule:: langchain.server
    :no-members:
    :no-inherited-members:

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: server

    server.main

:mod:`langchain.sql_database`: Sql Database
============================================

.. automodule:: langchain.sql_database
    :no-members:
    :no-inherited-members:

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: sql_database

    sql_database.truncate_word

:mod:`langchain.text_splitter`: Text Splitter
==============================================

.. automodule:: langchain.text_splitter
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: text_splitter
    :template: class.rst

    text_splitter.CharacterTextSplitter
    text_splitter.HeaderType
    text_splitter.Language
    text_splitter.LatexTextSplitter
    text_splitter.LineType
    text_splitter.MarkdownTextSplitter
    text_splitter.NLTKTextSplitter
    text_splitter.PythonCodeTextSplitter
    text_splitter.RecursiveCharacterTextSplitter
    text_splitter.SentenceTransformersTokenTextSplitter
    text_splitter.SpacyTextSplitter
    text_splitter.TextSplitter
    text_splitter.TokenTextSplitter

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: text_splitter

    text_splitter.split_text_on_tokens

:mod:`langchain.tools`: Tools
==============================

.. automodule:: langchain.tools
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: tools
    :template: class.rst

    tools.arxiv.tool.ArxivQueryRun
    tools.azure_cognitive_services.form_recognizer.AzureCogsFormRecognizerTool
    tools.azure_cognitive_services.image_analysis.AzureCogsImageAnalysisTool
    tools.azure_cognitive_services.speech2text.AzureCogsSpeech2TextTool
    tools.azure_cognitive_services.text2speech.AzureCogsText2SpeechTool
    tools.base.BaseTool
    tools.base.ChildTool
    tools.base.SchemaAnnotationError
    tools.base.StructuredTool
    tools.base.Tool
    tools.base.ToolException
    tools.base.ToolMetaclass
    tools.bing_search.tool.BingSearchResults
    tools.bing_search.tool.BingSearchRun
    tools.brave_search.tool.BraveSearch
    tools.convert_to_openai.FunctionDescription
    tools.ddg_search.tool.DuckDuckGoSearchResults
    tools.ddg_search.tool.DuckDuckGoSearchRun
    tools.file_management.copy.CopyFileTool
    tools.file_management.copy.FileCopyInput
    tools.file_management.delete.DeleteFileTool
    tools.file_management.delete.FileDeleteInput
    tools.file_management.file_search.FileSearchInput
    tools.file_management.file_search.FileSearchTool
    tools.file_management.list_dir.DirectoryListingInput
    tools.file_management.list_dir.ListDirectoryTool
    tools.file_management.move.FileMoveInput
    tools.file_management.move.MoveFileTool
    tools.file_management.read.ReadFileInput
    tools.file_management.read.ReadFileTool
    tools.file_management.utils.BaseFileToolMixin
    tools.file_management.utils.FileValidationError
    tools.file_management.write.WriteFileInput
    tools.file_management.write.WriteFileTool
    tools.gmail.base.GmailBaseTool
    tools.gmail.create_draft.CreateDraftSchema
    tools.gmail.create_draft.GmailCreateDraft
    tools.gmail.get_message.GmailGetMessage
    tools.gmail.get_message.SearchArgsSchema
    tools.gmail.get_thread.GetThreadSchema
    tools.gmail.get_thread.GmailGetThread
    tools.gmail.search.GmailSearch
    tools.gmail.search.Resource
    tools.gmail.search.SearchArgsSchema
    tools.gmail.send_message.GmailSendMessage
    tools.gmail.send_message.SendMessageSchema
    tools.google_places.tool.GooglePlacesSchema
    tools.google_places.tool.GooglePlacesTool
    tools.google_search.tool.GoogleSearchResults
    tools.google_search.tool.GoogleSearchRun
    tools.google_serper.tool.GoogleSerperResults
    tools.google_serper.tool.GoogleSerperRun
    tools.graphql.tool.BaseGraphQLTool
    tools.human.tool.HumanInputRun
    tools.ifttt.IFTTTWebhook
    tools.jira.tool.JiraAction
    tools.json.tool.JsonGetValueTool
    tools.json.tool.JsonListKeysTool
    tools.json.tool.JsonSpec
    tools.metaphor_search.tool.MetaphorSearchResults
    tools.office365.base.O365BaseTool
    tools.office365.create_draft_message.CreateDraftMessageSchema
    tools.office365.create_draft_message.O365CreateDraftMessage
    tools.office365.events_search.O365SearchEvents
    tools.office365.events_search.SearchEventsInput
    tools.office365.messages_search.O365SearchEmails
    tools.office365.messages_search.SearchEmailsInput
    tools.office365.send_event.O365SendEvent
    tools.office365.send_event.SendEventSchema
    tools.office365.send_message.O365SendMessage
    tools.office365.send_message.SendMessageSchema
    tools.openapi.utils.api_models.APIOperation
    tools.openapi.utils.api_models.APIProperty
    tools.openapi.utils.api_models.APIPropertyBase
    tools.openapi.utils.api_models.APIPropertyLocation
    tools.openapi.utils.api_models.APIRequestBody
    tools.openapi.utils.api_models.APIRequestBodyProperty
    tools.openweathermap.tool.OpenWeatherMapQueryRun
    tools.playwright.base.BaseBrowserTool
    tools.playwright.click.ClickTool
    tools.playwright.click.ClickToolInput
    tools.playwright.current_page.CurrentWebPageTool
    tools.playwright.extract_hyperlinks.ExtractHyperlinksTool
    tools.playwright.extract_hyperlinks.ExtractHyperlinksToolInput
    tools.playwright.extract_text.ExtractTextTool
    tools.playwright.get_elements.GetElementsTool
    tools.playwright.get_elements.GetElementsToolInput
    tools.playwright.navigate.NavigateTool
    tools.playwright.navigate.NavigateToolInput
    tools.playwright.navigate_back.NavigateBackTool
    tools.plugin.AIPlugin
    tools.plugin.AIPluginTool
    tools.plugin.AIPluginToolSchema
    tools.plugin.ApiConfig
    tools.powerbi.tool.InfoPowerBITool
    tools.powerbi.tool.ListPowerBITool
    tools.powerbi.tool.QueryPowerBITool
    tools.pubmed.tool.PubmedQueryRun
    tools.python.tool.PythonAstREPLTool
    tools.python.tool.PythonREPLTool
    tools.requests.tool.BaseRequestsTool
    tools.requests.tool.RequestsDeleteTool
    tools.requests.tool.RequestsGetTool
    tools.requests.tool.RequestsPatchTool
    tools.requests.tool.RequestsPostTool
    tools.requests.tool.RequestsPutTool
    tools.scenexplain.tool.SceneXplainInput
    tools.scenexplain.tool.SceneXplainTool
    tools.searx_search.tool.SearxSearchResults
    tools.searx_search.tool.SearxSearchRun
    tools.shell.tool.ShellInput
    tools.shell.tool.ShellTool
    tools.sleep.tool.SleepInput
    tools.sleep.tool.SleepTool
    tools.spark_sql.tool.BaseSparkSQLTool
    tools.spark_sql.tool.InfoSparkSQLTool
    tools.spark_sql.tool.ListSparkSQLTool
    tools.spark_sql.tool.QueryCheckerTool
    tools.spark_sql.tool.QuerySparkSQLTool
    tools.sql_database.tool.BaseSQLDatabaseTool
    tools.sql_database.tool.InfoSQLDatabaseTool
    tools.sql_database.tool.ListSQLDatabaseTool
    tools.sql_database.tool.QuerySQLCheckerTool
    tools.sql_database.tool.QuerySQLDataBaseTool
    tools.steamship_image_generation.tool.ModelName
    tools.steamship_image_generation.tool.SteamshipImageGenerationTool
    tools.vectorstore.tool.BaseVectorStoreTool
    tools.vectorstore.tool.VectorStoreQATool
    tools.vectorstore.tool.VectorStoreQAWithSourcesTool
    tools.wikipedia.tool.WikipediaQueryRun
    tools.wolfram_alpha.tool.WolframAlphaQueryRun
    tools.youtube.search.YouTubeSearchTool
    tools.zapier.tool.ZapierNLAListActions
    tools.zapier.tool.ZapierNLARunAction

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: tools

    tools.azure_cognitive_services.utils.detect_file_src_type
    tools.azure_cognitive_services.utils.download_audio_from_url
    tools.base.create_schema_from_function
    tools.base.tool
    tools.convert_to_openai.format_tool_to_openai_function
    tools.ddg_search.tool.DuckDuckGoSearchTool
    tools.file_management.utils.get_validated_relative_path
    tools.file_management.utils.is_relative_to
    tools.gmail.utils.build_resource_service
    tools.gmail.utils.clean_email_body
    tools.gmail.utils.get_gmail_credentials
    tools.gmail.utils.import_google
    tools.gmail.utils.import_googleapiclient_resource_builder
    tools.gmail.utils.import_installed_app_flow
    tools.interaction.tool.StdInInquireTool
    tools.office365.utils.authenticate
    tools.office365.utils.clean_body
    tools.playwright.base.lazy_import_playwright_browsers
    tools.playwright.utils.create_async_playwright_browser
    tools.playwright.utils.create_sync_playwright_browser
    tools.playwright.utils.get_current_page
    tools.playwright.utils.run_async
    tools.plugin.marshal_spec
    tools.python.tool.sanitize_input
    tools.steamship_image_generation.utils.make_image_public

:mod:`langchain.utilities`: Utilities
======================================

.. automodule:: langchain.utilities
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: utilities
    :template: class.rst

    utilities.apify.ApifyWrapper
    utilities.arxiv.ArxivAPIWrapper
    utilities.awslambda.LambdaWrapper
    utilities.bibtex.BibtexparserWrapper
    utilities.bing_search.BingSearchAPIWrapper
    utilities.brave_search.BraveSearchWrapper
    utilities.duckduckgo_search.DuckDuckGoSearchAPIWrapper
    utilities.google_places_api.GooglePlacesAPIWrapper
    utilities.google_search.GoogleSearchAPIWrapper
    utilities.google_serper.GoogleSerperAPIWrapper
    utilities.graphql.GraphQLAPIWrapper
    utilities.jira.JiraAPIWrapper
    utilities.metaphor_search.MetaphorSearchAPIWrapper
    utilities.openapi.HTTPVerb
    utilities.openapi.OpenAPISpec
    utilities.openweathermap.OpenWeatherMapAPIWrapper
    utilities.powerbi.PowerBIDataset
    utilities.pupmed.PubMedAPIWrapper
    utilities.python.PythonREPL
    utilities.scenexplain.SceneXplainAPIWrapper
    utilities.searx_search.SearxResults
    utilities.searx_search.SearxSearchWrapper
    utilities.serpapi.SerpAPIWrapper
    utilities.twilio.TwilioAPIWrapper
    utilities.wikipedia.WikipediaAPIWrapper
    utilities.wolfram_alpha.WolframAlphaAPIWrapper
    utilities.zapier.ZapierNLAWrapper

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: utilities

    utilities.loading.try_load_from_hub
    utilities.powerbi.fix_table_name
    utilities.powerbi.json_to_md
    utilities.vertexai.init_vertexai
    utilities.vertexai.raise_vertex_import_error

:mod:`langchain.utils`: Utils
==============================

.. automodule:: langchain.utils
    :no-members:
    :no-inherited-members:

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: utils

    utils.comma_list
    utils.get_from_dict_or_env
    utils.get_from_env
    utils.guard_import
    utils.mock_now
    utils.raise_for_status_with_text
    utils.stringify_dict
    utils.stringify_value
    utils.xor_args

:mod:`langchain.vectorstores`: Vectorstores
============================================

.. automodule:: langchain.vectorstores
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: vectorstores
    :template: class.rst

    vectorstores.alibabacloud_opensearch.AlibabaCloudOpenSearch
    vectorstores.analyticdb.AnalyticDB
    vectorstores.annoy.Annoy
    vectorstores.atlas.AtlasDB
    vectorstores.awadb.AwaDB
    vectorstores.azuresearch.AzureSearch
    vectorstores.azuresearch.AzureSearchVectorStoreRetriever
    vectorstores.base.VectorStore
    vectorstores.base.VectorStoreRetriever
    vectorstores.cassandra.Cassandra
    vectorstores.chroma.Chroma
    vectorstores.clarifai.Clarifai
    vectorstores.clickhouse.Clickhouse
    vectorstores.clickhouse.ClickhouseSettings
    vectorstores.deeplake.DeepLake
    vectorstores.docarray.base.DocArrayIndex
    vectorstores.docarray.hnsw.DocArrayHnswSearch
    vectorstores.docarray.in_memory.DocArrayInMemorySearch
    vectorstores.elastic_vector_search.ElasticKnnSearch
    vectorstores.elastic_vector_search.ElasticVectorSearch
    vectorstores.faiss.FAISS
    vectorstores.hologres.Hologres
    vectorstores.lancedb.LanceDB
    vectorstores.matching_engine.MatchingEngine
    vectorstores.milvus.Milvus
    vectorstores.mongodb_atlas.MongoDBAtlasVectorSearch
    vectorstores.myscale.MyScale
    vectorstores.myscale.MyScaleSettings
    vectorstores.opensearch_vector_search.OpenSearchVectorSearch
    vectorstores.pgvector.BaseModel
    vectorstores.pgvector.CollectionStore
    vectorstores.pgvector.DistanceStrategy
    vectorstores.pgvector.EmbeddingStore
    vectorstores.pgvector.PGVector
    vectorstores.pinecone.Pinecone
    vectorstores.qdrant.Qdrant
    vectorstores.redis.Redis
    vectorstores.redis.RedisVectorStoreRetriever
    vectorstores.rocksetdb.Rockset
    vectorstores.singlestoredb.DistanceStrategy
    vectorstores.singlestoredb.SingleStoreDB
    vectorstores.singlestoredb.SingleStoreDBRetriever
    vectorstores.sklearn.BaseSerializer
    vectorstores.sklearn.BsonSerializer
    vectorstores.sklearn.JsonSerializer
    vectorstores.sklearn.ParquetSerializer
    vectorstores.sklearn.SKLearnVectorStore
    vectorstores.sklearn.SKLearnVectorStoreException
    vectorstores.starrocks.StarRocks
    vectorstores.starrocks.StarRocksSettings
    vectorstores.supabase.SupabaseVectorStore
    vectorstores.tair.Tair
    vectorstores.tigris.Tigris
    vectorstores.typesense.Typesense
    vectorstores.vectara.Vectara
    vectorstores.vectara.VectaraRetriever
    vectorstores.weaviate.Weaviate
    vectorstores.zilliz.Zilliz

Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: vectorstores

    vectorstores.alibabacloud_opensearch.create_metadata
    vectorstores.annoy.dependable_annoy_import
    vectorstores.clickhouse.has_mul_sub_str
    vectorstores.faiss.dependable_faiss_import
    vectorstores.myscale.has_mul_sub_str
    vectorstores.starrocks.debug_output
    vectorstores.starrocks.get_named_result
    vectorstores.starrocks.has_mul_sub_str
    vectorstores.utils.maximal_marginal_relevance

