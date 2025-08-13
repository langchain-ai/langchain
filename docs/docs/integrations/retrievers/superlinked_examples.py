"""
SuperlinkedRetriever Usage Examples

This file demonstrates how to use the SuperlinkedRetriever with different
space configurations to showcase its flexibility across various use cases.
"""
# ruff: noqa: T201, E501
# mypy: ignore-errors

from datetime import datetime, timedelta

import superlinked.framework as sl

from langchain_superlinked import SuperlinkedRetriever


def example_1_simple_text_search():
    """
    Example 1: Simple text-based semantic search
    Use case: Basic document retrieval based on content similarity
    """
    print("=== Example 1: Simple Text Search ===")

    # 1. Define Schema
    class DocumentSchema(sl.Schema):
        id: sl.IdField
        content: sl.String

    doc_schema = DocumentSchema()

    # 2. Define Space and Index
    text_space = sl.TextSimilaritySpace(
        text=doc_schema.content, model="sentence-transformers/all-MiniLM-L6-v2"
    )

    doc_index = sl.Index([text_space])

    # 3. Define Query
    query = (
        sl.Query(doc_index)
        .find(doc_schema)
        .similar(text_space.text, sl.Param("query_text"))
        .select([doc_schema.content])
        .limit(sl.Param("limit"))
    )

    # 4. Set up data and app using executor pattern
    documents = [
        {
            "id": "doc1",
            "content": "Machine learning algorithms can process large datasets efficiently.",
        },
        {
            "id": "doc2",
            "content": "Natural language processing enables computers to understand human language.",
        },
        {
            "id": "doc3",
            "content": "Deep learning models require significant computational resources.",
        },
        {
            "id": "doc4",
            "content": "Data science combines statistics, programming, and domain expertise.",
        },
        {
            "id": "doc5",
            "content": "Artificial intelligence is transforming various industries.",
        },
    ]

    # Create source and executor
    source = sl.InMemorySource(schema=doc_schema)
    executor = sl.InMemoryExecutor(sources=[source], indices=[doc_index])
    app = executor.run()

    # Add data to the source after the app is running
    source.put(documents)

    # 5. Create Retriever
    retriever = SuperlinkedRetriever(
        sl_client=app, sl_query=query, page_content_field="content"
    )

    # 6. Use the retriever
    results = retriever.invoke("artificial intelligence and machine learning", limit=3)

    print("Query: 'artificial intelligence and machine learning'")
    print(f"Found {len(results)} documents:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.page_content}")
    print()


def example_2_multi_space_blog_search():
    """
    Example 2: Multi-space blog post search
    Use case: Blog search with content, category, and recency
    """
    print("=== Example 2: Multi-Space Blog Search ===")

    # 1. Define Schema
    class BlogPostSchema(sl.Schema):
        id: sl.IdField
        title: sl.String
        content: sl.String
        category: sl.String
        published_date: sl.Timestamp
        view_count: sl.Integer

    blog_schema = BlogPostSchema()

    # 2. Define Multiple Spaces
    # Text similarity for content
    content_space = sl.TextSimilaritySpace(
        text=blog_schema.content, model="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Title similarity
    title_space = sl.TextSimilaritySpace(
        text=blog_schema.title, model="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Category similarity
    category_space = sl.CategoricalSimilaritySpace(
        category_input=blog_schema.category,
        categories=["technology", "science", "business", "health", "travel"],
    )

    # Recency (favor recent posts)
    recency_space = sl.RecencySpace(
        timestamp=blog_schema.published_date,
        period_time_list=[
            sl.PeriodTime(timedelta(days=30)),  # Last month
            sl.PeriodTime(timedelta(days=90)),  # Last 3 months
            sl.PeriodTime(timedelta(days=365)),  # Last year
        ],
    )

    # Popularity (based on view count)
    popularity_space = sl.NumberSpace(
        number=blog_schema.view_count,
        min_value=0,
        max_value=10000,
        mode=sl.Mode.MAXIMUM,
    )

    # 3. Create Index
    blog_index = sl.Index(
        [content_space, title_space, category_space, recency_space, popularity_space]
    )

    # 4. Define Query with multiple weighted spaces
    blog_query = (
        sl.Query(
            blog_index,
            weights={
                content_space: sl.Param("content_weight"),
                title_space: sl.Param("title_weight"),
                category_space: sl.Param("category_weight"),
                recency_space: sl.Param("recency_weight"),
                popularity_space: sl.Param("popularity_weight"),
            },
        )
        .find(blog_schema)
        .similar(content_space.text, sl.Param("query_text"))
        .select(
            [
                blog_schema.title,
                blog_schema.content,
                blog_schema.category,
                blog_schema.published_date,
                blog_schema.view_count,
            ]
        )
        .limit(sl.Param("limit"))
    )

    # 5. Sample blog data
    from datetime import datetime

    # Convert datetime objects to unix timestamps (integers) as required by Timestamp schema field
    blog_posts = [
        {
            "id": "post1",
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is revolutionizing how we process data and make predictions.",
            "category": "technology",
            "published_date": int((datetime.now() - timedelta(days=5)).timestamp()),
            "view_count": 1500,
        },
        {
            "id": "post2",
            "title": "The Future of AI in Healthcare",
            "content": "Artificial intelligence is transforming medical diagnosis and treatment.",
            "category": "health",
            "published_date": int((datetime.now() - timedelta(days=15)).timestamp()),
            "view_count": 2300,
        },
        {
            "id": "post3",
            "title": "Business Analytics with Python",
            "content": "Learn how to use Python for business data analysis and visualization.",
            "category": "business",
            "published_date": int((datetime.now() - timedelta(days=45)).timestamp()),
            "view_count": 980,
        },
        {
            "id": "post4",
            "title": "Deep Learning Neural Networks",
            "content": "Understanding neural networks and their applications in modern AI.",
            "category": "technology",
            "published_date": int((datetime.now() - timedelta(days=2)).timestamp()),
            "view_count": 3200,
        },
    ]

    # Create source and executor
    source = sl.InMemorySource(schema=blog_schema)
    executor = sl.InMemoryExecutor(sources=[source], indices=[blog_index])
    app = executor.run()

    # Add data to the source after the app is running
    source.put(blog_posts)

    # 6. Create Retriever
    retriever = SuperlinkedRetriever(
        sl_client=app,
        sl_query=blog_query,
        page_content_field="content",
        metadata_fields=["title", "category", "published_date", "view_count"],
    )

    # 7. Demonstrate different weighting strategies
    scenarios = [
        {
            "name": "Content-focused search",
            "params": {
                "content_weight": 1.0,
                "title_weight": 0.3,
                "category_weight": 0.1,
                "recency_weight": 0.2,
                "popularity_weight": 0.1,
                "limit": 3,
            },
        },
        {
            "name": "Recent posts prioritized",
            "params": {
                "content_weight": 0.5,
                "title_weight": 0.2,
                "category_weight": 0.1,
                "recency_weight": 1.0,
                "popularity_weight": 0.1,
                "limit": 3,
            },
        },
        {
            "name": "Popular posts with category emphasis",
            "params": {
                "content_weight": 0.6,
                "title_weight": 0.3,
                "category_weight": 0.8,
                "recency_weight": 0.3,
                "popularity_weight": 0.9,
                "limit": 3,
            },
        },
    ]

    query_text = "machine learning and AI applications"

    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Query: '{query_text}'")

        results = retriever.invoke(query_text, **scenario["params"])

        for i, doc in enumerate(results, 1):
            print(
                f"  {i}. {doc.metadata['title']} (Category: {doc.metadata['category']}, Views: {doc.metadata['view_count']})"
            )

    print()


def example_3_ecommerce_product_search():
    """
    Example 3: E-commerce product search
    Use case: Product search with price range, brand preference, and ratings
    """
    print("=== Example 3: E-commerce Product Search ===")

    # 1. Define Schema
    class ProductSchema(sl.Schema):
        id: sl.IdField
        name: sl.String
        description: sl.String
        brand: sl.String
        price: sl.Float
        rating: sl.Float
        category: sl.String

    product_schema = ProductSchema()

    # 2. Define Spaces
    description_space = sl.TextSimilaritySpace(
        text=product_schema.description, model="sentence-transformers/all-MiniLM-L6-v2"
    )

    name_space = sl.TextSimilaritySpace(
        text=product_schema.name, model="sentence-transformers/all-MiniLM-L6-v2"
    )

    brand_space = sl.CategoricalSimilaritySpace(
        category_input=product_schema.brand,
        categories=["Apple", "Samsung", "Sony", "Nike", "Adidas", "Canon"],
    )

    category_space = sl.CategoricalSimilaritySpace(
        category_input=product_schema.category,
        categories=["electronics", "clothing", "sports", "photography"],
    )

    # Price space (lower prices get higher scores in MINIMUM mode)
    price_space = sl.NumberSpace(
        number=product_schema.price,
        min_value=10.0,
        max_value=2000.0,
        mode=sl.Mode.MINIMUM,  # Favor lower prices
    )

    # Rating space (higher ratings get higher scores)
    rating_space = sl.NumberSpace(
        number=product_schema.rating,
        min_value=1.0,
        max_value=5.0,
        mode=sl.Mode.MAXIMUM,  # Favor higher ratings
    )

    # 3. Create Index
    product_index = sl.Index(
        [
            description_space,
            name_space,
            brand_space,
            category_space,
            price_space,
            rating_space,
        ]
    )

    # 4. Define Query
    product_query = (
        sl.Query(
            product_index,
            weights={
                description_space: sl.Param("description_weight"),
                name_space: sl.Param("name_weight"),
                brand_space: sl.Param("brand_weight"),
                category_space: sl.Param("category_weight"),
                price_space: sl.Param("price_weight"),
                rating_space: sl.Param("rating_weight"),
            },
        )
        .find(product_schema)
        .similar(description_space.text, sl.Param("query_text"))
        .select(
            [
                product_schema.name,
                product_schema.description,
                product_schema.brand,
                product_schema.price,
                product_schema.rating,
                product_schema.category,
            ]
        )
        .limit(sl.Param("limit"))
    )

    # 5. Sample product data
    products = [
        {
            "id": "prod1",
            "name": "Wireless Bluetooth Headphones",
            "description": "High-quality wireless headphones with noise cancellation and long battery life.",
            "brand": "Sony",
            "price": 299.99,
            "rating": 4.5,
            "category": "electronics",
        },
        {
            "id": "prod2",
            "name": "Professional DSLR Camera",
            "description": "Full-frame DSLR camera perfect for professional photography and videography.",
            "brand": "Canon",
            "price": 1299.99,
            "rating": 4.8,
            "category": "photography",
        },
        {
            "id": "prod3",
            "name": "Running Shoes",
            "description": "Comfortable running shoes with excellent cushioning and support for athletes.",
            "brand": "Nike",
            "price": 129.99,
            "rating": 4.3,
            "category": "sports",
        },
        {
            "id": "prod4",
            "name": "Smartphone with 5G",
            "description": "Latest smartphone with 5G connectivity, advanced camera, and all-day battery.",
            "brand": "Samsung",
            "price": 899.99,
            "rating": 4.6,
            "category": "electronics",
        },
        {
            "id": "prod5",
            "name": "Bluetooth Speaker",
            "description": "Portable Bluetooth speaker with waterproof design and rich sound quality.",
            "brand": "Sony",
            "price": 79.99,
            "rating": 4.2,
            "category": "electronics",
        },
    ]

    # Create source and executor
    source = sl.InMemorySource(schema=product_schema)
    executor = sl.InMemoryExecutor(sources=[source], indices=[product_index])
    app = executor.run()

    # Add data to the source after the app is running
    source.put(products)

    # 6. Create Retriever
    retriever = SuperlinkedRetriever(
        sl_client=app,
        sl_query=product_query,
        page_content_field="description",
        metadata_fields=["name", "brand", "price", "rating", "category"],
    )

    # 7. Demonstrate different search strategies
    scenarios = [
        {
            "name": "Quality-focused search (high ratings matter most)",
            "query": "wireless audio device",
            "params": {
                "description_weight": 0.7,
                "name_weight": 0.5,
                "brand_weight": 0.2,
                "category_weight": 0.3,
                "price_weight": 0.1,
                "rating_weight": 1.0,
                "limit": 3,
            },
        },
        {
            "name": "Budget-conscious search (price matters most)",
            "query": "electronics device",
            "params": {
                "description_weight": 0.6,
                "name_weight": 0.4,
                "brand_weight": 0.1,
                "category_weight": 0.2,
                "price_weight": 1.0,
                "rating_weight": 0.3,
                "limit": 3,
            },
        },
        {
            "name": "Brand-focused search (brand loyalty)",
            "query": "sony products",
            "params": {
                "description_weight": 0.5,
                "name_weight": 0.3,
                "brand_weight": 1.0,
                "category_weight": 0.2,
                "price_weight": 0.2,
                "rating_weight": 0.4,
                "limit": 3,
            },
        },
    ]

    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Query: '{scenario['query']}'")

        results = retriever.invoke(scenario["query"], **scenario["params"])

        for i, doc in enumerate(results, 1):
            metadata = doc.metadata
            print(
                f"  {i}. {metadata['name']} ({metadata['brand']}) - ${metadata['price']} - ⭐{metadata['rating']}"
            )

    print()


def example_4_news_article_search():
    """
    Example 4: News article search with sentiment and topics
    Use case: News search with content, sentiment, topic categorization, and recency
    """
    print("=== Example 4: News Article Search ===")

    # 1. Define Schema
    class NewsArticleSchema(sl.Schema):
        id: sl.IdField
        headline: sl.String
        content: sl.String
        topic: sl.String
        sentiment_score: sl.Float  # -1 (negative) to 1 (positive)
        published_at: sl.Timestamp
        source: sl.String

    news_schema = NewsArticleSchema()

    # 2. Define Spaces
    content_space = sl.TextSimilaritySpace(
        text=news_schema.content, model="sentence-transformers/all-MiniLM-L6-v2"
    )

    headline_space = sl.TextSimilaritySpace(
        text=news_schema.headline, model="sentence-transformers/all-MiniLM-L6-v2"
    )

    topic_space = sl.CategoricalSimilaritySpace(
        category_input=news_schema.topic,
        categories=[
            "technology",
            "politics",
            "business",
            "sports",
            "entertainment",
            "science",
        ],
    )

    source_space = sl.CategoricalSimilaritySpace(
        category_input=news_schema.source,
        categories=["Reuters", "BBC", "CNN", "TechCrunch", "Bloomberg"],
    )

    # Sentiment space (can be configured to prefer positive or negative news)
    sentiment_space = sl.NumberSpace(
        number=news_schema.sentiment_score,
        min_value=-1.0,
        max_value=1.0,
        mode=sl.Mode.MAXIMUM,  # Default to preferring positive news
    )

    # Recency space
    recency_space = sl.RecencySpace(
        timestamp=news_schema.published_at,
        period_time_list=[
            sl.PeriodTime(timedelta(hours=6)),  # Last 6 hours
            sl.PeriodTime(timedelta(days=1)),  # Last day
            sl.PeriodTime(timedelta(days=7)),  # Last week
        ],
    )

    # 3. Create Index
    news_index = sl.Index(
        [
            content_space,
            headline_space,
            topic_space,
            source_space,
            sentiment_space,
            recency_space,
        ]
    )

    # 4. Define Query
    news_query = (
        sl.Query(
            news_index,
            weights={
                content_space: sl.Param("content_weight"),
                headline_space: sl.Param("headline_weight"),
                topic_space: sl.Param("topic_weight"),
                source_space: sl.Param("source_weight"),
                sentiment_space: sl.Param("sentiment_weight"),
                recency_space: sl.Param("recency_weight"),
            },
        )
        .find(news_schema)
        .similar(content_space.text, sl.Param("query_text"))
        .select(
            [
                news_schema.headline,
                news_schema.content,
                news_schema.topic,
                news_schema.sentiment_score,
                news_schema.published_at,
                news_schema.source,
            ]
        )
        .limit(sl.Param("limit"))
    )

    # 5. Sample news data
    # Convert datetime objects to unix timestamps (integers) as required by Timestamp schema field
    news_articles = [
        {
            "id": "news1",
            "headline": "Major Breakthrough in AI Research Announced",
            "content": "Scientists have developed a new artificial intelligence model that shows remarkable improvements in natural language understanding.",
            "topic": "technology",
            "sentiment_score": 0.8,
            "published_at": int((datetime.now() - timedelta(hours=2)).timestamp()),
            "source": "TechCrunch",
        },
        {
            "id": "news2",
            "headline": "Stock Market Faces Volatility Amid Economic Concerns",
            "content": "Financial markets experienced significant fluctuations today as investors react to new economic data and policy announcements.",
            "topic": "business",
            "sentiment_score": -0.3,
            "published_at": int((datetime.now() - timedelta(hours=8)).timestamp()),
            "source": "Bloomberg",
        },
        {
            "id": "news3",
            "headline": "New Climate Research Shows Promising Results",
            "content": "Recent studies indicate that innovative climate technologies are showing positive environmental impact and could help address climate change.",
            "topic": "science",
            "sentiment_score": 0.6,
            "published_at": int((datetime.now() - timedelta(hours=12)).timestamp()),
            "source": "Reuters",
        },
        {
            "id": "news4",
            "headline": "Tech Companies Report Strong Quarterly Earnings",
            "content": "Several major technology companies exceeded expectations in their quarterly earnings reports, driven by AI and cloud computing growth.",
            "topic": "technology",
            "sentiment_score": 0.7,
            "published_at": int((datetime.now() - timedelta(hours=4)).timestamp()),
            "source": "CNN",
        },
    ]

    # Create source and executor
    source = sl.InMemorySource(schema=news_schema)
    executor = sl.InMemoryExecutor(sources=[source], indices=[news_index])
    app = executor.run()

    # Add data to the source after the app is running
    source.put(news_articles)

    # 6. Create Retriever
    retriever = SuperlinkedRetriever(
        sl_client=app,
        sl_query=news_query,
        page_content_field="content",
        metadata_fields=[
            "headline",
            "topic",
            "sentiment_score",
            "published_at",
            "source",
        ],
    )

    # 7. Demonstrate different news search strategies
    print("Query: 'artificial intelligence developments'")

    # Recent technology news
    results = retriever.invoke(
        "artificial intelligence developments",
        content_weight=0.8,
        headline_weight=0.6,
        topic_weight=0.4,
        source_weight=0.2,
        sentiment_weight=0.3,
        recency_weight=1.0,  # Prioritize recent news
        limit=2,
    )

    print("\nRecent Technology News:")
    for i, doc in enumerate(results, 1):
        metadata = doc.metadata
        published_timestamp = metadata["published_at"]
        # Convert unix timestamp back to datetime for display calculation
        published_time = datetime.fromtimestamp(published_timestamp)
        hours_ago = (datetime.now() - published_time).total_seconds() / 3600
        sentiment = (
            "📈 Positive"
            if metadata["sentiment_score"] > 0
            else "📉 Negative"
            if metadata["sentiment_score"] < 0
            else "➡️ Neutral"
        )

        print(f"  {i}. {metadata['headline']}")
        print(f"     Source: {metadata['source']} | {sentiment} | {hours_ago:.1f}h ago")

    print()


def demonstrate_langchain_integration():
    """
    Example 5: Integration with LangChain RAG pipeline
    Shows how to use the SuperlinkedRetriever in a complete RAG workflow
    """
    print("=== Example 5: LangChain RAG Integration ===")

    # This would typically be used with an actual LLM
    # For demo purposes, we'll just show the retrieval part

    # Quick setup of a simple retriever
    class FAQSchema(sl.Schema):
        id: sl.IdField
        question: sl.String
        answer: sl.String
        category: sl.String

    faq_schema = FAQSchema()

    text_space = sl.TextSimilaritySpace(
        text=faq_schema.question, model="sentence-transformers/all-MiniLM-L6-v2"
    )

    category_space = sl.CategoricalSimilaritySpace(
        category_input=faq_schema.category,
        categories=["technical", "billing", "general", "account"],
    )

    faq_index = sl.Index([text_space, category_space])

    faq_query = (
        sl.Query(
            faq_index,
            weights={
                text_space: sl.Param("text_weight"),
                category_space: sl.Param("category_weight"),
            },
        )
        .find(faq_schema)
        .similar(text_space.text, sl.Param("query_text"))
        .select([faq_schema.question, faq_schema.answer, faq_schema.category])
        .limit(sl.Param("limit"))
    )

    # Sample FAQ data
    faqs = [
        {
            "id": "faq1",
            "question": "How do I reset my password?",
            "answer": "You can reset your password by clicking 'Forgot Password' on the login page and following the email instructions.",
            "category": "account",
        },
        {
            "id": "faq2",
            "question": "Why is my API not working?",
            "answer": "Check your API key, rate limits, and ensure you're using the correct endpoint URL.",
            "category": "technical",
        },
        {
            "id": "faq3",
            "question": "How do I upgrade my subscription?",
            "answer": "Visit the billing section in your account settings to upgrade your plan.",
            "category": "billing",
        },
    ]

    # Create source and executor
    source = sl.InMemorySource(schema=faq_schema)
    executor = sl.InMemoryExecutor(sources=[source], indices=[faq_index])
    app = executor.run()

    # Add data to the source after the app is running
    source.put(faqs)

    retriever = SuperlinkedRetriever(
        sl_client=app,
        sl_query=faq_query,
        page_content_field="answer",
        metadata_fields=["question", "category"],
    )

    # Simulate a RAG query
    user_question = "I can't access the API"

    print(f"User Question: '{user_question}'")
    print("Retrieving relevant context...")

    context_docs = retriever.invoke(
        user_question, text_weight=1.0, category_weight=0.3, limit=2
    )

    print("\nRetrieved Context:")
    for i, doc in enumerate(context_docs, 1):
        print(f"  {i}. Q: {doc.metadata['question']}")
        print(f"     A: {doc.page_content}")
        print(f"     Category: {doc.metadata['category']}")

    print(
        "\n[In a real RAG setup, this context would be passed to an LLM to generate a response]"
    )
    print()


def example_6_qdrant_vector_database():
    """
    Example 6: Same retriever with Qdrant vector database
    Use case: Production deployment with persistent vector storage

    This demonstrates that SuperlinkedRetriever is vector database agnostic.
    The SAME retriever code works with Qdrant (or Redis, MongoDB) by only
    changing the executor configuration, not the retriever implementation.
    """
    print("=== Example 6: Qdrant Vector Database ===")

    # 1. Define Schema (IDENTICAL to Example 1)
    class DocumentSchema(sl.Schema):
        id: sl.IdField
        content: sl.String

    doc_schema = DocumentSchema()

    # 2. Define Space and Index (IDENTICAL to Example 1)
    text_space = sl.TextSimilaritySpace(
        text=doc_schema.content, model="sentence-transformers/all-MiniLM-L6-v2"
    )

    doc_index = sl.Index([text_space])

    # 3. Define Query (IDENTICAL to Example 1)
    query = (
        sl.Query(doc_index)
        .find(doc_schema)
        .similar(text_space.text, sl.Param("query_text"))
        .select([doc_schema.content])
        .limit(sl.Param("limit"))
    )

    # 4. Configure Qdrant Vector Database (ONLY DIFFERENCE!)
    print("🔧 Configuring Qdrant vector database...")
    try:
        qdrant_vector_db = sl.QdrantVectorDatabase(
            url="https://your-qdrant-cluster.qdrant.io",  # Replace with your Qdrant URL
            api_key="your-api-key-here",  # Replace with your API key
            default_query_limit=10,
            vector_precision=sl.Precision.FLOAT16,
        )
        print(
            "✅ Qdrant configuration created (credentials needed for actual connection)"
        )
    except Exception as e:
        print(f"⚠️  Qdrant not configured (expected without credentials): {e}")
        print("📝 Using in-memory fallback for demonstration...")
        qdrant_vector_db = None

    # 5. Set up data and app (SLIGHT DIFFERENCE - vector database parameter)
    documents = [
        {
            "id": "doc1",
            "content": "Machine learning algorithms can process large datasets efficiently.",
        },
        {
            "id": "doc2",
            "content": "Natural language processing enables computers to understand human language.",
        },
        {
            "id": "doc3",
            "content": "Deep learning models require significant computational resources.",
        },
        {
            "id": "doc4",
            "content": "Data science combines statistics, programming, and domain expertise.",
        },
        {
            "id": "doc5",
            "content": "Artificial intelligence is transforming various industries.",
        },
    ]

    # Create source and executor with Qdrant (or fallback to in-memory)
    source = sl.InMemorySource(schema=doc_schema)

    if qdrant_vector_db:
        # Production setup with Qdrant
        executor = sl.InMemoryExecutor(
            sources=[source],
            indices=[doc_index],
            vector_database=qdrant_vector_db,  # 👈 This makes it use Qdrant!
        )
        storage_type = "Qdrant (persistent)"
    else:
        # Fallback to in-memory for demo
        executor = sl.InMemoryExecutor(sources=[source], indices=[doc_index])
        storage_type = "In-Memory (fallback)"

    app = executor.run()

    # Add data to the source after the app is running
    source.put(documents)

    # 6. Create Retriever (IDENTICAL CODE!)
    retriever = SuperlinkedRetriever(
        sl_client=app, sl_query=query, page_content_field="content"
    )

    # 7. Use the retriever (IDENTICAL CODE!)
    results = retriever.invoke("artificial intelligence and machine learning", limit=3)

    print(f"📊 Vector Storage: {storage_type}")
    print("🔍 Query: 'artificial intelligence and machine learning'")
    print(f"📄 Found {len(results)} documents:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.page_content}")

    print(
        "\n✅ Key Insight: Same SuperlinkedRetriever code works with any vector database!"
    )
    print(
        "✅ Only executor configuration changes, retriever implementation stays identical"
    )
    print("✅ Switch between in-memory → Qdrant → Redis → MongoDB without code changes")
    print()


def main():
    """
    Run all examples to demonstrate the flexibility of SuperlinkedRetriever
    """
    print("SuperlinkedRetriever Examples")
    print("=" * 50)
    print("This file demonstrates how the SuperlinkedRetriever can be used")
    print("with different space configurations for various use cases.\n")

    try:
        example_1_simple_text_search()
        example_2_multi_space_blog_search()
        example_3_ecommerce_product_search()
        example_4_news_article_search()
        demonstrate_langchain_integration()
        example_6_qdrant_vector_database()

        print("🎉 All examples completed successfully!")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Make sure you have 'superlinked' package installed:")
        print("pip install superlinked")


if __name__ == "__main__":
    main()
