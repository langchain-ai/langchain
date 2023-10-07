# Portkey

**Portkey** is a full-stack LLMOps platform that productionizes your Gen AI app reliably and securely.

#### Key Features of Portkey's Integration with Langchain:

<img src="https://portkey.ai/blog/content/images/2023/09/header.png" alt="header" width=600 />


1. **🚪 AI Gateway**:
    - **[Automated Fallbacks & Retries](#🔁-implementing-fallbacks-and-retries-with-portkey)**: Ensure your application remains functional even if a primary service fails.
    - **[Load Balancing](#⚖️-implementing-load-balancing-with-portkey)**: Efficiently distribute incoming requests among multiple models.
    - **[Semantic Caching](#🧠-implementing-semantic-caching-with-portkey)**: Reduce costs and latency by intelligently caching results.
2. **[🔬 Observability](#🔬-observability-with-portkey)**:
    - **Logging**: Keep track of all requests for monitoring and debugging.
    - **Requests Tracing**: Understand the journey of each request for optimization.
    - **Custom Tags**: Segment and categorize requests for better insights.
3. **[📝 Continuous Improvement with User Feedback](#📝-feedback-with-portkey)**:
    - **Feedback Collection**: Seamlessly gather feedback on any served request, be it on a generation or conversation level.
    - **Weighted Feedback**: Obtain nuanced information by attaching weights to user feedback values.
    - **Feedback Metadata**: Incorporate custom metadata with the feedback to provide context, allowing for richer insights and analyses.
4. **[🔑 Secure Key Management](#feedback-with-portkey)**:
    - **Virtual Keys**: Portkey transforms original provider keys into virtual keys, ensuring your primary credentials remain untouched.
    - **Multiple Identifiers**: Ability to add multiple keys for the same provider or the same key under different names for easy identification without compromising security.

To harness these features, check out this Colab notebook:

<a href="https://colab.research.google.com/drive/19LFG4az9j-0pUixiVhhtHLpmLSZlz1Ei?usp=sharing" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt=\"Open In Colab\" width=150 />
</a>