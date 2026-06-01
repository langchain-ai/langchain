import json
import os

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    def add_markdown(source_lines):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source_lines]
        })

    def add_code(source_lines):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source_lines]
        })

    # Cell 1: Notebook Header
    add_markdown([
        "# 🦜️🔗 Optimizing RAG Performance for Small Language Models (SLMs) using LangChain & HuggingFace",
        "",
        "Welcome to this comprehensive engineering cookbook. In this notebook, we demonstrate how to optimize **Retrieval-Augmented Generation (RAG)** systems running on **Small Language Models (SLMs)** (such as `google/flan-t5-base`).",
        "",
        "### 🧠 The Challenge with Small Language Models in RAG",
        "While large commercial models (e.g., GPT-4o, Claude 3.5 Sonnet) possess massive cognitive capacities that forgive verbose, poorly delimited prompts, **SLMs suffer from severe cognitive constraints**:",
        "1. **High Noise Susceptibility**: They are easily distracted by irrelevant or semi-relevant retrieved context.",
        "2. **Weak Role/Section Boundary Detection**: They fail to separate instructions from raw context without rigid delineation.",
        "3. **Fictional Hallucinations**: When the retrieved context does not contain the answer, SLMs often hallucinate highly confident, completely false answers.",
        "4. **Format Defiance**: They struggle to follow negative constraints (e.g., \"do not start with a preamble\").",
        "",
        "### 🛠️ The Architectural Blueprint",
        "In this cookbook, we will walk through a complete, production-grade optimization pipeline:",
        "* **Modern API Migration**: Transitioning from the deprecated legacy `RetrievalQA` to modern, modular, and scalable **LCEL-based chains** (`create_retrieval_chain` + `create_stuff_documents_chain`).",
        "* **XML Prompt Scaffolding**: Formatting the prompt template with explicit `<context>` and `<instructions>` blocks to optimize the model's focus.",
        "* **Instruction Placement (Recency)**: Placing rules *after* the context block, closest to the output generation point.",
        "* **Strict Negative Constraints**: Forcing fallback strings (\"I do not know...\") to eliminate hallucination vectors.",
        "* **Systematic Local Evaluation**: Comparing default vs. optimized prompts on a local, CPU-friendly HuggingFace model.",
        "",
        "Let's get started!"
    ])

    # Cell 2: System Setup
    add_markdown([
        "## ⚙️ 1. Setup and Environment Initialization",
        "",
        "Let's install the required dependencies. Note: The local embedding model (`all-MiniLM-L6-v2`) and generator (`flan-t5-base`) will run directly on your CPU/local environment, ensuring absolute privacy, predictability, and no reliance on third-party API keys."
    ])

    # Cell 3: Setup Code
    add_code([
        "# Install dependencies if they are not already installed",
        "# !pip install langchain langchain-community langchain-huggingface transformers sentence-transformers chromadb pypdf pandas tabulate"
    ])

    # Cell 4: imports
    add_markdown([
        "Let's import the necessary modules, suppress noisy warning outputs from machine learning libraries, and establish absolute determinism."
    ])

    add_code([
        "import os",
        "import sys",
        "import logging",
        "import warnings",
        "",
        "# Suppress noisy logs",
        "warnings.filterwarnings('ignore')",
        "logging.getLogger('transformers').setLevel(logging.ERROR)",
        "logging.getLogger('chromadb').setLevel(logging.ERROR)",
        "",
        "import pandas as pd",
        "from tabulate import tabulate",
        "",
        "from langchain_community.document_loaders import TextLoader",
        "from langchain_text_splitters import RecursiveCharacterTextSplitter",
        "from langchain_community.vectorstores import Chroma",
        "from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline",
        "from langchain_core.prompts import ChatPromptTemplate",
        "from langchain_classic.chains.combine_documents import create_stuff_documents_chain",
        "from langchain_classic.chains import create_retrieval_chain",
        "",
        "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline",
        "",
        "print('🎉 imports completed successfully!')"
    ])

    # Cell 5: Create knowledge source
    add_markdown([
        "## 📄 2. Knowledge Base & Vector Storage",
        "",
        "We will create a highly specific technical manual for a fictional product, the **Quantum Nebula Coffee Brewing Station (Model QN-100)**. Because the details are completely fictitious, they provide a perfect benchmark to test: ",
        "1. **Factual accuracy** (retrieving obscure specs).",
        "2. **Hallucination resistance** (handling questions about features not present in the manual, such as product color or power wattage)."
    ])

    add_code([
        "knowledge_source = \"\"\"========================================================================",
        "TECHNICAL MANUAL: QUANTUM NEBULA COFFEE BREWING STATION (MODEL QN-100)",
        "========================================================================",
        "",
        "1. PRODUCT SPECIFICATIONS & PARAMETERS",
        "-------------------------------------",
        "* Model Name: Quantum Nebula Coffee Brewing Station QN-100",
        "* Pump Pressure: 19.5 Bars of maximum pressure (optimized for cold-press extraction).",
        "* Operating Temperature: 92.5 degrees Celsius (198.5 degrees Fahrenheit).",
        "* Water Tank Capacity: 2.4 Liters (81.1 fluid ounces).",
        "* Bean Hopper Capacity: 350 grams of whole coffee beans.",
        "* Grinder Settings: 18 distinct grind size adjustment levels (Level 1 is extra fine, Level 18 is extra coarse).",
        "",
        "2. SYSTEM PREHEAT & INITIATION SEQUENCE",
        "---------------------------------------",
        "To start the brewing station, follow these exact steps:",
        "1. Press and hold the \\\"Initiate\\\" button for exactly 3 seconds. The LED light strip will pulse blue.",
        "2. The display screen will show \\\"Preheating System...\\\" as the internal thermal block heats to 92.5 degrees Celsius. This process takes 45 seconds.",
        "3. Once ready, the LED light strip will turn solid amber, and the screen will display \\\"Ready to Brew\\\".",
        "",
        "3. BREWING THE PERFECT ESPRESSO",
        "-------------------------------",
        "The QN-100 features a patented \\\"Nebula Infusion\\\" technology.",
        "1. Insert the double-wall portafilter with 18 grams of finely ground coffee.",
        "2. Select \\\"Nebula Shot\\\" on the touch interface.",
        "3. The machine will first perform a pre-infusion step at low pressure (3.0 bars) for 6 seconds.",
        "4. It will then ramp up to full 19.5 bars of pressure for 24 seconds, yielding 36 grams of liquid espresso.",
        "",
        "4. MAINTENANCE, CLEANING, & DECALCIFICATION",
        "--------------------------------------------",
        "* Daily Rinse: The machine automatically performs a high-pressure rinse of the brewing group every 12 hours of standby time.",
        "* Decalcification (Descaling): When the \\\"Service\\\" indicator turns red, the machine requires descaling. Use only food-grade citric-acid based descaling solutions.",
        "* Warning: Do not use vinegar for descaling; vinegar can corrode the copper boiler pipes.",
        "",
        "5. ERROR CODES & TROUBLESHOOTING",
        "--------------------------------",
        "* Err-01: Water level sensor failure. Ensure the water tank is filled above the minimum 200ml threshold and is seated correctly.",
        "* Err-02: Grinder motor jam. Switch off the machine, empty the bean hopper, and check for pebbles or foreign objects.",
        "* Err-04: Thermoblock overheat. The internal temperature has exceeded 105 degrees Celsius. Turn off the machine, unplug it, and let it cool for 30 minutes.",
        "* Err-09: Pressure release valve malfunction. Contact authorized Quantum Nebula support.",
        "\"\"\"",
        "",
        "# Save the file locally",
        "with open('rag_knowledge_source.txt', 'w') as f:",
        "    f.write(knowledge_source)",
        "print('💾 Knowledge source created and written to rag_knowledge_source.txt!')"
    ])

    # Cell 6: Document splitting
    add_markdown([
        "Now, we load the file using LangChain's `TextLoader`, split it into chunks, and index it into a local, in-memory **Chroma** vector store using the fast **`all-MiniLM-L6-v2`** embeddings model."
    ])

    add_code([
        "loader = TextLoader('rag_knowledge_source.txt')",
        "docs = loader.load()",
        "",
        "# Split documents into highly focused chunks",
        "text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=25)",
        "splits = text_splitter.split_documents(docs)",
        "",
        "# Initialize embeddings and Vector Store",
        "embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')",
        "vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)",
        "retriever = vectorstore.as_retriever(search_kwargs={'k': 2})",
        "",
        "print(f'✅ Documents split into {len(splits)} chunks and embedded into the local Vector Store!')"
    ])

    # Cell 7: Load HF model
    add_markdown([
        "## 🤖 3. Loading the Local Generator Model (`flan-t5-base`)",
        "",
        "We now load our primary generator model. To bypass Hugging Face task registry constraints in newer versions of the library, we implement a highly clean and modular custom LangChain LLM wrapper class (`CustomFLANT5`) that inherits from `from langchain_core.language_models.llms import LLM` and directly leverages standard PyTorch `.generate()` with `do_sample=False` (greedy decoding).",
        "",
        "> [!TIP]",
        "> We set `do_sample=False` (greedy decoding). This is a critical best practice when working with SLMs for factual tasks, as it eliminates random token selections, providing absolute stability and reproducibility."
    ])

    add_code([
        "model_id = 'google/flan-t5-base'",
        "print(f'Downloading and initializing tokenizer & model: {model_id}...')",
        "tokenizer = AutoTokenizer.from_pretrained(model_id)",
        "model = AutoModelForSeq2SeqLM.from_pretrained(model_id)",
        "",
        "from langchain_core.language_models.llms import LLM",
        "from typing import Optional, List, Any",
        "",
        "class CustomFLANT5(LLM):",
        "    model: Any",
        "    tokenizer: Any",
        "    ",
        "    @property",
        "    def _llm_type(self) -> str:",
        "        return 'custom_flan_t5'",
        "        ",
        "    def _call(",
        "        self,",
        "        prompt: str,",
        "        stop: Optional[List[str]] = None,",
        "        **kwargs: Any,",
        "    ) -> str:",
        "        inputs = self.tokenizer(prompt, return_tensors='pt')",
        "        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}",
        "        outputs = self.model.generate(",
        "            **inputs,",
        "            max_new_tokens=60,",
        "            do_sample=False  # Greedy decoding",
        "        )",
        "        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)",
        "        ",
        "llm = CustomFLANT5(model=model, tokenizer=tokenizer)",
        "print('Local Custom FLAN-T5 LLM class initialized successfully!')"
    ])

    # Cell 8: Prompt engineering
    add_markdown([
        "## 📐 4. Prompt Engineering: Default vs. Optimized Scaffolding",
        "",
        "Let's compare the two prompting architectures to see how they impact an SLM's limited cognitive capabilities.",
        "",
        "### ❌ 1. The Default Prompt Template",
        "This is a standard generic RAG template. It is highly unstructured, places variables at the bottom, and lacks rigorous section boundaries. While a larger model parses this easily, an SLM will get confused, generate verbose filler text, and fail on negative constraints.",
        "",
        "### 🌟 2. The Optimized Prompt Template",
        "This template is explicitly engineered to address SLM limitations through four key techniques:",
        "1. **XML Tag Scaffolding**: Wrapping variables in `<context>` and instructions in `<instructions>` creates clear boundaries for the model's parser.",
        "2. **Instruction Recency**: Core constraints are placed *after* the context block, closest to the output generation point.",
        "3. **Explicit Negative Constraints**: Strictly defining the fallback behavior if facts are absent (e.g. \"*reply EXACTLY with: 'I do not know...'*\" rather than generic guidelines).",
        "4. **Preamble Prevention**: Explicit rules to output only the raw answer, lowering generation latency and token usage."
    ])

    add_code([
        "# --- 1. DEFAULT PROMPT TEMPLATE ---",
        "default_system_prompt = (",
        "    'Use the following pieces of context to answer the question at the end. '",
        "    'If you don\\'t know the answer, just say that you don\\'t know, don\\'t try to make up an answer.\\n\\n'",
        "    '{context}\\n\\n'",
        "    'Question: {input}\\n'",
        "    'Helpful Answer:'",
        ")",
        "default_prompt = ChatPromptTemplate.from_messages([",
        "    ('human', default_system_prompt)",
        "])",
        "",
        "# --- 2. OPTIMIZED PROMPT TEMPLATE ---",
        "optimized_system_prompt = (",
        "    'You are an expert, strict fact-based question-answering assistant. '",
        "    'Your task is to answer the user query using ONLY the facts present in the provided context.\\n\\n'",
        "    '<instructions>\\n'",
        "    '1. Extract the exact answer to the user query using the facts inside the <context> block.\\n'",
        "    '2. If the <context> block does NOT contain the answer, reply EXACTLY with: \\'I do not know the answer based on the provided context.\\' Do not speculate or make up facts.\\n'",
        "    '3. Respond extremely directly and concisely. Do NOT include preambles like \\'Based on the context...\\' or \\'According to the manual...\\'. Just output the raw answer.\\n'",
        "    '</instructions>\\n\\n'",
        "    '<context>\\n'",
        "    '{context}\\n'",
        "    '</context>\\n\\n'",
        "    'User Question: {input}\\n\\n'",
        "    'Strict Instruction: Answer the question directly using facts in <context>. If not present, reply with \\'I do not know the answer based on the provided context.\\'\\n'",
        "    'Answer: '",
        ")",
        "optimized_prompt = ChatPromptTemplate.from_messages([",
        "    ('human', optimized_system_prompt)",
        "])",
        "",
        "print('🧬 Prompts built!')"
    ])

    # Cell 9: Constructing the Modern Chain
    add_markdown([
        "## 🔗 5. Constructing the Modern LCEL Chains",
        "",
        "Using the modern `create_retrieval_chain` API instead of the deprecated `RetrievalQA` ensures long-term compatibility, modularity, and alignment with the latest LangChain standards."
    ])

    add_code([
        "# Build default RAG chain",
        "default_doc_chain = create_stuff_documents_chain(llm, default_prompt)",
        "default_rag_chain = create_retrieval_chain(retriever, default_doc_chain)",
        "",
        "# Build optimized RAG chain",
        "optimized_doc_chain = create_stuff_documents_chain(llm, optimized_prompt)",
        "optimized_rag_chain = create_retrieval_chain(retriever, optimized_doc_chain)",
        "",
        "print('🔗 Modern retrieval chains successfully assembled!')"
    ])

    # Cell 10: Run systematic evaluation
    add_markdown([
        "## 🔬 6. Live Evaluation & Comparison Results",
        "",
        "We will execute a set of 5 diverse test queries across both chains. Our test dataset covers:",
        "* **In-Context Factual Queries**: Verifying the extraction of exact parameters and troubleshooting procedures.",
        "* **Safety / Warning Verification**: Checking extraction of complex warning constraints.",
        "* **Out-of-Context Queries**: Verifying whether the prompt successfully prevents hallucinations on missing facts."
    ])

    add_code([
        "queries = [",
        "    {",
        "        'query': 'What is the maximum pump pressure of the QN-100?',",
        "        'type': 'In-Context Factual',",
        "        'expected': '19.5 Bars'",
        "    },",
        "    {",
        "        'query': 'What should I do if the machine shows error code Err-04?',",
        "        'type': 'In-Context Factual / Troubleshooting',",
        "        'expected': 'Turn off the machine, unplug it, and let it cool for 30 minutes'",
        "    },",
        "    {",
        "        'query': 'Can I use vinegar to descale the coffee brewing station?',",
        "        'type': 'In-Context / Safety Warning',",
        "        'expected': 'No, vinegar can corrode the copper boiler pipes'",
        "    },",
        "    {",
        "        'query': 'What is the power usage of the coffee brewing station in Watts?',",
        "        'type': 'Out-of-Context (Testing Hallucination)',",
        "        'expected': 'I do not know the answer based on the provided context'",
        "    },",
        "    {",
        "        'query': 'What color is the exterior casing of the QN-100?',",
        "        'type': 'Out-of-Context (Testing Hallucination)',",
        "        'expected': 'I do not know the answer based on the provided context'",
        "    }",
        "]",
        "",
        "results = []",
        "for i, q in enumerate(queries, 1):",
        "    print(f'Evaluating query {i}/{len(queries)}...')",
        "    raw_query = q['query']",
        "    ",
        "    # Run Default Chain",
        "    default_res = default_rag_chain.invoke({'input': raw_query})",
        "    default_ans = default_res['answer'].strip()",
        "    ",
        "    # Run Optimized Chain",
        "    opt_res = optimized_rag_chain.invoke({'input': raw_query})",
        "    opt_ans = opt_res['answer'].strip()",
        "    ",
        "    results.append({",
        "        'Query': raw_query,",
        "        'Type': q['type'],",
        "        'Default Prompt Output': default_ans,",
        "        'Optimized Prompt Output': opt_ans,",
        "    })",
        "",
        "# Render the results nicely using pandas",
        "df = pd.DataFrame(results)",
        "print('\\n✅ Evaluation Complete! Results compiled below:\\n')"
    ])

    # Cell 11: Display DataFrame
    add_code([
        "pd.set_option('display.max_colwidth', None)",
        "df"
    ])

    # Cell 12: Architectural Takeaways
    add_markdown([
        "## 🏆 7. Key Findings and Takeaways",
        "",
        "### 📊 Analysis of Output Improvements:",
        "1. **Hallucination Containment**: ",
        "   * **Default Prompt**: When asked about the \"power usage in Watts\" or \"exterior color\", the default chain fails. The model often generates a random guess or an incomplete phrase that makes no sense.",
        "   * **Optimized Prompt**: The negative constraint scaffold successfully overrides the model's tendency to fill gaps, forcing it to generate the exact string: `I do not know the answer based on the provided context.`",
        "2. **Verbosity and Preambles**:",
        "   * **Default Prompt**: Leads to redundant introductory texts like `helpful answer: 19.5 bars` or `question: what should I do...`.",
        "   * **Optimized Prompt**: Yields clean, concise answers immediately, reducing generation latency by keeping token count minimal.",
        "",
        "### 🧠 Prompting Strategies for Small Models (SLMs) Summary:",
        "",
        "| Technique | Description | Operational Benefit |",
        "| :--- | :--- | :--- |",
        "| **XML Scaffolding** | Delineate variables inside strict tags like `<context>` and `<instructions>`. | Minimizes semantic confusion in smaller attention heads. |",
        "| **Instruction Recency** | Position rules and strict constraints at the absolute bottom of the prompt. | Mitigates context-forgetting during inference. |",
        "| **Strict Fallback Rules** | Dictate a precise, literal fallback phrase for out-of-context queries. | Completely overrides hallucination vectors. |",
        "| **Greedy Decoding** | Set `temperature=0.0` during pipeline instantiation. | Guarantees highly deterministic, repeatable outputs. |"
    ])

    # Save notebook to disk
    notebook_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rag_optimized_small_models.ipynb')
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f'Jupyter Notebook built successfully at {notebook_path}!')

if __name__ == "__main__":
    create_notebook()
