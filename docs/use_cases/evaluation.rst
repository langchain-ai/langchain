Evaluation
==============

Generative models are notoriously hard to evaluate with traditional metrics. One new way of evaluating them is using language models themselves to do the evaluation. LangChain provides some prompts/chains for assisting in this.

The examples here all highlight how to use language models to assist in evaluation of themselves.

`Question Answering <./evaluation/question_answering.html>`_: An overview of LLMs aimed at evaluating question answering systems in general.

`Data Augmented Question Answering <./evaluation/data_augmented_question_answering.html>`_: An end-to-end example of evaluating a question answering system focused on a specific document (a VectorDBQAChain to be precise). This example highlights how to use LLMs to come up with question/answer examples to evaluate over, and then highlights how to use LLMs to evaluate performance on those generated examples.

`Hugging Face Datasets <./evaluation/huggingface_datasets.html>`_: Covers an example of loading and using a dataset from Hugging Face for evaluation.


.. toctree::
   :maxdepth: 1
   :glob:
   :hidden:

   evaluation/*
