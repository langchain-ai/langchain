# Deployments

So you've made a really cool chain - now what? How do you deploy it and make it easily sharable with the world?

This section covers several options for that.
Note that these are meant as quick deployment options for prototypes and demos, and not for production systems.
If you are looking for help with deployment of a production system, please contact us directly.

What follows is a list of template GitHub repositories aimed that are intended to be
very easy to fork and modify to use your chain.
This is far from an exhaustive list of options, and we are EXTREMELY open to contributions here.

## [Streamlit](https://github.com/hwchase17/langchain-streamlit-template)

This repo serves as a template for how to deploy a LangChain with Streamlit.
It implements a chatbot interface.
It also contains instructions for how to deploy this app on the Streamlit platform.

## [Gradio (on Hugging Face)](https://github.com/hwchase17/langchain-gradio-template)

This repo serves as a template for how deploy a LangChain with Gradio.
It implements a chatbot interface, with a "Bring-Your-Own-Token" approach (nice for not wracking up big bills).
It also contains instructions for how to deploy this app on the Hugging Face platform.
This is heavily influenced by James Weaver's [excellent examples](https://huggingface.co/JavaFXpert).

## [Beam](https://github.com/slai-labs/get-beam/tree/main/examples/langchain-question-answering)

This repo serves as a template for how deploy a LangChain with [Beam](https://beam.cloud).

It implements a Question Answering app and contains instructions for deploying the app as a serverless REST API.

## [Vercel](https://github.com/homanp/vercel-langchain)

A minimal example on how to run LangChain on Vercel using Flask.


## [SteamShip](https://github.com/steamship-core/steamship-langchain/)
This repository contains LangChain adapters for Steamship, enabling LangChain developers to rapidly deploy their apps on Steamship.
This includes: production ready endpoints, horizontal scaling across dependencies, persistant storage of app state, multi-tenancy support, etc.
