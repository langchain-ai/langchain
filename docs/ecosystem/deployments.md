# Deployments

So, you've created a really cool chain - now what? How do you deploy it and make it easily shareable with the world?

This section covers several options for that. Note that these options are meant for quick deployment of prototypes and demos, not for production systems. If you need help with the deployment of a production system, please contact us directly.

What follows is a list of template GitHub repositories designed to be easily forked and modified to use your chain. This list is far from exhaustive, and we are EXTREMELY open to contributions here.

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

## [FastAPI + Vercel](https://github.com/msoedov/langcorn)

A minimal example on how to run LangChain on Vercel using FastAPI and LangCorn/Uvicorn.

## [Kinsta](https://github.com/kinsta/hello-world-langchain)

A minimal example on how to deploy LangChain to [Kinsta](https://kinsta.com) using Flask.

## [Fly.io](https://github.com/fly-apps/hello-fly-langchain)

A minimal example of how to deploy LangChain to [Fly.io](https://fly.io/) using Flask.

## [Digitalocean App Platform](https://github.com/homanp/digitalocean-langchain)

A minimal example on how to deploy LangChain to DigitalOcean App Platform.

## [Google Cloud Run](https://github.com/homanp/gcp-langchain)

A minimal example on how to deploy LangChain to Google Cloud Run.

## [SteamShip](https://github.com/steamship-core/steamship-langchain/)

This repository contains LangChain adapters for Steamship, enabling LangChain developers to rapidly deploy their apps on Steamship. This includes: production-ready endpoints, horizontal scaling across dependencies, persistent storage of app state, multi-tenancy support, etc.

## [Langchain-serve](https://github.com/jina-ai/langchain-serve)

This repository allows users to serve local chains and agents as RESTful, gRPC, or WebSocket APIs, thanks to [Jina](https://docs.jina.ai/). Deploy your chains & agents with ease and enjoy independent scaling, serverless and autoscaling APIs, as well as a Streamlit playground on Jina AI Cloud.

## [BentoML](https://github.com/ssheng/BentoChain)

This repository provides an example of how to deploy a LangChain application with [BentoML](https://github.com/bentoml/BentoML). BentoML is a framework that enables the containerization of machine learning applications as standard OCI images. BentoML also allows for the automatic generation of OpenAPI and gRPC endpoints. With BentoML, you can integrate models from all popular ML frameworks and deploy them as microservices running on the most optimal hardware and scaling independently.

## [Databutton](https://databutton.com/home?new-data-app=true)

These templates serve as examples of how to build, deploy, and share LangChain applications using Databutton. You can create user interfaces with Streamlit, automate tasks by scheduling Python code, and store files and data in the built-in store. Examples include a Chatbot interface with conversational memory, a Personal search engine, and a starter template for LangChain apps. Deploying and sharing is just one click away.
