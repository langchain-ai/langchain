# Psychic

This page covers how to use [Psychic](https://www.psychic.dev/) within LangChain.

## What is Psychic?

Psychic is a platform for integrating with your customer’s SaaS tools like Notion, Zendesk, Confluence, and Google Drive via OAuth and syncing documents from these applications to your SQL or vector database. You can think of it like Plaid for unstructured data. Psychic is easy to set up - you use it by importing the react library and configuring it with your Sidekick API key, which you can get from the [Psychic dashboard](https://dashboard.psychic.dev/). When your users connect their applications, you can view these connections from the dashboard and retrieve data using the server-side libraries.

## Quick start

1. Create an account in the [dashboard](https://dashboard.psychic.dev/).
2. Use the [react library](https://docs.psychic.dev/sidekick-link) to add the Psychic link modal to your frontend react app. Users will use this to connect their SaaS apps.
3. Once your user has created a connection, you can use the langchain PsychicLoader by following the [example notebook](../modules/indexes/document_loaders/examples/psychic.ipynb)


# Advantages vs Other Document Loaders

1.	**Universal API:** Instead of building OAuth flows and learning the APIs for every SaaS app, you integrate Psychic once and leverage our universal API to retrieve data.
2.	**Data Syncs:** Data in your customers' SaaS apps can get stale fast. With Psychic you can configure webhooks to keep your documents up to date on a daily or realtime basis.
3.	**Simplified OAuth:** Psychic handles OAuth end-to-end so that you don't have to spend time creating OAuth clients for each integration, keeping access tokens fresh, and handling OAuth redirect logic.