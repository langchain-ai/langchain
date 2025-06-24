# Project Tasks for RAG Examples

## 1. Build Ingestion Scripts
- [x] Parameterize vector store choice (Chroma, Elasticsearch, Weaviate)
- [x] Accept CLI args for source and persist dirs
- [x] Create logging output of number of documents ingested

## 2. Web Interface
- [x] FastAPI service exposing question-answering endpoint
- [x] Environment variable to configure vector store type
- [x] Document how to query the endpoint with curl

## 3. DevOps Integration (Future Work)
- [ ] Plan integration with Jira, GitHub, GitLab
- [ ] Document steps for Confluence and Bitbucket usage
- [ ] Setup deployment under cloudcurio.cc domain

Completion criteria: each item should have example commands, configuration files, or documentation proving functionality.
