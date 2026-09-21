# Paperless-Ngx RAG

## Table of Contents

+ [About](#about)
+ [Getting Started](#getting_started)
+ [Usage](#usage)

## About <a name = "about"></a>

A small project to pull documents from a paperless instance to store in a vector database. Retrieves information from the database to answer question based on facts grounded in your documents citing the source document ID(s). Uses OpenAI-compatible API's for an embedding model, an optional reranking model and the LLM Chat model that formulates the final answer.

## Features

* Ask questions in natural language and get answers containing source document ID(s)
* Pre-Filter documents based on
  * document type
  * tags
  * date created
  * date modified
* The metadata is read from the paperless-ngx instance and stored together with the document content inside the vector database

## Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Things you need to make use of the project:

* Have a running instance of [paperless-ngx](https://github.com/paperless-ngx/paperless-ngx)
  * requires a user configured for api access with permissions to view documents, document types, and tags that should be ingested by the RAG system
* (Optional) Have a running [Qdrant]([github.com/qdrant/qdrant](https://github.com/qdrant/qdrant)) instance
  * can use a built-in ChromaDB instead
* [uv]([docs.astral.sh/uv](https://docs.astral.sh/uv/)) python package manager installed
* Access to the following models served over an OpenAI-compatible API
  * embedding model
  * LLM chat model
  * (optional) reranking model
* You need API urls and tokens to access the models

### Installing

Clone the repository files to a local directory of your choice

```Shell
# install the project dependencies
git clone https://github.com/Raimon-Gramlich/paperless_rag.git
cd paperless_rag
```

Install the project dependencies

```Shell
uv sync
```

Create a .env file in the project directory with the following keys

* Required
  * `PAPERLESS_API_URL`
  * `PAPERLESS_API_TOKEN`
* Recommended
  * `LOCAL_VLLM_BASE_URL`
  * `LOCAL_VLLM_API_KEY`
  * `LOCAL_EMBEDDING_MODEL_NAME`
  * `LOCAL_LLM_MODEL_NAME`
  * `LOCAL_RERANKING_URL`
  * `LOCAL_RERANKING_MODEL_NAME`
* Inside the app the url and tokens can be entered separately for each model

## Usage <a name = "usage"></a>

Run the streamlit app using

```Shell
uv run streamlit run Main.py
```
