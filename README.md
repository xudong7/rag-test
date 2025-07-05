# RAG Test

## Overview

This project demonstrates a simple Retrieval-Augmented Generation (RAG) system using a text embedding model. It includes functionality to embed text, store embeddings in a database, and query the database for relevant chunks based on a question.

## Getting Started

To get started with this project, you need to set up your environment and install the required dependencies. Follow the steps below:

### Clone the repository:

```bash
git clone <repository-url>
cd rag-test
```

### Create a virtual environment and activate it:

```bash
uv init . -p ${PYTHON_VERSION}
```

### Use uv to install the required dependencies:

```bash
uv sync
```

### Set up the environment variables:

create the `.env` file and configure the following variables:

```bash
# .env
GOOGLE_API_KEY=AIzaSyB3xpW3loC76XYjcAe4GMKJRCkyZxBwtQ0
```

### Run the embedding script to process your text data:

```bash
uv run embed.py
```

P.S. you can replace `data.md` with your own text file and change question in `embed.py` to test with different queries.

## Usage

Just for test purposes, the `embed.py` script will:

1. Load the text data from `data.md`.
2. Split the text into chunks and embed each chunk using the text embedding model.
3. Store the embeddings in the database.
4. Allow you to query the database with a question and retrieve relevant chunks.

also, you can write your own `chunk` function to customize how the text is split into chunks.
