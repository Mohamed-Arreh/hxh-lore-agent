# HxH Lore Agent

An AI agent that answers questions about *Hunter × Hunter* lore. It's a tool-calling agent built with Python, LangChain, and LangGraph, backed by a Chroma vector database for retrieval (RAG). The point of it is that the model picks which tool to use on its own depending on what you ask.

## What it does

It has two tools:

- **Ask a lore question** : general questions about the world, Nen, characters, how things work. It searches the lore and answers from what it finds, so it doesn't make stuff up.
- **Build a character dossier** : give it a character name and it pulls together a profile (overview, abilities, affiliations, key events) from everything in the lore about them.

You don't tell it which tool to use. Ask "what is Nen?" and it runs a search; ask "give me a dossier on Killua" and it builds the profile. The model reads the question and routes it. That's the agent part.

## How it works

Two stages.

First, ingestion (`ingest.py`): it reads the lore text files, splits them into overlapping chunks, embeds each chunk with OpenAI's `text-embedding-3-small`, and stores the vectors in a Chroma database on disk.

Then, querying (`search.py`): when a question comes in, it gets embedded with the same model (has to be the same one or the similarity scores are meaningless), Chroma finds the closest chunks, and those get handed to Claude with an instruction to answer only from what was retrieved. Both the Q&A and the dossier are wrapped as tools, and a LangGraph agent handles picking between them.

`main.py` puts a FastAPI server in front of all this with a `POST /ask` endpoint.

## Stack

Python, FastAPI, LangChain + LangGraph, Chroma for the vector store, OpenAI for embeddings, Claude (`claude-opus-4-8`) for the answers.

## Running it

You'll need Python, an OpenAI key, and an Anthropic key.

```bash
git clone https://github.com/Mohamed-Arreh/hxh-lore-agent.git
cd hxh-lore-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Make a `.env` file with your keys:
