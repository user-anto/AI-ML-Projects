# Take Notes

Take Notes is an app that turns YouTube videos and playlists into searchable study notes, stores them in a vector database, and provides a RAG chatbot grounded on those notes.

This project is built for practical, production-style workflows using FastAPI, LangGraph, LangChain, Ollama, and Chroma.

## Why This Project

The motivation for this project was a personal pain-point: Long-form video content is hard to revisit quickly.

It provides a full pipeline:
- ingest YouTube content (video or playlist)
- generate structured notes in your preferred style
- persist notes in a vector DB
- answer follow-up questions with source-aware retrieval

## Core Features

- Video + playlist ingestion
- Transcript-only processing (no transcript => skip/cancel)
- Note generation with `llama3.2:3b`
- Persistent vector storage in Chroma (`./chroma_db`)
- Tool-based RAG chat with `qwen2.5:7b`
- Specific video position retrieval (with fallback `video {n} does not exist.`)
- Simple, polished web UI

## Tech Stack

- FastAPI (backend + API routes)
- LangGraph (ingestion workflow graph)
- LangChain + LangChain tools (tool-calling retrieval orchestration)
- Ollama (`llama3.2:3b`, `qwen2.5:7b`, `nomic-embed-text:v1.5`)
- Chroma (persistent vector database)

## Architecture Overview

### Ingestion Flow (LangGraph)

1. Parse YouTube URL and detect video ID
2. Fetch transcript
3. Generate notes chunk-by-chunk
4. Merge chunk notes
5. Store notes + metadata in Chroma

Stored metadata includes:
- `video_id`
- `title`
- `url`
- `note_style`
- `playlist_position` (for playlist items)

### Chat Flow (Tool-Based RAG)

The chat model is bound to two tools:
- `get_notes_by_position(position: int)`
- `search_notes(query: str, k: int = 4)`

The model decides which retrieval tool to call, then answers using tool output only.

## Project Structure

```text
.
├── main.py
├── requirements.txt
├── templates/
│   └── index.html
└── chroma_db/               # created at runtime
```

## Prerequisites

- Python 3.12+
- Ollama installed and running

Pull required models:

```bash
ollama pull llama3.2:3b
ollama pull qwen2.5:7b
ollama pull nomic-embed-text:v1.5
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload
```

Open: `http://127.0.0.1:8000`

## API Endpoints

### `POST /ingest`
Ingest one video or an entire playlist and generate/store notes.

Form fields:
- `link` (string, required)
- `note_format` (`short | medium | detailed | custom`)
- `custom_format` (required only when `note_format=custom`)

Example:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -F "link=https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
  -F "note_format=medium"
```

Possible response (video):

```json
{
  "mode": "video",
  "stored": 1,
  "title": "...",
  "url": "https://www.youtube.com/watch?v=..."
}
```

Possible response (playlist):

```json
{
  "mode": "playlist",
  "total_videos": 12,
  "stored": 10,
  "warnings": ["Video 4: skipped (...)"]
}
```

### `POST /chat`
Ask questions over stored notes.

Request:

```bash
curl -X POST http://127.0.0f.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize the key ideas from video 3"}'
```

Response:

```json
{
  "answer": "...",
  "sources": [
    {
      "title": "...",
      "url": "https://www.youtube.com/watch?v=...",
      "playlist_position": 3
    }
  ]
}
```

## Notes on Retrieval Behavior

- If a transcript is unavailable, ingestion for that video is canceled.
- For explicit video-position requests, the app attempts direct position lookup.
- If requested position does not exist, response is:
  - `video {n} does not exist.`

## Portfolio Highlights

This project demonstrates:
- LLM app architecture: ingest -> transform -> persist -> retrieve -> answer
- orchestration with LangGraph
- tool-calling RAG instead of prompt-only retrieval
- local model serving with Ollama

## Future Improvements

- Add streaming token responses for chat
- Add asynchronous/background playlist ingestion jobs