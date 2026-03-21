from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict
from urllib.parse import parse_qs, urlparse

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from pydantic import BaseModel
from pytube import Playlist, YouTube
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi


PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "youtube_notes"
TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"

note_llm = ChatOllama(model="llama3.2:3b", temperature=0)
chat_llm = ChatOllama(model="qwen2.5:7b", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
vector_db = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
)
splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=300)
index_html = TEMPLATE_PATH.read_text(encoding="utf-8")


class IngestState(TypedDict):
    url: str
    note_style: str
    video_id: str
    title: str
    transcript: str
    notes: str
    playlist_position: int
    status: str
    warning: str


class ChatRequest(BaseModel):
    question: str


app = FastAPI(title="Send Notes")


def to_text(result: object) -> str:
    content = getattr(result, "content", result)
    return content if isinstance(content, str) else str(content)


def extract_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    if host.endswith("youtu.be"):
        video_id = parsed.path.strip("/")
        return video_id or None

    if "youtube.com" not in host:
        return None

    if parsed.path == "/watch":
        return parse_qs(parsed.query).get("v", [None])[0]

    parts = parsed.path.strip("/").split("/")
    if len(parts) > 1 and parts[0] in {"shorts", "embed"}:
        return parts[1]

    return None


def is_playlist_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.path == "/playlist" or "list" in parse_qs(parsed.query)


def get_playlist_video_urls(url: str) -> list[str]:
    return list(Playlist(url).video_urls)


def get_video_title(url: str, fallback_id: str) -> str:
    try:
        return YouTube(url).title
    except Exception:
        return fallback_id


def get_transcript_text(video_id: str) -> str:
    try:
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        else:
            transcript_data = YouTubeTranscriptApi().fetch(video_id)
    except Exception as exc:
        message = str(exc).lower()
        if any(token in message for token in ("transcript", "subtitles", "caption", "disabled")):
            print("Transcripts not available")
            raise RuntimeError("Transcripts not available") from exc
        raise

    lines: list[str] = []
    for item in transcript_data:
        if isinstance(item, dict):
            lines.append(str(item.get("text", "")))
        else:
            lines.append(str(getattr(item, "text", "")))
    return " ".join(line for line in lines if line).strip()


def fetch_transcript_node(state: IngestState) -> IngestState:
    video_id = extract_video_id(state["url"])
    if not video_id:
        return {**state, "status": "failed", "warning": "Could not parse video id from URL"}

    try:
        transcript = get_transcript_text(video_id)
    except RuntimeError as exc:
        return {**state, "video_id": video_id, "status": "cancelled", "warning": str(exc)}
    except Exception as exc:
        return {
            **state,
            "video_id": video_id,
            "status": "failed",
            "warning": f"Failed to fetch transcript: {exc}",
        }

    if not transcript:
        print("Transcripts not available")
        return {
            **state,
            "video_id": video_id,
            "status": "cancelled",
            "warning": "Transcripts not available",
        }

    return {
        **state,
        "video_id": video_id,
        "title": get_video_title(state["url"], video_id),
        "transcript": transcript,
        "status": "transcript_ready",
        "warning": "",
    }


def generate_notes_node(state: IngestState) -> IngestState:
    if state.get("status") != "transcript_ready":
        return state

    chunks = splitter.split_text(state["transcript"])
    if not chunks:
        return {**state, "status": "failed", "warning": "Transcript split produced no text"}

    chunk_notes = []
    for chunk in chunks:
        chunk_notes.append(
            to_text(
                note_llm.invoke(
                    f"""
You are writing notes from a YouTube transcript.

Requested format/style:
{state['note_style']}

Rules:
- Keep factual fidelity to transcript.
- Use bullets.
- Include important definitions, steps, examples, and conclusions.

Transcript chunk:
{chunk}
""".strip()
                )
            )
        )

    notes = to_text(
        note_llm.invoke(
            f"""
Combine these chunk notes into one clean note document.

Requested format/style:
{state['note_style']}

Chunk notes:
{chr(10).join(chunk_notes)}
""".strip()
        )
    )
    return {**state, "notes": notes, "status": "notes_ready", "warning": ""}


def store_notes_node(state: IngestState) -> IngestState:
    if state.get("status") != "notes_ready":
        return state

    vector_db.add_documents(
        [
            Document(
                page_content=state["notes"],
                metadata={
                    "video_id": state["video_id"],
                    "title": state["title"],
                    "url": state["url"],
                    "note_style": state["note_style"],
                    "playlist_position": state["playlist_position"],
                },
            )
        ]
    )
    return {**state, "status": "stored", "warning": ""}


def build_ingest_graph():
    graph = StateGraph(IngestState)
    graph.add_node("fetch_transcript", fetch_transcript_node)
    graph.add_node("generate_notes", generate_notes_node)
    graph.add_node("store_notes", store_notes_node)
    graph.set_entry_point("fetch_transcript")
    graph.add_edge("fetch_transcript", "generate_notes")
    graph.add_edge("generate_notes", "store_notes")
    graph.add_edge("store_notes", END)
    return graph.compile()


ingest_graph = build_ingest_graph()


def ingest_one_video(url: str, note_style: str, playlist_position: int = 0) -> IngestState:
    return ingest_graph.invoke(
        {
            "url": url,
            "note_style": note_style,
            "video_id": "",
            "title": "",
            "transcript": "",
            "notes": "",
            "playlist_position": playlist_position,
            "status": "",
            "warning": "",
        }
    )


def build_source_item(title: str, url: str, playlist_position: int | str | None) -> dict:
    source = {"title": title, "url": url}
    if isinstance(playlist_position, int) and playlist_position > 0:
        source["playlist_position"] = playlist_position
    if isinstance(playlist_position, str) and playlist_position.isdigit():
        source["playlist_position"] = int(playlist_position)
    return source


def notes_from_docs(docs: list[Document]) -> list[dict]:
    notes: list[dict] = []
    for doc in docs:
        metadata = doc.metadata or {}
        notes.append(
            {
                "title": metadata.get("title", "unknown"),
                "url": metadata.get("url", ""),
                "playlist_position": metadata.get("playlist_position"),
                "content": doc.page_content,
            }
        )
    return notes


@tool
def get_notes_by_position(position: int) -> str:
    """Fetch full notes for an exact playlist position (1-based)."""
    if position < 1:
        return json.dumps({"status": "not_found", "position": position, "notes": []})

    result = vector_db.get(
        where={"playlist_position": position},
        include=["documents", "metadatas"],
    )
    documents = result.get("documents", []) or []
    metadatas = result.get("metadatas", []) or []

    notes: list[dict] = []
    for i, text in enumerate(documents):
        metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
        notes.append(
            {
                "title": metadata.get("title", "unknown"),
                "url": metadata.get("url", ""),
                "playlist_position": metadata.get("playlist_position"),
                "content": text,
            }
        )

    if not notes:
        return json.dumps({"status": "not_found", "position": position, "notes": []})
    return json.dumps({"status": "ok", "position": position, "notes": notes})


@tool
def search_notes(query: str, k: int = 4) -> str:
    """Search notes semantically and return full note text for relevant matches."""
    k = max(1, min(int(k), 8))
    docs = vector_db.similarity_search(query, k=k)
    return json.dumps({"status": "ok", "notes": notes_from_docs(docs)})


TOOLS = [get_notes_by_position, search_notes]
TOOLS_BY_NAME = {tool_item.name: tool_item for tool_item in TOOLS}
chat_with_tools = chat_llm.bind_tools(TOOLS)


def parse_tool_output(raw_output: object) -> dict:
    if isinstance(raw_output, dict):
        return raw_output
    if isinstance(raw_output, str):
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            return {}
    return {}


def collect_sources_from_notes(notes: list[dict], sources: list[dict], seen: set[str]) -> None:
    for note in notes:
        title = note.get("title", "unknown")
        url = note.get("url", "")
        key = f"{title}|{url}"
        if key in seen:
            continue
        seen.add(key)
        sources.append(build_source_item(title, url, note.get("playlist_position")))


def rag_answer(question: str) -> dict:
    messages = [
        SystemMessage(
            content="""
You are a RAG assistant for YouTube notes.
You must call a retrieval tool before answering.
- For specific position requests (for example: "4th video", "video number four"), use get_notes_by_position.
- For general questions, use search_notes.
- Use only tool results; do not invent facts.
""".strip()
        ),
        HumanMessage(content=question),
    ]
    sources: list[dict] = []
    seen_source_keys: set[str] = set()

    for _ in range(4):
        ai_message = chat_with_tools.invoke(messages)
        messages.append(ai_message)

        tool_calls = getattr(ai_message, "tool_calls", [])
        if not tool_calls:
            answer_text = to_text(ai_message).strip()
            return {"answer": answer_text or "I am not sure.", "sources": sources}

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            selected_tool = TOOLS_BY_NAME.get(tool_name)
            if selected_tool is None:
                tool_output = json.dumps({"status": "error", "message": f"Unknown tool {tool_name}"})
            else:
                try:
                    tool_output = selected_tool.invoke(tool_call.get("args", {}))
                except Exception as exc:
                    tool_output = json.dumps({"status": "error", "message": str(exc)})

            payload = parse_tool_output(tool_output)
            notes = payload.get("notes", [])
            if isinstance(notes, list):
                collect_sources_from_notes(notes, sources, seen_source_keys)

            if tool_name == "get_notes_by_position" and payload.get("status") == "not_found":
                position = payload.get("position")
                return {"answer": f"video {position} does not exist.", "sources": []}

            messages.append(
                ToolMessage(
                    content=tool_output if isinstance(tool_output, str) else json.dumps(tool_output),
                    tool_call_id=tool_call["id"],
                )
            )

    return {"answer": "I am not sure.", "sources": sources}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return index_html


@app.post("/ingest")
def ingest(
    link: str = Form(...),
    note_format: str = Form(...),
    custom_format: str = Form(""),
) -> dict:
    link = link.strip()
    if not link:
        raise HTTPException(status_code=400, detail="Link is required")

    note_style = custom_format.strip() if note_format == "custom" else note_format.strip()
    if note_format == "custom" and not note_style:
        raise HTTPException(status_code=400, detail="Custom format is required")

    if is_playlist_url(link):
        urls = get_playlist_video_urls(link)
        if not urls:
            raise HTTPException(status_code=400, detail="No videos found in playlist")

        stored = 0
        warnings: list[str] = []

        for index, video_url in enumerate(
            tqdm(urls, desc="Converting playlist to notes", unit="video"),
            start=1,
        ):
            try:
                result = ingest_one_video(video_url, note_style, playlist_position=index)
            except Exception as exc:
                warning = f"Video {index}: skipped ({exc})"
                print(warning)
                warnings.append(warning)
                continue

            if result.get("status") == "stored":
                stored += 1
                continue

            warning = f"Video {index}: skipped ({result.get('warning') or 'unknown reason'})"
            print(warning)
            warnings.append(warning)

        return {
            "mode": "playlist",
            "total_videos": len(urls),
            "stored": stored,
            "warnings": warnings,
        }

    result = ingest_one_video(link, note_style)
    if result.get("status") == "stored":
        return {
            "mode": "video",
            "stored": 1,
            "title": result.get("title"),
            "url": link,
        }

    if result.get("warning") == "Transcripts not available":
        raise HTTPException(status_code=422, detail="Transcripts not available")

    raise HTTPException(status_code=500, detail=result.get("warning", "Failed to ingest video"))


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    return rag_answer(question)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
